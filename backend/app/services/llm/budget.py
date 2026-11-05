"""One durable budget for every physical model request, across retries and threads.

Quota reservation happens before HTTP. Unknown/failed attempts retain their
token reservation; calls are never refunded. No prompts or credentials enter
the ledger. Missing scope/storage fails closed rather than creating new quotas.
"""
from contextlib import contextmanager,asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass,field
from datetime import datetime,timedelta,timezone
from hashlib import sha256
import json
import time
import asyncio
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import select,update,text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core import clock
from app.core.config import settings
from app.db.model_budget_models import ModelQuota,ModelCallLedger,ModelResultCache


class ModelBudgetError(RuntimeError):
    def __init__(self,code):self.code=code;super().__init__(code)


class ModelBudgetUnavailable(ModelBudgetError):pass


class ModelTaskBudgetError(ModelBudgetError):
    """Only the current semantic task is exhausted; unrelated tasks continue."""


class ModelPrepareBudgetError(ModelBudgetError):
    """The fixed per-request preparation identity is exhausted."""


def is_budget_policy_stop(exc):
    """True only for a real budget/quota stop, including a wrapped cause.

    Shared by legacy and ReAct paths so neither has to import the other.
    """
    cause = exc.__cause__ if isinstance(getattr(exc, "__cause__", None), BaseException) else None
    for item in (exc, cause):
        if item is None:
            continue
        if isinstance(item, ModelBudgetError):
            return True
        category = str(getattr(getattr(item, "category", None), "name", None) or "")
        if category == "POLICY":
            return True
    return False


def is_task_budget_stop(exc):
    """True only when the stop belongs to a user semantic task.

    Preparation, root/deadline/cancel, daily and storage failures stay global.
    """
    cause = exc.__cause__ if isinstance(getattr(exc, "__cause__", None), BaseException) else None
    return any(isinstance(item, ModelTaskBudgetError) for item in (exc, cause))


def key(value):return sha256(str(value).encode('utf-8')).hexdigest()


def quota_lock_order(scope_id, quota_ids):
    """One stable lock/settlement order for reserve and finish.

    Root first, then the task/preparation child, then every shared daily/
    background/stage row in lexicographic order. Old ledgers stored a fully
    sorted id list; reconstruction moves only the root/task rows and keeps the
    shared rows stable, so legacy settlement and new policy never form a
    reserve/finish cycle.
    """
    ordered=[]
    for value in quota_ids or []:
        if isinstance(value,str) and value and value not in ordered:
            ordered.append(value)
    root=[scope_id] if scope_id in ordered else []
    tasks=sorted(value for value in ordered
                 if value.startswith("task:") or value.startswith("prepare:"))
    shared=sorted(value for value in ordered if value not in root and value not in tasks)
    return root+tasks+shared


def utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def daily_window(moment):
    local=moment.astimezone(ZoneInfo('Asia/Shanghai'))
    midnight=datetime.combine(local.date(),datetime.min.time(),tzinfo=local.tzinfo).astimezone(timezone.utc)
    end=datetime.combine(local.date()+timedelta(days=1),datetime.min.time(),tzinfo=local.tzinfo).astimezone(timezone.utc)
    return 'daily:'+key(midnight.isoformat()),midnight,end


def daily_extra_calls(session,moment):
    identifier,start,end=daily_window(moment)
    grant=session.get(ModelQuota,'grant:'+identifier)
    if grant is None:return 0
    if (grant.kind!='daily_grant' or utc(grant.expires_at)!=end
            or not start<=utc(grant.created_at)<=moment
            or not 1<=grant.max_calls<=10000 or grant.calls!=0
            or grant.tokens!=0 or grant.max_tokens!=0):
        raise ModelBudgetUnavailable('invalid_daily_grant')
    return grant.max_calls


BUDGET_STORAGE_TABLES=(ModelQuota.__tablename__,ModelCallLedger.__tablename__,ModelResultCache.__tablename__)


def probe_budget_storage():
    """Read-only availability/structure check for the durable budget tables.

    Returns ``(True, None)`` when all three tables exist and every mapped
    column can be selected. It never issues DDL or writes, so callers can run
    it before starting consumers. On PostgreSQL the probe runs in its own
    transaction with bounded lock/statement timeouts: a concurrent migration
    or backup holding a table lock becomes ``budget_storage_unavailable``
    instead of hanging startup. ``SET LOCAL`` scopes the limits to this
    transaction, so pooled connections are not affected afterwards. Failure
    reasons are safe enums.
    """
    from sqlalchemy import inspect as sqlalchemy_inspect
    from app.db import session as db
    if db.engine is None or db.SessionLocal is None:
        return False,'budget_storage_unconfigured'
    try:
        with db.engine.connect() as connection:
            with connection.begin():
                if connection.dialect.name=='postgresql':
                    connection.execute(text("SET LOCAL lock_timeout='2000ms'"))
                    connection.execute(text("SET LOCAL statement_timeout='5000ms'"))
                inspector=sqlalchemy_inspect(connection)
                if any(not inspector.has_table(name) for name in BUDGET_STORAGE_TABLES):
                    return False,'budget_tables_missing'
                for model in (ModelQuota,ModelCallLedger,ModelResultCache):
                    connection.execute(select(model).limit(0))
        return True,None
    except Exception:
        return False,'budget_storage_unavailable'


active_scope=ContextVar('paid_model_scope',default=None)
active_stages=ContextVar('paid_model_stages',default=())
active_task=ContextVar('paid_model_task',default=None)
store_override=ContextVar('paid_model_store_override',default=None)

POLICY_KIND='budget_policy'
TASK_POLICY='per_task'


@dataclass
class TaskBudget:
    """Durable identity and frozen allowance of one semantic user goal.

    ``quota_id`` is derived from the owner, root request identity and the
    server-frozen source goal. Renaming or retyping the task keeps the same id;
    the stored row can only be tightened, never upgraded.
    """

    quota_id:str
    kind:str
    goal:str
    max_calls:int
    max_tokens:int
    root_quota_id:str
    task_type:str=''
    limited:bool=False
    exhausted_code:str=''


def prepare_budget(scope):
    """The one fixed server-preparation identity for a root request."""
    if scope.owner_id is None:
        raise ModelBudgetUnavailable('model_task_owner_required')
    from app.core.config import settings
    return TaskBudget(
        quota_id='prepare:'+key(str(scope.owner_id)+'|'+scope.quota_id+'|prepare-v1'),
        kind='prepare',goal='prepare',root_quota_id=scope.quota_id,
        max_calls=settings.model_prepare_max_calls,
        max_tokens=settings.model_task_max_tokens)


class BudgetStore:
    def __init__(self,factory=None,*,enforce_daily_limits=True):
        # Only an explicitly supplied isolated factory may waive the daily
        # call/token ceilings. Every other limit and all ledger writes stay.
        if type(enforce_daily_limits) is not bool:
            raise ValueError('enforce_daily_limits_must_be_bool')
        if enforce_daily_limits is False and factory is None:
            raise ValueError('daily_enforcement_requires_explicit_factory')
        self.factory=factory
        self.enforce_daily_limits=enforce_daily_limits

    def session(self):
        factory=self.factory
        if factory is None:
            from app.db import session as db
            factory=db.SessionLocal
        if factory is None:raise ModelBudgetUnavailable('model_budget_storage_unavailable')
        return factory()

    @staticmethod
    def prepare(session):
        if session.get_bind().dialect.name=='postgresql':
            session.execute(text("SET LOCAL lock_timeout='2000ms'"))
            session.execute(text("SET LOCAL statement_timeout='5000ms'"))

    @staticmethod
    def insert(session,model,values):
        dialect=session.get_bind().dialect.name
        if dialect not in ('sqlite','postgresql'):raise ModelBudgetUnavailable('unsupported_budget_storage')
        statement=(sqlite_insert if dialect=='sqlite' else pg_insert)(model).values(**values)
        session.execute(statement.on_conflict_do_nothing())

    def daily_status(self):
        moment=clock.utc_now();identifier,_,end=daily_window(moment)
        with self.session() as session:
            self.prepare(session)
            extra=daily_extra_calls(session,moment)
            row=session.get(ModelQuota,identifier)
            ceiling=settings.model_daily_max_calls+extra
            limit=min(row.max_calls,ceiling) if row else settings.model_daily_max_calls
            calls=row.calls if row else 0
            return dict(calls=calls,max_calls=limit,remaining_calls=max(0,limit-calls),
                        extra_calls=extra,expires_at=end.isoformat(),
                        daily_enforced=self.enforce_daily_limits)

    @staticmethod
    def policy_quota_id(scope):
        return 'policy:'+key(scope.quota_id)

    def resolve_scope_policy(self,scope):
        """Read-only old/new decision used before cache-key construction.

        A root row with previous consumption and no policy marker was started
        under the old shared-request policy; it must keep that policy on
        recovery. Missing rows mean a genuinely new request. Storage failures
        leave the in-memory policy unchanged and fail later at reservation, so
        zero-model responses are not turned into budget errors.
        """
        if getattr(scope,'policy',None)!=TASK_POLICY:
            return scope.policy
        try:
            with self.session() as session:
                self.prepare(session)
                root=session.get(ModelQuota,scope.quota_id)
                if root is None:
                    return scope.policy
                marker=session.get(ModelQuota,self.policy_quota_id(scope))
                if marker is not None:
                    self._validate_policy_marker(marker,root)
                    return scope.policy
                if root.calls>0 or root.tokens>0:
                    scope.policy='shared'
        except ModelBudgetError:
            # A corrupt marker/quota is not silently downgraded; the next real
            # reservation fails closed inside the request's normal policy path.
            return scope.policy
        except Exception:
            return scope.policy
        return scope.policy

    @staticmethod
    def _validate_policy_marker(marker,root):
        if (marker.kind!=POLICY_KIND or marker.calls!=0 or marker.tokens!=0
                or marker.max_calls!=1 or marker.max_tokens!=0
                or utc(marker.expires_at)!=utc(root.expires_at)):
            raise ModelBudgetUnavailable('invalid_budget_policy')

    def policy_in_session(self,session,scope,root=None,create_marker=True):
        """Decide and persist the old/new policy under the root row lock.

        The marker and all task rows are inserted by the caller's transaction.
        Any failure therefore rolls the decision back with the reservation.
        """
        moment=clock.utc_now()
        if root is None:
            root=session.scalar(select(ModelQuota).where(
                ModelQuota.quota_id==scope.quota_id).with_for_update())
        if root is None:
            raise ModelBudgetUnavailable('budget_scope_row_missing')
        if utc(root.expires_at)<=moment:
            raise ModelBudgetError('model_budget_expired')
        marker_id=self.policy_quota_id(scope)
        marker=session.scalar(select(ModelQuota).where(
            ModelQuota.quota_id==marker_id).with_for_update())
        if marker is not None:
            self._validate_policy_marker(marker,root)
            return True
        if root.calls>0 or root.tokens>0:
            return False
        if not create_marker:
            return True
        self.insert(session,ModelQuota,dict(quota_id=marker_id,kind=POLICY_KIND,calls=0,tokens=0,
            max_calls=1,max_tokens=0,created_at=moment,expires_at=root.expires_at))
        marker=session.scalar(select(ModelQuota).where(
            ModelQuota.quota_id==marker_id).with_for_update())
        if marker is None:
            raise ModelBudgetUnavailable('budget_policy_row_missing')
        self._validate_policy_marker(marker,root)
        return True

    def grant_daily_calls(self,additional_calls,*,day,apply=False):
        """Explicit local operator grant, idempotent for the current Shanghai day.

        Only the daily CALL ceiling is extended. Spending, token limits, request,
        stage, background and CLI limits remain intact; nothing is refunded.
        """
        moment=clock.utc_now();identifier,start,end=daily_window(moment)
        if (type(additional_calls) is not int or not 1<=additional_calls<=10000
                or day!=start.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()):
            raise ValueError('grant_requires_current_day_and_positive_calls')
        if not apply:
            return {**self.daily_status(),'apply':False,'requested_extra_calls':additional_calls}
        try:
            with self.session() as session,session.begin():
                self.prepare(session)
                self.insert(session,ModelQuota,dict(quota_id=identifier,kind='daily',calls=0,tokens=0,
                    max_calls=settings.model_daily_max_calls,max_tokens=settings.model_daily_max_tokens,
                    created_at=moment,expires_at=end))
                row=session.scalar(select(ModelQuota).where(ModelQuota.quota_id==identifier).with_for_update())
                existing=daily_extra_calls(session,moment)
                if existing and existing!=additional_calls:
                    raise ValueError('different_daily_grant_already_exists')
                if row.kind!='daily' or utc(row.expires_at)!=end:
                    raise ModelBudgetUnavailable('invalid_daily_quota')
                if not existing:
                    base=min(row.max_calls,settings.model_daily_max_calls)
                    if base+additional_calls>10000:raise ValueError('daily_grant_ceiling_exceeded')
                    row.max_calls=base+additional_calls
                    session.add(ModelQuota(quota_id='grant:'+identifier,kind='daily_grant',calls=0,tokens=0,
                        max_calls=additional_calls,max_tokens=0,created_at=moment,expires_at=end))
            return {**self.daily_status(),'apply':True,'grant_added':not bool(existing)}
        except (ValueError,ModelBudgetError):raise
        except Exception as exc:raise ModelBudgetUnavailable('daily_grant_failed') from exc

    @staticmethod
    def _task_error(kind,token=False):
        if kind=='prepare':
            return ModelPrepareBudgetError('model_prepare_'+
                ('token_budget_exhausted' if token else 'budget_exhausted'))
        return ModelTaskBudgetError('model_task_'+
            ('token_budget_exhausted' if token else 'budget_exhausted')+':'+kind)

    def reserve(self,scope,stages,capability,model,reserved_tokens,task=None):
        moment=clock.utc_now();stage=stages[-1][0]
        daily_id,midnight,day_end=daily_window(moment)
        root_expires=moment+timedelta(seconds=scope.ttl_seconds)
        effective_task=None
        per_task=False
        try:
            with self.session() as session,session.begin():
                self.prepare(session)
                # Fixed lock order for every writer: root, policy marker, task /
                # preparation, daily, background, then stages sorted by id.
                self.insert(session,ModelQuota,dict(quota_id=scope.quota_id,kind=scope.kind,
                    calls=0,tokens=0,max_calls=scope.max_calls,max_tokens=scope.max_tokens,
                    created_at=moment,expires_at=root_expires))
                root=session.scalar(select(ModelQuota).where(
                    ModelQuota.quota_id==scope.quota_id).with_for_update())
                if utc(root.expires_at)<=moment:raise ModelBudgetError('model_budget_expired')
                if getattr(scope,'policy',None)==TASK_POLICY:
                    per_task=self.policy_in_session(session,scope,root=root)
                    if not per_task:
                        # Old requests with prior shared consumption keep the
                        # frozen old strategy; they receive no new task quota.
                        scope.policy='shared'
                if per_task:
                    effective_task=task or prepare_budget(scope)
                    if effective_task.root_quota_id!=scope.quota_id:
                        raise ModelBudgetUnavailable('model_task_scope_mismatch')
                root_expiry=utc(root.expires_at)
                quotas=[(scope.quota_id,scope.kind,scope.max_calls,scope.max_tokens,
                         root_expiry,'root',not per_task)]
                if effective_task is not None:
                    quotas.append((effective_task.quota_id,effective_task.kind,
                        effective_task.max_calls,effective_task.max_tokens,
                        root_expiry,'prepare' if effective_task.kind=='prepare' else 'task',
                        True))
                quotas.append((daily_id,'daily',settings.model_daily_max_calls,
                               settings.model_daily_max_tokens,day_end,'shared',True))
                if scope.kind=='background':
                    quotas.append(('background-day:'+key(midnight.isoformat()),'background_day',
                        settings.model_background_daily_max_calls,settings.model_daily_max_tokens,
                        day_end,'shared',True))
                stage_base=scope.quota_id+(effective_task.quota_id if effective_task else '')
                stage_tokens=effective_task.max_tokens if effective_task else scope.max_tokens
                for name,maximum in stages:
                    quotas.append(('stage:'+key(stage_base+name),'stage',maximum,
                                   stage_tokens,root_expiry,'shared',True))
                by_id={}
                for item in quotas:
                    by_id.setdefault(item[0],item)
                # The exact same root -> task/prepare -> shared order is used
                # by finish(), including ledgers whose stored ids were sorted.
                ordered=[by_id[qid] for qid in quota_lock_order(
                    scope.quota_id,[item[0] for item in quotas])]
                for qid,kind,maximum,tokens,expires,actor,enforced in ordered:
                    self.insert(session,ModelQuota,dict(quota_id=qid,kind=kind,calls=0,tokens=0,
                        max_calls=maximum,max_tokens=tokens,created_at=moment,expires_at=expires))
                    row=session.scalar(select(ModelQuota).where(
                        ModelQuota.quota_id==qid).with_for_update())
                    if utc(row.expires_at)<=moment:raise ModelBudgetError('model_budget_expired')
                    if kind=='daily':maximum+=daily_extra_calls(session,moment)
                    # Tightening config applies immediately; raising config does
                    # not silently replenish an existing task/request/job budget.
                    # An explicit isolated simulation store may skip only the
                    # daily ceiling checks; the daily row still accumulates.
                    limit=min(row.max_calls,maximum);token_limit=min(row.max_tokens,tokens)
                    enforce=enforced and (kind!='daily' or self.enforce_daily_limits)
                    if enforce and row.calls>=limit:
                        if actor=='task':raise self._task_error(kind)
                        if actor=='prepare':raise self._task_error('prepare')
                        raise ModelBudgetError('model_budget_exhausted:'+kind)
                    if enforce and row.tokens+reserved_tokens>token_limit:
                        if actor=='task':raise self._task_error(kind,token=True)
                        if actor=='prepare':raise self._task_error('prepare',token=True)
                        raise ModelBudgetError('model_token_budget_exhausted:'+kind)
                    row.max_calls=limit;row.max_tokens=token_limit
                    row.calls+=1;row.tokens+=reserved_tokens
                call_id=str(uuid4())
                session.add(ModelCallLedger(call_id=call_id,scope_id=scope.quota_id,kind=scope.kind,
                    capability=capability,model=str(model)[:100],stage_id=key(stage),
                    quota_ids=[item[0] for item in ordered],
                    reserved_tokens=reserved_tokens,status='started',started_at=moment))
            return call_id
        except ModelBudgetError:raise
        except Exception as exc:raise ModelBudgetUnavailable('model_budget_reservation_failed') from exc

    @classmethod
    def open_in_session(cls,session,name,kind,max_calls,max_tokens,ttl_seconds):
        moment=clock.utc_now()
        identifier='scope:'+key(kind+':'+name)
        cls.insert(session,ModelQuota,dict(quota_id=identifier,kind=kind,calls=0,tokens=0,
                   max_calls=max_calls,max_tokens=max_tokens,created_at=moment,expires_at=moment+timedelta(seconds=ttl_seconds)))
        return identifier

    def finish(self,call_id,status,usage,error_code=None):
        # A missing completion record keeps the conservative reservation. A
        # successful provider result must not be turned into a second request.
        try:
            with self.session() as session,session.begin():
                self.prepare(session)
                row=session.get(ModelCallLedger,call_id)
                if row is None or row.finished_at is not None:return False
                row.status=status;row.usage=usage or None;row.error_code=error_code;row.finished_at=clock.utc_now()
                total=usage.get('total_tokens') if isinstance(usage,dict) else None
                if total is None and isinstance(usage,dict):
                    first=usage.get('prompt_tokens',usage.get('input_tokens'))
                    last=usage.get('completion_tokens',usage.get('output_tokens'))
                    if isinstance(first,int) and isinstance(last,int):total=first+last
                    elif row.capability!='chat' and isinstance(first,int):total=first
                if isinstance(total,int) and not isinstance(total,bool) and total>=0:
                    difference=total-row.reserved_tokens
                    for qid in quota_lock_order(row.scope_id,row.quota_ids):
                        session.execute(update(ModelQuota).where(ModelQuota.quota_id==qid).values(tokens=ModelQuota.tokens+difference))
            return True
        except Exception:return False

    def cached(self,cache_id):
        try:
            with self.session() as session:
                self.prepare(session)
                row=session.get(ModelResultCache,cache_id)
                return json.loads(json.dumps(row.result)) if row else None
        except Exception as exc:raise ModelBudgetUnavailable('model_cache_read_failed') from exc

    def cache(self,cache_id,scope,capability,result):
        try:
            with self.session() as session,session.begin():
                self.prepare(session)
                self.insert(session,ModelResultCache,dict(cache_id=cache_id,scope_id=scope.quota_id,capability=capability,
                    owner_id=scope.owner_id,result=result,created_at=clock.utc_now()))
            return True
        except Exception:return False


@dataclass
class ModelScope:
    name:str
    kind:str
    store:BudgetStore
    max_calls:int
    max_tokens:int
    ttl_seconds:float
    owner_id:str|None=None
    policy:str='shared'
    started:float=field(default_factory=time.monotonic)
    cancelled:bool=False
    policy_stops:set=field(default_factory=set)
    @property
    def quota_id(self):return 'scope:'+key(self.kind+':'+self.name)
    def remaining(self):return max(0,self.ttl_seconds-(time.monotonic()-self.started))
    def check(self):
        if self.cancelled:
            self.policy_stops.add('model_request_cancelled');raise ModelBudgetError('model_request_cancelled')
        if self.remaining()<=0:
            self.policy_stops.add('model_request_deadline');raise ModelBudgetError('model_request_deadline')


@contextmanager
def model_scope(name,kind='request',*,store=None,max_calls=None,max_tokens=None,ttl_seconds=None,
                reuse=False,owner_id=None,policy='shared'):
    if policy not in ('shared',TASK_POLICY):
        raise ValueError('model_scope_policy_invalid')
    current=active_scope.get()
    if reuse and current is not None:
        yield current;return
    scope=ModelScope(str(name),kind,store or store_override.get() or BudgetStore(),
                     max_calls or (settings.model_background_max_calls if kind=='background' else settings.model_cli_max_calls if kind=='cli' else settings.model_request_max_calls),
                     max_tokens or (settings.model_background_max_tokens if kind=='background' else settings.model_cli_max_tokens if kind=='cli' else settings.model_request_max_tokens),
                     ttl_seconds or (settings.memory_outbox_ttl_seconds if kind=='background' else settings.model_cli_deadline_seconds if kind=='cli' else settings.model_request_deadline_seconds))
    scope.owner_id=owner_id;scope.policy=policy
    token=active_scope.set(scope);stage_token=active_stages.set(())
    # A new independent root (background/CLI/direct child) must not inherit a
    # parent task context. ContextVar values are restored on exit.
    task_token=active_task.set(None)
    try:yield scope
    except BaseException as exc:
        if type(exc).__name__ in ('CancelledError','GeneratorExit'):scope.cancelled=True
        raise
    finally:active_task.reset(task_token);active_stages.reset(stage_token);active_scope.reset(token)


@contextmanager
def model_stage(name,max_calls=2):
    token=active_stages.set((*active_stages.get(),(str(name),int(max_calls))))
    try:yield
    finally:active_stages.reset(token)


def _task_budget(scope,goal,task_type,max_calls=None,max_tokens=None):
    from app.core.config import settings
    if scope.owner_id is None:
        raise ModelBudgetUnavailable('model_task_owner_required')
    goal=str(goal)
    if not goal:
        raise ValueError('model_task_goal_required')
    # task_type still describes the business route and the durable row kind,
    # but it never selects a different allowance. Every new task uses the one
    # canonical ceiling; an explicit stricter max_calls can only tighten it.
    kind='task_plan' if str(task_type)=='training_plan' else 'task_standard'
    calls=settings.model_task_max_calls
    if max_calls is not None:calls=min(calls,int(max_calls))
    tokens=min(settings.model_task_max_tokens,int(max_tokens)) if max_tokens is not None \
        else settings.model_task_max_tokens
    return TaskBudget(
        quota_id='task:'+key(str(scope.owner_id)+'|'+scope.quota_id+'|'+goal),
        kind=kind,goal=goal,root_quota_id=scope.quota_id,task_type=str(task_type),
        max_calls=calls,max_tokens=tokens)


@contextmanager
def model_task(goal,*,task_type='',max_calls=None,max_tokens=None):
    """Charge every physical call in the block to one frozen semantic goal.

    On a new-policy request the quota is durable and can only be tightened.
    On old/shared or low-level explicit scopes this is a no-op; those paths keep
    their original request/CLI/background semantics.
    """
    scope=active_scope.get()
    if scope is None or getattr(scope,'policy',None)!=TASK_POLICY:
        yield None;return
    current=active_task.get()
    if current is not None:
        if current.root_quota_id!=scope.quota_id:
            raise ModelBudgetUnavailable('model_task_scope_mismatch')
        if current.kind!='prepare':
            # Every internal child step remains charged to the already-bound
            # user task. A new label/type never buys another allowance.
            yield current;return
    task=_task_budget(scope,goal,task_type,max_calls,max_tokens)
    token=active_task.set(task)
    try:yield task
    finally:active_task.reset(token)


@contextmanager
def model_prepare():
    """Charge shared understanding/safety/first planning to one fixed identity."""
    scope=active_scope.get()
    if scope is None or getattr(scope,'policy',None)!=TASK_POLICY:
        yield None;return
    current=active_task.get()
    if current is not None:
        if current.root_quota_id!=scope.quota_id:
            raise ModelBudgetUnavailable('model_task_scope_mismatch')
        if current.kind!='prepare':
            # Preparation work nested inside an exhausted user task cannot
            # escape to the shared preparation allowance.
            yield current;return
        yield current;return
    task=prepare_budget(scope)
    token=active_task.set(task)
    try:yield task
    finally:active_task.reset(token)


def require_scope():
    scope=active_scope.get()
    if scope is None:raise ModelBudgetError('model_budget_scope_required')
    scope.check();return scope


def cache_key(capability,model,payload,provider=''):
    scope=require_scope()
    task=active_task.get()
    if task is None and getattr(scope,'policy',None)==TASK_POLICY:
        task=prepare_budget(scope)
    serialized=json.dumps(payload,ensure_ascii=False,sort_keys=True,default=str)
    target=task.quota_id if task is not None else ''
    return key(scope.quota_id+target+capability+str(model)+provider+serialized)


class PaidAttempt:
    def __init__(self,capability,model,payload,output_tokens=0,provider=''):
        self.scope=require_scope();self.capability=capability
        self.task=active_task.get()
        if self.task is None and getattr(self.scope,'policy',None)==TASK_POLICY:
            self.task=prepare_budget(self.scope)
        encoded=json.dumps(payload,ensure_ascii=False,sort_keys=True,default=str).encode('utf-8')
        # Chat reuses the exact finalized request's window estimate; other
        # capabilities retain the conservative byte bound. Neither is usage.
        from app.services.context.request_tokens import PreparedPayload, count_request
        if capability == 'chat' and isinstance(payload, dict) and payload.get('model') == model and isinstance(payload.get('messages'), list):
            tokens = payload.counted_tokens() if isinstance(payload, PreparedPayload) else count_request(payload)
            reserve = tokens.input_tokens + max(0, int(output_tokens))
        else:
            reserve = len(encoded) + max(0, int(output_tokens))
        automatic=(capability+':'+key(str(model)+encoded.hex()),settings.model_stage_max_calls)
        stages=(*active_stages.get(),automatic)
        try:self.call_id=self.scope.store.reserve(self.scope,stages,capability,model,reserve,
                                                  task=self.task)
        except ModelTaskBudgetError as exc:
            if self.task is not None:
                self.task.limited=True;self.task.exhausted_code=exc.code
            raise
        except ModelBudgetError as exc:
            self.scope.policy_stops.add(exc.code);raise
        self.usage={};self.error_code=None
    def observe(self,data):
        usage=data.get('usage') if isinstance(data,dict) else None
        if isinstance(usage,dict):
            # Streaming frames report cumulative usage. Never sum them.
            for k,v in usage.items():
                if isinstance(v,int) and not isinstance(v,bool) and v>=0:self.usage[k]=max(v,self.usage.get(k,0))
    def finish(self,error=None):
        status='success' if error is None else 'cancelled' if type(error).__name__ in ('CancelledError','GeneratorExit') else 'error'
        code=None if error is None else type(error).__name__
        status_code=getattr(error,'status_code',None) or getattr(getattr(error,'response',None),'status_code',None)
        if status_code:code+=':http_'+str(status_code)
        self.scope.store.finish(self.call_id,status,self.usage,code)


@contextmanager
def paid_attempt(capability,model,payload,output_tokens=0,provider=''):
    attempt=PaidAttempt(capability,model,payload,output_tokens,provider)
    try:yield attempt
    except BaseException as exc:attempt.finish(exc);raise
    else:attempt.finish()


@asynccontextmanager
async def async_paid_attempt(capability,model,payload,output_tokens=0,provider=''):
    # SQL quota locks never block the event loop. Cancellation during reserve
    # leaves a conservative reservation but never starts the HTTP request.
    attempt=await asyncio.to_thread(PaidAttempt,capability,model,payload,output_tokens,provider)
    attempt.scope.check()
    try:yield attempt
    except BaseException as exc:
        await asyncio.shield(asyncio.to_thread(attempt.finish,exc));raise
    else:await asyncio.to_thread(attempt.finish)


async def cached_async(cache_id):
    return await asyncio.to_thread(require_scope().store.cached,cache_id)


async def cache_async(cache_id,capability,result):
    scope=require_scope()
    return await asyncio.to_thread(scope.store.cache,cache_id,scope,capability,result)


def stage_operation(name,maximum=2):
    from functools import wraps
    def decorate(function):
        @wraps(function)
        async def call(*args,**kwargs):
            with model_stage(name,maximum):return await function(*args,**kwargs)
        return call
    return decorate
