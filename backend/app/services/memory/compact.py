"""Single-person durable memory. PostgreSQL is authoritative; Wiki addresses select context."""
from __future__ import annotations
from app.core import clock
import asyncio
import hashlib
import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from sqlalchemy import select, or_, func
from app.core.config import settings
from app.db import session as db
from app.db.storage import StorageUnavailableError
from app.db.models import UserProfileModel
from app.db.memory_models import MemoryTurn, MemoryOutbox, UserFact
from app.services.context.token_counter import TokenCounter
from app.services.rag.bm25 import BM25Index
from app.services.rag.blocking import run_retrieval
from app.services.rag.retrieval import RetrievedChunk, retrieval_service
from app.services.llm.budget import model_scope, model_stage, BudgetStore, key, utc, probe_budget_storage
from app.db.model_budget_models import ModelQuota
from app.services.memory.retry_policy import classify_memory_failure
from app.services.rag.embedding_client import embedding_client

current_turn: ContextVar[str | None] = ContextVar('current_memory_turn', default=None)

logger = logging.getLogger(__name__)

def now():
    return clock.utc_now()

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()

def session():
    if not db.DB_READY or db.SessionLocal is None:
        raise StorageUnavailableError('memory_database_unavailable')
    return db.SessionLocal()

def lock_profile(s):
    # SQLite ignores SELECT FOR UPDATE. Acquire its write reservation before
    # reading facts so concurrent final commits cannot both replace a stale
    # active row. PostgreSQL continues to use the profile row lock below.
    connection = s.connection()
    if connection.dialect.name == 'sqlite':
        driver = connection.connection.driver_connection
        if not driver.in_transaction:
            connection.exec_driver_sql('BEGIN IMMEDIATE')
    profile = s.scalar(select(UserProfileModel).where(UserProfileModel.user_id == 'demo-user').with_for_update())
    if profile is None:
        profile = UserProfileModel(user_id='demo-user', memory_epoch=0)
        s.add(profile)
        s.flush()
    return profile

def epoch_value():
    with session() as s, s.begin():
        return lock_profile(s).memory_epoch

def enqueue(s, turn, op='index'):
    event = f'{turn.turn_id}:{turn.content_hash[:16]}:{turn.memory_epoch}'
    if s.scalar(select(MemoryOutbox).where(MemoryOutbox.source_event_id == event, MemoryOutbox.op == op)):
        return
    BudgetStore.open_in_session(s, 'memory:'+turn.turn_id+':'+turn.content_hash, 'background',
        settings.model_background_max_calls, settings.model_background_max_tokens, settings.memory_outbox_ttl_seconds)
    # A profile epoch change/re-enqueue must not reset the same content's budget.
    inherited=s.scalar(select(func.max(MemoryOutbox.attempts)).where(MemoryOutbox.turn_id==turn.turn_id, MemoryOutbox.content_hash==turn.content_hash, MemoryOutbox.op==op)) or 0
    s.add(MemoryOutbox(event_id=str(uuid4()), source_event_id=event, turn_id=turn.turn_id,
        op=op, content_hash=turn.content_hash, memory_epoch=turn.memory_epoch, attempts=inherited))

def replayable_response(request_id, turn_id):
    """Best-effort recovery for a completed turn without a stored response.

    Never regenerates identifiers and never re-runs business operations: the
    diagnostics record is reused when available, otherwise the caller gets an
    explicit "cannot replay" answer.
    """
    rebuilt = None
    try:
        from app.db.quality_repository import load_run_for_replay
        rebuilt = load_run_for_replay(request_id)
    except Exception:
        logger.exception("replay_reconstruction_failed", extra={"request_id": request_id})
    if rebuilt is not None:
        return rebuilt
    return {
        'answer': '这次请求已经执行完成，但原始回答没有成功保存。为避免重复写入，本次没有重新执行任何记录操作；请发送一条新的消息继续。',
        'intent': 'replay_unavailable',
        'citations': [],
        'safety_level': 'normal',
        'model_used': False,
        'memory_status': 'degraded',
        'diagnostics_status': 'unavailable',
        'request_id': request_id,
        'run_id': None,
        'message_id': turn_id + ':assistant',
    }


def begin_turn(request):
    completed = False
    response = None
    requested_mode = getattr(request, "mode", None) or "legacy"
    with session() as s, s.begin():
        profile = lock_profile(s)
        row = s.scalar(select(MemoryTurn).where(MemoryTurn.request_id == request.request_id))
        if row:
            if row.deleted_at or row.hidden_at:
                raise ValueError('request_was_deleted')
            if row.conversation_id != request.conversation_id or row.messages[0]['content'] != request.message:
                raise ValueError('request_id_payload_conflict')
            # 同 request_id 变更请求模式沿用 RequestIdConflict（阶段B）。
            frozen_mode = 'legacy'
            first_message = row.messages[0] if isinstance(row.messages, list) and row.messages else None
            if isinstance(first_message, dict) and isinstance(first_message.get('app_meta'), dict):
                frozen_mode = first_message['app_meta'].get('requested_mode') or 'legacy'
            if requested_mode != frozen_mode:
                raise ValueError('request_id_payload_conflict')
            tid = row.turn_id
            completed = row.status == 'completed'
            response = row.response if completed else None
        else:
            tid = str(uuid4())
            messages = [{'role':'user', 'content':request.message, 'message_id':tid+':user', 'turn_id':tid,
                         'app_meta': {'requested_mode': requested_mode}}]
            turn = MemoryTurn(turn_id=tid, request_id=request.request_id, conversation_id=request.conversation_id,
                messages=messages, content_hash=digest(messages), memory_epoch=profile.memory_epoch)
            s.add(turn); s.flush()
            from app.services.memory.lifecycle import allocate_source_version
            version = allocate_source_version(s, turn, owner=profile.user_id, epoch=profile.memory_epoch)
            tagged = json.loads(json.dumps(messages, ensure_ascii=False))
            tagged[0]['app_meta']['_memory_version'] = version
            turn.messages = tagged
            turn.content_hash = digest(turn.messages)
    if completed and not isinstance(response, dict):
        # Historical split commit: completed with answer but no full response.
        response = replayable_response(request.request_id, tid)
    return tid, response


def unfinished_frozen_mode(request):
    """已有未完成请求冻结的 requested_mode；不存在/已完成/已删除返回 None。

    运行时在 ``begin_turn`` 之前调用：同一 request_id 的未完成请求不得因为服务器
    开关变化而重新选择 legacy（阶段B fail closed）。旧行缺少模式元数据仍按 legacy
    兼容，与 ``begin_turn`` 的冲突检查保持同一默认值。
    """
    try:
        with session() as s:
            row = s.scalar(select(MemoryTurn).where(MemoryTurn.request_id == request.request_id))
    except Exception:
        logger.exception("unfinished_frozen_mode_unavailable")
        return None
    if row is None or row.deleted_at or row.hidden_at or row.status in ('completed', 'deleted'):
        return None
    first = row.messages[0] if isinstance(row.messages, list) and row.messages else None
    if isinstance(first, dict) and isinstance(first.get('app_meta'), dict):
        value = first['app_meta'].get('requested_mode')
        if value == 'react_readonly':
            return 'react_readonly'
    return 'legacy'


def attach_effective_mode(tid, effective_mode, mode_reason):
    """把已判定的 effective_mode/reason 合并进首条消息应用元数据（现有JSON，无新列）。"""
    if not tid:
        return False
    with session() as s, s.begin():
        row = s.get(MemoryTurn, tid)
        if row is None or row.deleted_at or row.hidden_at or row.status in ('completed', 'deleted'):
            return False
        messages = list(row.messages or [])
        first = messages[0] if messages and isinstance(messages[0], dict) else None
        if first is None:
            return False
        meta = dict(first.get('app_meta') or {})
        meta.setdefault('requested_mode', 'legacy')
        meta['effective_mode'] = str(effective_mode)
        meta['mode_reason'] = str(mode_reason or '')
        first['app_meta'] = meta
        messages[0] = first
        row.messages = messages
        row.content_hash = digest(messages)
    return True

def recent_dialogue(rows):
    """Keep the latest question/answer pair intact before semantic routing.

    Older turns use their extractive windows and can be restored by validated
    source IDs. The model gateway enforces the final input budget; it must not
    silently remove the actual question a short user reply is answering.
    """
    output = []
    for index, row in enumerate(rows):
        messages = row.messages if index == len(rows) - 1 else (
            json.loads(row.window_summary) if row.window_summary else soft_window(row.messages))
        output.extend(m for m in messages if m.get("role") in ("user", "assistant"))
    return output


def soft_window(messages):
    # Extractive window representation. Original messages remain untouched in SQL.
    counter = TokenCounter()
    if counter.messages(messages) <= settings.memory_turn_soft_tokens:
        return messages
    output=[]
    for item in messages:
        if item['role'] not in ('user','assistant'):
            continue
        content=item['content']
        allowance=max(128, settings.memory_turn_soft_tokens // 2)
        if counter.text(content) > allowance:
            if item.get("app_meta", {}).get("semantic_critical"):
                output.append(dict(item))
                continue
            cap = max(1, (allowance-80)//8)
            selected = [content[:cap], "（中段省略）", content[-cap:]]
            content='[短期窗口压缩摘录，部分原文省略]'+''.join(selected)
        output.append({**item,'content':content})
    return output

def _state_confirmation_text(confirmations: list[dict]) -> str:
    """One concise confirmation per scope, not one line per temporary value."""
    parts: list[str] = []
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for item in confirmations:
        scope = "本轮" if item.get("scope") == "request" else "本会话"
        if scope not in grouped:
            grouped[scope] = {"values": [], "cleared": False}
            order.append(scope)
        if not item.get("effective"):
            grouped[scope]["stale"] = True
        elif item.get("status") == "cleared":
            grouped[scope]["cleared"] = True
        else:
            value = str(item.get("value") or "").strip()
            if value:
                grouped[scope]["values"].append(value)
    for scope in order:
        item = grouped[scope]
        if item.get("stale"):
            parts.append(f"{scope}临时条件未更新：已有更新版本。")
            continue
        if item["cleared"]:
            parts.append(f"{scope}已解除临时条件。")
        if item["values"]:
            parts.append(f"{scope}已更新临时条件：" + "、".join(dict.fromkeys(item["values"])) + "。")
    return " ".join(parts)


def finish_turn(tid, answer, response=None, *, semantic_values=()):
    """Single final commit: assistant history, completion and full response.

    Live requests must pass the complete response so a completed turn is always
    replayable. ``response=None`` is kept only for legacy callers that already
    predate the atomic commit; it must not be used for new request handling.
    """
    with session() as s, s.begin():
        profile=lock_profile(s)
        row=s.get(MemoryTurn,tid)
        if row is None or row.deleted_at or row.memory_epoch != profile.memory_epoch:
            raise RuntimeError('memory_changed_during_request')
        if row.status == 'completed':
            if response is not None:
                row.response=response
            return response
        assistant_message = {'role':'assistant','content':answer,'message_id':tid+':assistant','turn_id':tid}
        preview = ((response or {}).get('metrics') or {}).get('plan_preview')
        if isinstance(preview, dict) and preview:
            assistant_message['app_meta'] = {'plan_preview': preview}
        row.messages=[*row.messages, assistant_message]
        row.status='completed'; row.completed_at=now()
        row.response=response
        # Only explicit user requests create facts; never use the assistant as evidence.
        user=row.messages[0]
        semantic = next((value for source, value in semantic_values if source == user["content"]), None)
        from app.services.agent.result_reference import completed_result_targets
        from app.services.agent.execution_state import _request_memory
        targets = completed_result_targets(semantic, response, user.get('message_id',''), _request_memory.get())
        if targets:
            last = dict(row.messages[-1])
            last['app_meta'] = {**(last.get('app_meta') or {}), 'result_targets': targets}
            row.messages = [*row.messages[:-1], last]
        memory_receipts = []
        lifecycle_receipts = []
        state_confirmations: list[dict] = []
        if semantic and not semantic.get("__semantic_unavailable__"):
            first = dict(row.messages[0])
            meta = dict(first.get("app_meta") or {})
            meta["semantic_critical"] = bool(semantic.get("current_is_critical"))
            first["app_meta"] = meta
            row.messages = [first, *row.messages[1:]]
            if not semantic.get("needs_clarification"):
                from app.services.memory.facts import store_explicit_fact
                for decision in semantic.get("memory_requests") or []:
                    if decision.get('kind') == 'transient':
                        accepted = semantic.get('safety_assessed') and (semantic.get('safety') or {}).get('level') in {'normal', 'medical'} and not semantic.get('current_is_critical') and user.get('app_meta', {}).get('_memory_version') is not None
                        receipt = {'status': 'active' if accepted else 'pending_confirmation',
                                   'quote': decision['quote'], 'temporary': True}
                    else:
                        receipt = store_explicit_fact(s, row=row, user=user, owner=profile.user_id,
                            epoch=profile.memory_epoch, decision=decision,
                            allowed_fields=semantic.get("allowed_memory_fields") or [],
                            target_sources=semantic.get("memory_target_sources") or {},
                            digest=digest, timestamp=row.completed_at, new_id=lambda: str(uuid4()))
                    if receipt:
                        memory_receipts.append(receipt)
                        lifecycle_receipts.append((decision, receipt))
                if semantic.get('safety_assessed') and (semantic.get('safety') or {}).get('level') in {'normal', 'medical'}:
                    from app.services.memory.lifecycle import commit_context
                    state_confirmations = commit_context(
                        s, row, semantic, owner=profile.user_id, epoch=profile.memory_epoch,
                        receipts=lifecycle_receipts) or []
        # No model decision means no trusted memory extraction: an explicit
        # "remember" keyword alone cannot create a fact.
        if response and (memory_receipts or (response.get("metrics") or {}).get("memory_update_requested")):
            saved = list(dict.fromkeys(item["quote"] for item in memory_receipts if item["status"] == "active"))
            pending = any(item["status"] != "active" for item in memory_receipts)
            # Only format the receipt; source quotes/fact values stay verbatim.
            answer = ("已记住：" + "；".join(q.rstrip("。；; ") for q in saved) + "。") if saved else "这次记忆尚未确认保存。"
            if saved and all(item.get('temporary') for item in memory_receipts):
                answer = answer.replace('已记住：', '已临时记住（到期后不再作为当前条件）：', 1)
            if pending:
                answer += " 另有信息与已有记忆冲突，需要你明确确认后才能替换。"
            if not (response.get("metrics") or {}).get("memory_update_requested"):
                answer = str(response.get("answer") or "").rstrip() + "\n\n" + answer
            response = {**response, "answer": answer,
                        "metrics": {**response.get("metrics", {}), "memory_update_status": "pending_confirmation" if pending else "committed" if saved else "not_committed"}}
            row.response = response
            row.messages = [*row.messages[:-1], {**row.messages[-1], "content": answer}]
        if response and (state_confirmations or (response.get("metrics") or {}).get("state_update_requested")):
            confirmation = _state_confirmation_text(state_confirmations)
            statuses = [item.get("effective") is True for item in state_confirmations]
            state_status = ("committed" if state_confirmations and all(statuses)
                            else "superseded" if state_confirmations and not any(statuses)
                            else "partial" if state_confirmations else "not_committed")
            if not confirmation:
                confirmation = "本次会话临时条件未更新或未确认。"
            if (response.get("metrics") or {}).get("state_update_requested"):
                answer = confirmation
            else:
                answer = str(response.get("answer") or "").rstrip()
                answer = answer + "\n\n" + confirmation if answer else confirmation
            response = {**response, "answer": answer,
                        "metrics": {**response.get("metrics", {}),
                                    "state_update_status": state_status}}
            row.response = response
            row.messages = [*row.messages[:-1], {**row.messages[-1], "content": answer}]
        # Publish task-continuity addresses with the answer in the same transaction.
        from app.services.agent.execution_state import _request_memory
        context = _request_memory.get() or {}
        if context.get('wiki_conversation_id') == row.conversation_id and context.get('wiki_prepared'):
            first = dict(row.messages[0]); meta = dict(first.get('app_meta') or {})
            meta['wiki_pages'] = list((context.get('wiki_loaded_pages') or {}).keys())
            first['app_meta'] = meta; row.messages = [first, *row.messages[1:]]
        # A later receipt may append to the assistant content. Keep the preview
        # artifact valid by recomputing its digest over the final answer.
        if isinstance(((response or {}).get('metrics') or {}).get('plan_preview'), dict):
            last = dict(row.messages[-1])
            meta = dict(last.get('app_meta') or {})
            artifact = meta.get('plan_preview')
            if isinstance(artifact, dict):
                artifact = {**artifact, 'answer_digest': digest(str(last.get('content') or ''))}
                meta['plan_preview'] = artifact
                last['app_meta'] = meta
                row.messages = [*row.messages[:-1], last]
        row.content_hash=digest(row.messages)
        row.window_summary=json.dumps(soft_window(row.messages),ensure_ascii=False)
        s.flush()
        from app.services.memory.wiki import materialize
        materialize(s,row)
        complete=list(s.scalars(select(MemoryTurn).where(MemoryTurn.conversation_id==row.conversation_id,
            MemoryTurn.status=='completed',MemoryTurn.deleted_at.is_(None),MemoryTurn.hidden_at.is_(None)).order_by(MemoryTurn.completed_at.desc(),MemoryTurn.turn_id.desc())))
        for old in complete[settings.memory_short_window_turns:]:
            if old.index_status == 'pending_window':
                old.index_status='pending'; enqueue(s,old)
    # Return only after the transaction context committed successfully.
    return response

def fail_turn(tid):
    with session() as s, s.begin():
        row=s.get(MemoryTurn,tid)
        if row and row.status != 'completed' and not row.deleted_at:
            row.status='failed'

def load_request_state(request_id):
    """Return a detached copy of the persisted execution state (P1-2)."""
    with session() as s:
        row=s.scalar(select(MemoryTurn).where(MemoryTurn.request_id==request_id))
        if row is None or row.deleted_at or row.hidden_at or not isinstance(row.operations, dict):
            return None
        return json.loads(json.dumps(row.operations,ensure_ascii=False,default=str))

def save_request_state(request_id,state):
    """Replace the execution state atomically; only for the live turn."""
    payload=json.loads(json.dumps(state,ensure_ascii=False,default=str))
    with session() as s,s.begin():
        row=s.scalar(select(MemoryTurn).where(MemoryTurn.request_id==request_id).with_for_update())
        if row is None or row.deleted_at or row.hidden_at or row.status=='deleted':
            return False
        row.operations=payload
        return True

def append_observation(tid,name,arguments,result):
    with session() as s,s.begin():
        profile=lock_profile(s); row=s.get(MemoryTurn,tid)
        if not row or row.deleted_at or row.hidden_at or row.memory_epoch!=profile.memory_epoch or row.status=='completed':
            return
        # Store the business observation, not transport headers/credentials.
        # Tool arguments can contain datetime/date values from validated
        # workout and nutrition payloads. Normalize them before persisting the
        # JSON observation so metadata serialization cannot mask a successful
        # business write as a failed save.
        safe_arguments=json.loads(json.dumps(arguments,ensure_ascii=False,default=str))
        mid=f'{tid}:tool:{len(row.messages)}'
        row.messages=[*row.messages,{'role':'tool','name':name,'content':json.dumps(result,ensure_ascii=False,default=str),
            'message_id':mid,'turn_id':tid,'arguments':safe_arguments}]
        row.content_hash=digest(row.messages)

def recent_rows(cid):
    with session() as s:
        return list(reversed(list(s.scalars(select(MemoryTurn).where(MemoryTurn.conversation_id==cid,
            MemoryTurn.status=='completed',MemoryTurn.deleted_at.is_(None),
            MemoryTurn.hidden_at.is_(None)).order_by(MemoryTurn.completed_at.desc(),MemoryTurn.turn_id.desc()).limit(settings.memory_short_window_turns)))))


def latest_result_dialogue(cid, epoch, *, latest_messages=None, cached_messages=()):
    """One visible current-conversation exchange for short follow-up identity.

    Wiki retrieval still owns older history. Filtering and limit happen in SQL;
    a deleted/hidden/old-epoch result never becomes the current follow-up target.
    Limited execution replies describe a failed attempt, not a new result.
    ``latest_messages`` must come from the same filtered, visible source set.
    """
    with session() as s:
        rows=([latest_messages] if latest_messages is not None else list(s.scalars(select(MemoryTurn.messages).where(
            MemoryTurn.conversation_id==cid, MemoryTurn.memory_epoch==epoch,
            MemoryTurn.status=='completed', MemoryTurn.deleted_at.is_(None),
            MemoryTurn.hidden_at.is_(None),
            func.coalesce(MemoryTurn.response['intent'].as_string(), '') != 'execution_limited',
            func.coalesce(MemoryTurn.response['metrics']['execution_mode'].as_string(), '') != 'limited',
            func.coalesce(MemoryTurn.response['metrics']['react_status'].as_string(), 'complete') == 'complete').order_by(
                MemoryTurn.completed_at.desc(),MemoryTurn.turn_id.desc()).limit(1))))
        latest = []
        for messages in rows:
            for message in messages:
                if message.get('role') == 'user':
                    latest.append(message)
                elif message.get('role') == 'assistant':
                    meta=message.get('app_meta') or {}
                    latest.append({k:v for k,v in message.items() if k in {'role','message_id','turn_id'}} | {
                        'content':'','content_status':'not_loaded',
                        'app_meta':{'result_targets':meta.get('result_targets') or [],
                                    'plan_preview':{'preview_id':(meta.get('plan_preview') or {}).get('preview_id')}}})
        # Regenerations retain pointers to original user goals, not copied
        # prose. Resolve them in one bounded query under the same visibility
        # checks, so repeated regeneration cannot forget its initial goal or
        # revive deleted/hidden facts through assistant metadata.
        source_ids={mid for m in latest for t in m.get('app_meta',{}).get('result_targets',[])
                    for mid in [*t.get('source_user_ids',[]), *t.get('context_message_ids',[])]}
        known={m.get('message_id') for m in latest}
        def context_origin(message):
            if message.get('role') == 'user':
                return message
            # Background answer identities stay addressable, but their body is
            # restored only after selection and live visibility/version checks.
            return {k: v for k, v in message.items() if k in {'role', 'message_id', 'turn_id'}} | {
                'content': '', 'content_status': 'not_loaded',
                'app_meta': {'result_context_only': True,
                             'recent_content_hash': message.get('app_meta', {}).get('recent_content_hash')}}
        cached=[context_origin(m) for m in cached_messages if m.get('role') in {'user', 'assistant'}
                and m.get('message_id') in source_ids-known]
        known.update(m.get('message_id') for m in cached)
        latest=[*cached,*latest]
        missing=source_ids-known
        if not missing:
            return latest
        turn_ids={mid.rpartition(':')[0] for mid in missing if mid.rpartition(':')[2] in {'user', 'assistant'}}
        originals=list(s.execute(select(MemoryTurn.turn_id, MemoryTurn.content_hash, MemoryTurn.messages).where(
            MemoryTurn.turn_id.in_(turn_ids), MemoryTurn.conversation_id==cid,
            MemoryTurn.memory_epoch==epoch, MemoryTurn.status=='completed',
            MemoryTurn.deleted_at.is_(None), MemoryTurn.hidden_at.is_(None)).order_by(
                MemoryTurn.completed_at,MemoryTurn.turn_id))) if turn_ids else []
        selected=[context_origin({**m, 'turn_id': row.turn_id,
                    'app_meta': {**m.get('app_meta', {}), 'recent_content_hash': row.content_hash}})
                  for row in originals for m in row.messages
                  if m.get('role') in {'user', 'assistant'} and m.get('message_id') in missing]
        return [*selected,*latest]

def current_facts():
    from app.services.memory.lifecycle import snapshot
    return snapshot()['long_memory']

class MemoryAdapter:
    corpus='memory'
    def __init__(self, exclude=()):
        self.exclude=set(exclude)
        self.epoch=epoch_value()
    def filters(self):
        return [MemoryTurn.deleted_at.is_(None), MemoryTurn.hidden_at.is_(None),
                MemoryTurn.status=='completed',
                MemoryTurn.retrieval_summary.is_not(None),MemoryTurn.turn_id.not_in(self.exclude)]
    @staticmethod
    def candidate(r,score):
        return RetrievedChunk('历史对话',r.retrieval_summary or '',score,'memory',
            {'content_hash':r.content_hash,'conversation_id':r.conversation_id},r.turn_id)
    def lexical(self,query,k):
        with session() as s:
            rows=list(s.scalars(select(MemoryTurn).where(*self.filters())))
            if not rows:return []
            idx=BM25Index([r.retrieval_summary for r in rows])
            return [self.candidate(rows[i],score) for i,score in idx.rank(query,k) if score>0]
    def vector(self,qvec,k):
        with session() as s:
            distance=MemoryTurn.summary_embedding.cosine_distance(qvec)
            rows=s.execute(select(MemoryTurn,distance).where(*self.filters(),MemoryTurn.summary_embedding.is_not(None),
                MemoryTurn.embedding_version==settings.retrieval_embedding_index_version).order_by(distance).limit(k))
            return [self.candidate(row,1-float(d)) for row,d in rows]
    def resolve(self,candidates):
        with session() as s,s.begin():
            if lock_profile(s).memory_epoch != self.epoch:return []
            rows={r.turn_id:r for r in s.scalars(select(MemoryTurn).where(MemoryTurn.turn_id.in_([x.chunk_id for x in candidates]),MemoryTurn.deleted_at.is_(None),MemoryTurn.hidden_at.is_(None)))}
            result=[]
            for hit in candidates:
                row=rows.get(hit.chunk_id)
                if row is None or row.content_hash!=hit.metadata['content_hash']:continue
                result.append(RetrievedChunk(hit.title,json.dumps(row.messages,ensure_ascii=False),hit.score,'memory',
                    {**hit.metadata,'memory_epoch':self.epoch,'source_ids':[m['message_id'] for m in row.messages]},row.turn_id))
            return result

def claim_event():
    with session() as s,s.begin():
        profile=lock_profile(s)
        row=s.scalar(select(MemoryOutbox).where(
            or_((MemoryOutbox.status.in_(['pending','retry'])) & (MemoryOutbox.next_attempt_at<=now()),
                (MemoryOutbox.status=='running') & (MemoryOutbox.lease_until<=now()))
            ).order_by(MemoryOutbox.next_attempt_at).with_for_update(skip_locked=True).limit(1))
        if not row:return None
        turn=s.get(MemoryTurn,row.turn_id)
        if not turn or turn.deleted_at or turn.hidden_at or turn.content_hash!=row.content_hash:
            row.status='cancelled'; return None
        if row.memory_epoch!=profile.memory_epoch:
            row.status='cancelled'; turn.memory_epoch=profile.memory_epoch; enqueue(s,turn); return None
        if row.attempts >= settings.memory_outbox_max_attempts:
            row.status='dead_letter'; row.last_error_code='attempts_exhausted'; row.lease_until=None
            if turn.index_status!='ready':turn.index_status='failed'
            return None
        identifier='scope:'+key('background:memory:'+row.turn_id+':'+row.content_hash)
        budget=s.get(ModelQuota,identifier)
        if budget is None or utc(budget.expires_at)<=now():
            row.status='dead_letter';row.last_error_code='job_budget_expired' if budget else 'legacy_budget_unknown';row.lease_until=None
            if turn.index_status!='ready':turn.index_status='failed'
            return None
        row.status='running'; row.attempts+=1; row.lease_until=now()+timedelta(seconds=180)
        return {'event_id':row.event_id,'turn_id':row.turn_id,'content_hash':row.content_hash,
                'epoch':row.memory_epoch,'attempt':row.attempts,'messages':turn.messages,
                'cached_summary':turn.retrieval_summary if turn.index_status in ('pending_vector','ready') else None,
                'conversation_id':turn.conversation_id}

def commit_index(event, summary, vector):
    with session() as s,s.begin():
        profile=lock_profile(s); task=s.get(MemoryOutbox,event['event_id']); row=s.get(MemoryTurn,event['turn_id'])
        if not task or task.status!='running' or task.attempts!=event['attempt']:return False
        if not row or row.deleted_at or row.hidden_at or row.content_hash!=event['content_hash'] or profile.memory_epoch!=event['epoch']:
            task.status='cancelled'
            if row and not row.deleted_at and not row.hidden_at and row.content_hash==event['content_hash']:
                row.memory_epoch=profile.memory_epoch; enqueue(s,row)
            return False
        row.retrieval_summary=summary; row.summary_embedding=vector
        row.embedding_version=settings.retrieval_embedding_index_version
        row.index_status='ready'; task.status='done'; task.lease_until=None
        from app.services.memory.wiki import materialize
        materialize(s,row)
        return True

def cache_index_summary(event, summary):
    with session() as s,s.begin():
        profile=lock_profile(s);task=s.get(MemoryOutbox,event['event_id']);turn=s.get(MemoryTurn,event['turn_id'])
        if (task is None or task.status!='running' or task.attempts!=event['attempt'] or
                turn is None or turn.deleted_at or turn.hidden_at or turn.content_hash!=event['content_hash'] or profile.memory_epoch!=event['epoch']):
            return False
        turn.retrieval_summary=summary;turn.index_status='pending_vector'
        return True


def complete_empty_index(event):
    """A valid empty summary completes the job without an embedding or retry."""
    with session() as s,s.begin():
        profile=lock_profile(s);task=s.get(MemoryOutbox,event['event_id']);turn=s.get(MemoryTurn,event['turn_id'])
        if (task is None or task.status!='running' or task.attempts!=event['attempt'] or
                turn is None or turn.deleted_at or turn.hidden_at or turn.content_hash!=event['content_hash'] or profile.memory_epoch!=event['epoch']):
            return False
        turn.retrieval_summary='';turn.summary_embedding=None;turn.embedding_version=None
        turn.index_status='skipped'
        task.status='done';task.lease_until=None;task.last_error_code=None
        from app.services.memory.wiki import materialize
        page=materialize(s,turn)
        if page is not None:
            page.summary_embedding=None;page.embedding_version=None
        return True


def retry_event(event,code,*,retryable=False):
    with session() as s,s.begin():
        row=s.get(MemoryOutbox,event['event_id'])
        if row and row.status=='running' and row.attempts==event['attempt']:
            exhausted=row.attempts>=settings.memory_outbox_max_attempts
            row.status='retry' if retryable and not exhausted else 'dead_letter'
            row.last_error_code=('attempts_exhausted:' if exhausted else '')+code[:55]
            row.next_attempt_at=now()+timedelta(seconds=min(300,2**min(row.attempts,8)))
            row.lease_until=None
            if row.status=='dead_letter':
                turn=s.get(MemoryTurn,row.turn_id)
                if turn and turn.content_hash==row.content_hash and turn.index_status!='ready':turn.index_status='failed'


async def consume_one():
    from app.services.memory.background import background_memory_enabled
    if not background_memory_enabled():return False
    event=await asyncio.to_thread(claim_event)
    if not event:return False
    identity='memory:'+event['turn_id']+':'+event['content_hash']
    with model_scope(identity,kind='background',owner_id=key('demo-user:'+event['conversation_id'])):
        try:
            from app.services.memory.deepseek_summary import deepseek_summary
            summary=event.get('cached_summary')
            if not summary:
                messages=[m for m in event['messages'] if m['role'] in ('user','assistant')]
                with model_stage('summary',2):
                    result=await asyncio.wait_for(deepseek_summary.summarize(messages),timeout=settings.memory_summary_timeout_seconds)
                summary='\n'.join(f"{i['speaker']}: {i['text']}" for i in result['items'])
                if not summary:
                    await asyncio.to_thread(complete_empty_index,event)
                    return True
                if not await asyncio.to_thread(cache_index_summary,event,summary):return True
            with model_stage('embedding',3):
                vector=(await asyncio.to_thread(embedding_client.embed,[summary]))[0]
            await asyncio.to_thread(commit_index,event,summary,vector)
        except Exception as exc:
            allowed,code=classify_memory_failure(exc)
            await asyncio.to_thread(retry_event,event,code,retryable=allowed)
    return True


def hide_conversation(cid):
    """User-facing soft delete: hide one conversation from every user read.

    Messages, responses, facts and wiki pages stay on disk; only reads filter
    ``hidden_at``. Physical cleanup remains ``delete_memory``. Unfinished managed
    write leases are revoked in the same real transaction as the privacy change.
    """
    with session() as s,s.begin():
        rows=list(s.scalars(select(MemoryTurn).where(MemoryTurn.conversation_id==cid,
            MemoryTurn.deleted_at.is_(None)).with_for_update()))
        stamp=now()
        for row in rows:
            if row.hidden_at is None:
                row.hidden_at=stamp
        from app.services.agent.react_checkpoint_store import ReactCheckpointStore
        ReactCheckpointStore(session).invalidate_in_session(
            s, user_id='demo-user', conversation_id=cid, purge=False)
        return len(rows)

def delete_memory(cid=None):
    with session() as s,s.begin():
        profile=lock_profile(s); profile.memory_epoch+=1
        q=select(MemoryTurn).where(MemoryTurn.deleted_at.is_(None))
        if cid is not None:q=q.where(MemoryTurn.conversation_id==cid)
        rows=list(s.scalars(q.with_for_update())); ids=[r.turn_id for r in rows]
        from app.db.quality_repository import scrub_in_session
        if cid is not None:
            scrub_in_session(s, cid)
        else:
            scrub_in_session(s, None, all_conversations=True)
        # Delete derived raw model results with the conversation, but retain
        # non-content quota counters so deletion cannot replenish spending.
        from app.db.model_budget_models import ModelResultCache
        from sqlalchemy import delete
        cache_query=delete(ModelResultCache)
        if cid is not None:cache_query=cache_query.where(ModelResultCache.owner_id==key('demo-user:'+cid))
        s.execute(cache_query)
        # Same-transaction privacy revoke/purge of the managed write chain
        # (lease, events, operations, tasks, requests) for the affected
        # conversations; real business rows and quota counters stay exact.
        from app.services.agent.react_checkpoint_store import ReactCheckpointStore
        ReactCheckpointStore(session).invalidate_in_session(
            s, user_id='demo-user', conversation_id=cid, purge=True)
        from app.services.memory.wiki import purge_sources
        purge_sources(s,ids)
        for row in rows:
            row.messages=[]; row.response=None; row.window_summary=None; row.retrieval_summary=None
            row.summary_embedding=None; row.deleted_at=now(); row.status='deleted'; row.index_status='deleted'
            row.operations=None
        for fact in s.scalars(select(UserFact).where(UserFact.source_turn_id.in_(ids))):
            s.delete(fact)
        for event in s.scalars(select(MemoryOutbox).where(MemoryOutbox.turn_id.in_(ids))):
            event.status='cancelled'
        return len(rows)

class CompactMemory:
    def __init__(self,legacy):
        self.legacy=legacy; self.task=None
        self.background_disabled_reason='startup_not_completed'
    def __getattr__(self,name):return getattr(self.legacy,name)
    def enabled(self):return settings.memory_read_backend=='compact_v1' and db.DB_READY
    def status(self):
        from app.services.memory.background import background_memory_enabled
        background={'background_enabled':background_memory_enabled(),
                    'background_worker_running':self.task is not None and not self.task.done(),
                    'background_disabled_reason':self.background_disabled_reason}
        if not self.enabled():
            if settings.memory_read_backend=='compact_v1' and settings.wiki_enabled:
                # Wiki/degraded keeps its honest persistence/status contract and
                # does not read legacy memory. The summary diagnostics are a
                # safe configuration view only (no model, message or Wiki read).
                from app.services.memory.deepseek_summary import deepseek_summary
                return {'backend':'postgresql-compact-v1','persistent':False,'status':'degraded',
                        'context_strategy':'wiki_addressed','window_turns':None,
                        'summary_model':deepseek_summary.status(),**background}
            # Keep legacy diagnostics and always append the background gate
            # fields so health still shows why the compact worker is absent.
            legacy=self.legacy.status()
            return {**legacy,**background} if isinstance(legacy,dict) else {**background,'legacy_status':legacy}
        return {'backend':'postgresql-compact-v1','persistent':db.DB_READY,'status':'ready' if db.DB_READY else 'degraded',
                'context_strategy':'wiki_addressed' if settings.wiki_enabled else 'recent_window',
                'window_turns':settings.memory_short_window_turns,
                'recent_max_tokens':settings.memory_recent_max_tokens,
                'embedding_configured':embedding_client.configured(),
                **background}
    async def initialize(self):return await self.legacy.initialize()
    async def start(self):
        from app.services.memory.background import background_memory_enabled
        self.background_disabled_reason=None
        if not background_memory_enabled():
            # Explicit config opt-out wins over every storage probe.
            self.background_disabled_reason='background_disabled_by_config'
            return
        if not (self.enabled() and db.DB_READY):
            self.background_disabled_reason='compact_memory_unavailable'
            return
        ready,reason=await asyncio.to_thread(probe_budget_storage)
        if not ready:
            # Fail closed: no worker, no claim, no attempt/status change, no
            # model call. One warning and one observable reason per startup;
            # the probe is not repeated until the process restarts.
            self.background_disabled_reason=reason
            logger.warning("background_memory_disabled reason=%s",reason)
            return
        await asyncio.to_thread(epoch_value)
        from app.services.memory.wiki import initialize, backfill
        await asyncio.to_thread(initialize)
        await asyncio.to_thread(backfill)
        self.task=asyncio.create_task(self._consume())
        logger.info("background_memory_worker_started")
    async def _consume(self):
        from app.services.memory.background import background_memory_enabled
        while background_memory_enabled():
            try:
                work=await consume_one()
            except Exception:
                work=False
            await asyncio.sleep(.1 if work else settings.memory_outbox_interval_seconds)
    async def close(self):
        if self.task:
            self.task.cancel(); await asyncio.gather(self.task,return_exceptions=True)
        await self.legacy.close()
    async def load(self,user_id,cid,query=''):
        empty={'short_memory':[],'long_memory':{},'semantic_memory':[],'memory_summary':'',
               'message_importance':{},'retrieved_history':[],'memory_backend':'postgresql-compact-v1'}
        if not self.enabled():
            if settings.memory_read_backend=='compact_v1' and settings.wiki_enabled:
                return {**empty,'degraded_reason':'memory_database_unavailable',
                        'wiki_error':'wiki_database_unavailable','context_strategy':'wiki_addressed'}
            return await self.legacy.load(user_id,cid)
        try:
            if settings.wiki_enabled:
                from app.services.memory.wiki_context import catalog
                # ChatRequest normalizes its owner to the single-person namespace.
                if user_id != 'demo-user':
                    return {**empty,'degraded_reason':'memory_owner_unavailable'}
                current_epoch=await asyncio.to_thread(epoch_value)
                from app.services.memory.lifecycle import snapshot
                view=await asyncio.to_thread(snapshot,cid,epoch=current_epoch)
                index=await asyncio.to_thread(catalog,cid,current_epoch,query,
                    settings.wiki_catalog_max_tokens,initial=True)
                from app.services.memory.recent_window import load_window
                recent=await asyncio.to_thread(load_window,cid,current_epoch)
                latest=(recent.get('recent_result_dialogue') if recent else
                        await asyncio.to_thread(latest_result_dialogue,cid,current_epoch))
                return {**empty,**view,'memory_epoch':current_epoch,
                    **recent,
                    'recent_result_dialogue':latest,
                    'wiki_catalog':index,'wiki_conversation_id':cid,'context_strategy':'wiki_addressed'}
            rows=await asyncio.to_thread(recent_rows,cid)
            short=recent_dialogue(rows)
            facts=await asyncio.to_thread(current_facts)
            current_epoch=await asyncio.to_thread(epoch_value)
            # SQL validates source/epoch on every load. Cache is expendable, never an authority.
            if await self.legacy.initialize():
                try:
                    await self.legacy.client.set(f'fitness:compact:window:{current_epoch}:{cid}',json.dumps(short,ensure_ascii=False),ex=settings.memory_short_ttl_seconds)
                except Exception:pass
            hits=[]; retrieval_error=None
            if query:
                try:
                    from app.services.memory.wiki import WikiAdapter
                    excluded = [r.turn_id for r in rows]
                    def retrieve():
                        # Adapter construction also reads SQL; keep it off the event loop.
                        adapter = (WikiAdapter if settings.wiki_enabled else MemoryAdapter)(excluded)
                        return retrieval_service.search(query, adapter, 3)
                    hits = await run_retrieval(retrieve)
                except Exception as exc:
                    retrieval_error=type(exc).__name__
            return {**empty,'degraded_reason':retrieval_error,'short_memory':short,'long_memory':facts,'memory_epoch':current_epoch,
                'retrieved_history':[{'turn_id':h.chunk_id,'content':h.content,'score':h.score,**h.metadata} for h in hits]}
        except Exception:
            return {**empty,'degraded_reason':'memory_database_or_retrieval_unavailable',
                **({'wiki_error':'wiki_database_unavailable'} if settings.wiki_enabled else {})}
    async def append_turn(self,user_id,cid,user,assistant):
        if not self.enabled():return await self.legacy.append_turn(user_id,cid,user,assistant)
        # Compatibility only. The live request path commits through
        # ``_finish_memory`` with the complete response in one transaction; an
        # answer-only call would create a completed turn that cannot be
        # replayed, so it is refused instead of split-committing.
        raise RuntimeError('compact_final_commit_requires_turn_response')
    async def remember_explicit(self,user_id,message):
        if not self.enabled():await self.legacy.remember_explicit(user_id,message)
    async def sync_profile(self,user_id,profile):
        if not self.enabled():await self.legacy.sync_profile(user_id,profile)
    async def load_history(self,user_id,cid):
        if not self.enabled():return await self.legacy.load_history(user_id,cid)
        def read():
            from app.db.quality_models import AnswerFeedback, ChatRun
            with session() as s:
                rows=list(s.scalars(select(MemoryTurn).where(MemoryTurn.conversation_id==cid,MemoryTurn.deleted_at.is_(None),MemoryTurn.hidden_at.is_(None)).order_by(MemoryTurn.started_at)))
                # Batch diagnostics/feedback lookups; never one query per message.
                request_ids=[row.request_id for row in rows]
                runs_by_request={}
                if request_ids:
                    # Ordered oldest→newest so a retried request always binds
                    # the newest completed diagnostic for feedback identity.
                    for run in s.scalars(select(ChatRun).where(
                            ChatRun.request_id.in_(request_ids),
                            ChatRun.status=='completed').order_by(ChatRun.created_at, ChatRun.run_id)):
                        runs_by_request[run.request_id]=run
                run_ids={run.run_id for run in runs_by_request.values()}
                feedback={}
                if run_ids:
                    feedback={item.run_id:item.rating for item in s.scalars(
                        select(AnswerFeedback).where(AnswerFeedback.run_id.in_(run_ids)))}
                messages=[]
                for row in rows:
                    response=row.response if isinstance(row.response,dict) else None
                    run=runs_by_request.get(row.request_id)
                    for message in row.messages:
                        if message.get('role') not in ('user','assistant'):
                            continue
                        item=dict(message)
                        if item['role']=='assistant':
                            # The completed answer keeps the identifiers it was
                            # committed with; nothing is generated here, and a
                            # failed/cancelled attempt can never contribute one.
                            # run_id is exposed only when the completed
                            # diagnostic exists, so the feedback button always
                            # has a usable identifier.
                            if run is not None:
                                item['run_id']=run.run_id
                                item['message_id']=(response or {}).get('message_id') or run.message_id
                            elif response is not None and response.get('message_id'):
                                item['message_id']=response['message_id']
                            citations=response.get('citations') if response is not None else (
                                run.citations if run is not None else None)
                            if citations is not None:
                                item['citations']=citations or []
                            if item.get('run_id'):
                                rating=feedback.get(item['run_id'])
                                if rating:
                                    item['feedback']=rating
                        messages.append(item)
                return messages
        return {'messages':await asyncio.to_thread(read),'memory_backend':'postgresql-compact-v1','max_messages':None,'ttl_seconds':None}
    async def list_conversations(self,user_id):
        if not self.enabled():return await self.legacy.list_conversations(user_id)
        def read():
            groups={}
            with session() as s:
                for row in s.scalars(select(MemoryTurn).where(MemoryTurn.deleted_at.is_(None),MemoryTurn.hidden_at.is_(None)).order_by(MemoryTurn.started_at)):
                    item=groups.setdefault(row.conversation_id,{'conversation_id':row.conversation_id,
                        'title':row.messages[0]['content'][:80],'created_at':row.started_at.isoformat(),'message_count':0})
                    item['updated_at']=(row.completed_at or row.started_at).isoformat(); item['message_count']+=sum(m['role'] in ('user','assistant') for m in row.messages)
            return sorted(groups.values(),key=lambda x:x['updated_at'],reverse=True)
        return await asyncio.to_thread(read)
    async def delete_conversation(self,user_id,cid):
        if not self.enabled():return await self.legacy.delete_conversation(user_id,cid)
        # User delete is a soft hide: keep the durable transcript for audit and
        # recovery, only stop showing it. ``delete_memory`` stays the explicit
        # physical purge path.
        count=await asyncio.to_thread(hide_conversation,cid)
        await self.legacy.delete_conversation(user_id,cid)
        from app.services.agent.food_calculator import _pending_food, _PENDING_LOCK
        with _PENDING_LOCK:_pending_food.pop(('demo-user',cid),None)
        return bool(count)
    async def restore_latest_history(self,user_id,cid):
        if not self.enabled():return await self.legacy.restore_latest_history(user_id,cid)
        return {**await self.load_history(user_id,cid),'recovered':False}

