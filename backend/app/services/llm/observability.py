"""Read-only aggregate model accounting, including autonomous and CLI work."""
from collections import Counter
from datetime import datetime,time,timedelta,timezone
from zoneinfo import ZoneInfo
from sqlalchemy import select
from app.core import clock
from app.core.config import settings
from app.db.model_budget_models import ModelQuota,ModelCallLedger
from app.services.llm.budget import active_scope,BudgetStore,key,ModelBudgetError,daily_extra_calls


def model_budget_summary():
    moment=clock.utc_now()
    local=moment.astimezone(ZoneInfo('Asia/Shanghai'))
    start=datetime.combine(local.date(),time.min,tzinfo=local.tzinfo).astimezone(timezone.utc)
    scope=active_scope.get();store=scope.store if scope else BudgetStore()
    try:
        with store.session() as session:
            quota=session.get(ModelQuota,'daily:'+key(start.isoformat()))
            extra=daily_extra_calls(session,moment)
            effective=min(quota.max_calls,settings.model_daily_max_calls+extra) if quota else settings.model_daily_max_calls
            rows=list(session.scalars(select(ModelCallLedger).where(ModelCallLedger.started_at>=start)))
        return {'ready':True,'day':local.date().isoformat(),'timezone':'Asia/Shanghai',
                'daily_max_calls':effective,'daily_default_max_calls':settings.model_daily_max_calls,
                'daily_extra_calls':extra,'remaining_calls':max(0,effective-(quota.calls if quota else 0)),
                'reserved_attempts':quota.calls if quota else 0,'known_or_reserved_tokens':quota.tokens if quota else 0,
                'attempts_by_kind':dict(Counter(row.kind for row in rows)),
                'attempts_by_capability':dict(Counter(row.capability for row in rows)),
                'statuses':dict(Counter(row.status for row in rows)),
                'usage_unknown_attempts':sum(not row.usage for row in rows),
                'background_enabled':settings.memory_background_enabled and settings.memory_summary_enabled,
                'currency_cost':None,'note':'Reservations/attempts are not confirmed paid requests; unknown usage keeps its reservation.'}
    except Exception:return {'ready':False,'code':'model_budget_storage_unavailable','currency_cost':None}
