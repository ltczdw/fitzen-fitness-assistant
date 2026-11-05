import json
from pathlib import Path

from sqlalchemy import select

from app.db import session as db_session
from app.db.storage import require_ready, storage_session
from app.db.models import TrainingPlanModel


DEFAULT_SESSIONS = [{"day": "周一", "name": "上肢力量", "items": ["卧推", "坐姿划船", "哑铃肩推"], "status": "已完成"}, {"day": "周三", "name": "下肢力量", "items": ["深蹲", "臀桥", "反向弓步"], "status": "今天"}, {"day": "周五", "name": "全身训练", "items": ["硬拉", "俯卧撑", "核心训练"], "status": "待开始"}]


def get_plan(user_id: str = "demo-user") -> dict:
    require_ready()
    with storage_session() as session:
        row = session.scalars(select(TrainingPlanModel).where(TrainingPlanModel.user_id == user_id)).first()
        if row: return _serialize(row)
    return {"user_id": user_id, "plan_name": "", "goal": "", "days_per_week": 0, "sessions": []}


def get_stored_plan(user_id: str = "demo-user") -> dict | None:
    """Return only a real stored plan row; never project defaults as user data."""
    require_ready()
    with storage_session() as session:
        row = session.scalars(select(TrainingPlanModel).where(TrainingPlanModel.user_id == user_id)).first()
        return _serialize(row) if row else None


def save_plan(user_id: str, values: dict) -> dict:
    item = {**get_plan(user_id), **values, "user_id": user_id}
    require_ready()
    with storage_session() as session:
        row = session.scalars(select(TrainingPlanModel).where(TrainingPlanModel.user_id == user_id)).first()
        if row is None: row = TrainingPlanModel(user_id=user_id); session.add(row)
        row.plan_name = item.get("plan_name", "基础全身训练"); row.goal = item.get("goal", "保持健康"); row.days_per_week = int(item.get("days_per_week", 3)); row.plan_json = json.dumps(item.get("sessions", DEFAULT_SESSIONS), ensure_ascii=False)
        session.commit(); session.refresh(row); return _serialize(row)


def _serialize(row: TrainingPlanModel) -> dict:
    try: sessions = json.loads(row.plan_json)
    except json.JSONDecodeError: sessions = DEFAULT_SESSIONS
    return {"user_id": row.user_id, "plan_name": row.plan_name, "goal": row.goal, "days_per_week": row.days_per_week, "sessions": sessions}




