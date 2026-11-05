from app.core import clock

from fastapi import APIRouter

from app.db.repositories import recovery_event_summary


router = APIRouter(tags=["recovery"])


@router.get("/recovery")
def get_recovery(user_id: str = "demo-user") -> dict:
    """根据真实训练事件给出轻量恢复摘要，不伪造睡眠或心率数据。

    一个 workout session 只计一次训练事件；只有未被同用户 session/exercise
    来源关系代表的 legacy 记录才作为兼容事件计一次。近期窗口不含未来日期，
    末次训练日期取全部有效历史，避免 7 天窗口丢掉更早记录。
    """
    today = clock.today()
    summary = recovery_event_summary(user_id, today=today)
    last_workout = summary.get("last_workout_date")
    rest_days = (today - last_workout).days if last_workout else None
    recent_count = int(summary.get("recent_events") or 0)

    if not last_workout:
        score, label = 70, "等待训练数据"
        recommendation = "先记录几次训练，恢复建议会随着你的实际节奏变得更准确。"
    elif rest_days == 0 and recent_count >= 4:
        score, label = 62, "建议放松"
        recommendation = "最近 7 天训练较密集，今天优先低强度活动或休息。"
    elif rest_days == 0:
        score, label = 76, "适合适度训练"
        recommendation = "今天有训练记录，后续留意动作质量和身体反馈。"
    elif rest_days is not None and rest_days >= 2:
        score, label = 88, "恢复窗口较好"
        recommendation = "距离上次训练已有休息时间，可以按计划逐步开始。"
    else:
        score, label = 82, "状态不错"
        recommendation = "训练间隔适中，保持热身和循序渐进即可。"

    return {
        "user_id": user_id,
        "score": score,
        "label": label,
        "recommendation": recommendation,
        "last_workout_date": last_workout.isoformat() if last_workout else None,
        "rest_days": rest_days,
        "recent_workouts": recent_count,
        "sleep_hours": None,
        "sleep_label": "未记录",
    }
