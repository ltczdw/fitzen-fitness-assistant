from fastapi import APIRouter, HTTPException

from app.db.workout_session_repository import create_session, delete_session, list_sessions, update_session
from app.schemas.workout_session import WorkoutSessionPayload


router = APIRouter(tags=["workout-sessions"])


@router.get("/workout-sessions")
def get_workout_sessions(user_id: str = "demo-user", limit: int = 30) -> list[dict]:
    return list_sessions(user_id, min(max(limit, 1), 100))


@router.post("/workout-sessions")
def post_workout_session(payload: WorkoutSessionPayload) -> dict:
    return create_session(payload.model_dump())


@router.put("/workout-sessions/{session_id}")
def put_workout_session(session_id: int, payload: WorkoutSessionPayload, user_id: str = "demo-user") -> dict:
    updated = update_session(user_id, session_id, payload.model_dump())
    if updated is None:
        raise HTTPException(status_code=404, detail="训练会话不存在或不属于当前用户")
    return updated


@router.delete("/workout-sessions/{session_id}")
def remove_workout_session(session_id: int, user_id: str = "demo-user") -> dict:
    if not delete_session(user_id, session_id):
        raise HTTPException(status_code=404, detail="训练会话不存在或不属于当前用户")
    return {"deleted": True, "id": session_id}
