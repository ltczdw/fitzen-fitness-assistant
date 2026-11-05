from datetime import date

from fastapi import APIRouter, HTTPException

from app.db.exercise_library_repository import list_exercises, save_custom_exercise
from app.db.storage import WorkoutAssociationError
from app.db.workout_session_repository import WorkoutMetricError, add_set, create_session, delete_exercise, delete_session, delete_set, list_sessions, update_session, update_set
from app.schemas.workout_session import CustomExercisePayload, WorkoutSessionPayload, WorkoutSetPayload, WorkoutSetUpdate


router = APIRouter(tags=["workout-sessions"])


@router.get("/workout-exercises")
def get_workout_exercises(user_id: str = "demo-user", search: str = "", limit: int = 200) -> dict:
    exercises = list_exercises(user_id, search, min(max(limit, 1), 300))
    return {"exercises": exercises, "count": len(exercises), "source": "local_library"}


@router.post("/workout-exercises/custom")
def post_custom_workout_exercise(payload: CustomExercisePayload) -> dict:
    try:
        exercise = save_custom_exercise(
            payload.user_id,
            payload.exercise_name,
            category=payload.category,
            target_muscles=payload.target_muscles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"saved": True, "exercise": exercise}


@router.get("/workout-sessions")
def get_workout_sessions(user_id: str = "demo-user", limit: int = 30, start_date: date | None = None, end_date: date | None = None) -> list[dict]:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="开始日期不能晚于结束日期")
    return list_sessions(user_id, min(max(limit, 1), 100), start_date, end_date)


@router.post("/workout-sessions")
def post_workout_session(payload: WorkoutSessionPayload) -> dict:
    return create_session(payload.model_dump())


@router.put("/workout-sessions/{session_id}")
def put_workout_session(session_id: int, payload: WorkoutSessionPayload, user_id: str = "demo-user") -> dict:
    try:
        updated = update_session(user_id, session_id, payload.model_dump())
    except WorkoutAssociationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="训练会话不存在或不属于当前用户")
    return updated


@router.delete("/workout-sessions/{session_id}")
def remove_workout_session(session_id: int, user_id: str = "demo-user") -> dict:
    try:
        removed = delete_session(user_id, session_id)
    except WorkoutAssociationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="训练会话不存在或不属于当前用户")
    return {"deleted": True, "id": session_id}


@router.post("/workout-sessions/{session_id}/exercises/{exercise_id}/sets")
def post_workout_set(session_id: int, exercise_id: int, payload: WorkoutSetPayload, user_id: str = "demo-user") -> dict:
    try:
        updated = add_set(user_id, session_id, exercise_id, payload.model_dump())
    except WorkoutMetricError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkoutAssociationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="workout exercise not found")
    return updated


@router.delete("/workout-session-sets/{set_id}")
def delete_workout_set(set_id: int, user_id: str = "demo-user") -> dict:
    try:
        updated = delete_set(user_id, set_id)
    except WorkoutAssociationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="训练组不存在或不属于当前用户")
    if updated.get("error"):
        raise HTTPException(status_code=409, detail=updated["error"])
    return updated

@router.patch("/workout-session-sets/{set_id}")
def patch_workout_set(set_id: int, payload: WorkoutSetUpdate, user_id: str = "demo-user") -> dict:
    try:
        updated = update_set(user_id, set_id, payload.model_dump(exclude_unset=True))
    except WorkoutMetricError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WorkoutAssociationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="训练组不存在或不属于当前用户")
    return updated


@router.delete("/workout-sessions/{session_id}/exercises/{exercise_id}")
def remove_workout_exercise(session_id: int, exercise_id: int, user_id: str = "demo-user") -> dict:
    try:
        removed = delete_exercise(user_id, session_id, exercise_id)
    except WorkoutAssociationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="训练动作不存在或不属于当前用户")
    return {"deleted": True, "session_id": session_id, "exercise_id": exercise_id}
