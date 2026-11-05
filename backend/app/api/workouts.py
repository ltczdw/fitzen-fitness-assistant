from datetime import date

from fastapi import APIRouter, HTTPException

from app.db.repositories import create_workout, delete_workout, list_workouts, update_workout
from app.db.storage import WorkoutAssociationError, WorkoutDetailRequired
from app.schemas.workout import WorkoutRecord, WorkoutRecordPayload


router = APIRouter(tags=["workouts"])


@router.get("/workouts")
def get_workouts(user_id: str = "demo-user", limit: int = 30, start_date: date | None = None, end_date: date | None = None) -> list[dict]:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="开始日期不能晚于结束日期")
    return list_workouts(min(max(limit, 1), 100), user_id, start_date, end_date)


@router.post("/workouts")
def post_workout(record: WorkoutRecord) -> dict:
    return create_workout(**record.model_dump())


@router.put("/workouts/{record_id}")
def put_workout(record_id: int, record: WorkoutRecordPayload, user_id: str = "demo-user") -> dict:
    try:
        # Only explicitly submitted fields may change; omitted metadata must
        # not be reset by schema defaults.
        updated = update_workout(user_id, record_id, record.model_dump(exclude_unset=True))
    except (WorkoutDetailRequired, WorkoutAssociationError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="训练记录不存在或不属于当前用户")
    return updated


@router.delete("/workouts/{record_id}")
def remove_workout(record_id: int, user_id: str = "demo-user") -> dict:
    try:
        removed = delete_workout(user_id, record_id)
    except WorkoutAssociationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="训练记录不存在或不属于当前用户")
    return {"deleted": True, "id": record_id}
