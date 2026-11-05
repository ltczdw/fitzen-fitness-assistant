"""Update existing library entries by identity; saved meals keep their snapshots."""
from typing import Literal
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from app.db import session as db
from app.db.food_repository import ensure_food_schema, normalize_food_name

def update_library_food(food_id: int, food_name: str, nutrition: dict, *, source: Literal["local", "custom"], user_id: str = "demo-user") -> dict | None:
    if not (db.DB_READY and db.SessionLocal is not None and ensure_food_schema()):
        raise RuntimeError("数据库暂时不可用，无法编辑食物")
    name=" ".join(food_name.strip().split())
    if not name:
        raise ValueError("食物名称不能为空")
    personal=source == "custom"
    table="user_custom_food" if personal else "food_nutrition"
    scope=" AND user_id=:user_id" if personal else ""
    params={"id":food_id,"name":name,"user_id":user_id.strip() or "demo-user",**nutrition}
    returned="id, food_name, energy_kcal, protein_g, carb_g, fat_g, fiber_g"
    returned+=", use_count, last_used_at" if personal else ", category, preparation_state"
    try:
        with db.SessionLocal() as session:
            previous=session.execute(text(f"SELECT food_name FROM {table} WHERE id=:id AND is_active=TRUE{scope}"),params).mappings().first()
            if previous is None:
                return None
            # Retain a seed identity after renaming, so startup does not recreate the old entry.
            if not personal and previous["food_name"] != name:
                session.execute(text("""INSERT INTO food_alias_index(food_id,alias_text,normalized_alias,alias_type)
                    VALUES(:id,:original,:normalized,'library_original')
                    ON CONFLICT(food_id,normalized_alias) DO UPDATE SET alias_type='library_original'"""),
                    {"id":food_id,"original":previous["food_name"],"normalized":normalize_food_name(previous["food_name"])})
            timestamp=", updated_at=CURRENT_TIMESTAMP" if personal else ""
            row=session.execute(text(f"""UPDATE {table} SET food_name=:name,energy_kcal=:kcal,protein_g=:protein_g,
                carb_g=:carbs_g,fat_g=:fat_g,fiber_g=:fiber_g{timestamp}
                WHERE id=:id AND is_active=TRUE{scope} RETURNING {returned}"""),params).mappings().first()
            if row is None:
                session.rollback()
                return None
            food={"id":int(row["id"]),"name":row["food_name"],"grams":100,"source":source,
                  "kcal":float(row["energy_kcal"]),"protein":float(row["protein_g"]),"carbs":float(row["carb_g"]),"fat":float(row["fat_g"]),"fiber":float(row["fiber_g"]),
                  "category":"自定义" if personal else row["category"],"preparation_state":"用户自定义" if personal else row["preparation_state"]}
            if personal:
                food.update(usage_count=int(row["use_count"] or 0),last_used_at=str(row["last_used_at"]) if row["last_used_at"] else None)
            session.commit()
            return food
    except IntegrityError as exc:
        raise ValueError("已存在同名食物，请换一个名称后保存") from exc
    except SQLAlchemyError as exc:
        raise RuntimeError("编辑食物失败，请稍后重试") from exc
