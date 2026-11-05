from __future__ import annotations

import math
import json
import re
import sys
from functools import lru_cache
from datetime import date, timedelta
from typing import Any

from sqlalchemy import func, select, text

from app.db import session as db_session
from app.db.workout_session_models import UserCustomExerciseModel


# Curated local starter library. User-created exercises live in PostgreSQL;
# usage statistics are derived from saved workout records so they cannot drift.
DEFAULT_EXERCISES: tuple[dict[str, str], ...] = (
    {"name": "杠铃卧推", "category": "胸部", "target_muscles": "胸大肌 · 肱三头肌"},
    {"name": "哑铃卧推", "category": "胸部", "target_muscles": "胸大肌 · 肱三头肌"},
    {"name": "上斜哑铃卧推", "category": "胸部", "target_muscles": "上胸 · 肱三头肌"},
    {"name": "上斜杠铃卧推", "category": "胸部", "target_muscles": "上胸 · 肱三头肌"},
    {"name": "哑铃飞鸟", "category": "胸部", "target_muscles": "胸大肌"},
    {"name": "绳索夹胸", "category": "胸部", "target_muscles": "胸大肌"},
    {"name": "蝴蝶机夹胸", "category": "胸部", "target_muscles": "胸大肌"},
    {"name": "俯卧撑", "category": "胸部", "target_muscles": "胸大肌 · 肱三头肌"},
    {"name": "双杠臂屈伸", "category": "胸部", "target_muscles": "下胸 · 肱三头肌"},
    {"name": "高位下拉", "category": "背部", "target_muscles": "背阔肌 · 肱二头肌"},
    {"name": "引体向上", "category": "背部", "target_muscles": "背阔肌 · 肱二头肌"},
    {"name": "坐姿划船", "category": "背部", "target_muscles": "背阔肌 · 菱形肌"},
    {"name": "杠铃划船", "category": "背部", "target_muscles": "背阔肌 · 菱形肌"},
    {"name": "单臂哑铃划船", "category": "背部", "target_muscles": "背阔肌 · 菱形肌"},
    {"name": "直臂下压", "category": "背部", "target_muscles": "背阔肌"},
    {"name": "面拉", "category": "背部", "target_muscles": "后束 · 上背"},
    {"name": "硬拉", "category": "背部", "target_muscles": "臀腿 · 下背"},
    {"name": "杠铃深蹲", "category": "腿部", "target_muscles": "股四头肌 · 臀大肌"},
    {"name": "高脚杯深蹲", "category": "腿部", "target_muscles": "股四头肌 · 臀大肌"},
    {"name": "腿举", "category": "腿部", "target_muscles": "股四头肌 · 臀大肌"},
    {"name": "腿屈伸", "category": "腿部", "target_muscles": "股四头肌"},
    {"name": "腿弯举", "category": "腿部", "target_muscles": "腘绳肌"},
    {"name": "罗马尼亚硬拉", "category": "腿部", "target_muscles": "腘绳肌 · 臀大肌"},
    {"name": "臀推", "category": "腿部", "target_muscles": "臀大肌"},
    {"name": "箭步蹲", "category": "腿部", "target_muscles": "股四头肌 · 臀大肌"},
    {"name": "保加利亚分腿蹲", "category": "腿部", "target_muscles": "股四头肌 · 臀大肌"},
    {"name": "站姿提踵", "category": "腿部", "target_muscles": "小腿"},
    {"name": "哑铃肩推", "category": "肩部", "target_muscles": "三角肌前束 · 肱三头肌"},
    {"name": "杠铃推举", "category": "肩部", "target_muscles": "三角肌 · 肱三头肌"},
    {"name": "哑铃侧平举", "category": "肩部", "target_muscles": "三角肌中束"},
    {"name": "反向飞鸟", "category": "肩部", "target_muscles": "三角肌后束"},
    {"name": "杠铃弯举", "category": "手臂", "target_muscles": "肱二头肌"},
    {"name": "哑铃弯举", "category": "手臂", "target_muscles": "肱二头肌"},
    {"name": "绳索下压", "category": "手臂", "target_muscles": "肱三头肌"},
    {"name": "仰卧臂屈伸", "category": "手臂", "target_muscles": "肱三头肌"},
    {"name": "平板支撑", "category": "核心", "target_muscles": "核心肌群"},
    {"name": "卷腹", "category": "核心", "target_muscles": "腹直肌"},
    {"name": "悬垂举腿", "category": "核心", "target_muscles": "腹肌 · 髋屈肌"},
    {"name": "跑步", "category": "有氧", "target_muscles": "心肺 · 下肢"},
    {"name": "椭圆机", "category": "有氧", "target_muscles": "心肺 · 全身"},
    {"name": "划船机", "category": "有氧", "target_muscles": "心肺 · 全身"},
    {"name": "动感单车", "category": "有氧", "target_muscles": "心肺 · 下肢"},
)


def _normalize_name(value: str) -> str:
    return " ".join(str(value or "").strip().split())


@lru_cache(maxsize=1)
def _postgres_casefold_parameters() -> dict:
    """PG16 LOWER differs from Python casefold (e.g. Straße / STRASSE).

    Translate single-character folds once; expand the uncommon multi-character
    folds only when present. No database functions, schema changes or locale
    assumptions are required.
    """
    singles, expansions = {}, {}
    for codepoint in range(sys.maxunicode + 1):
        char = chr(codepoint)
        folded = char.casefold()
        if folded != char:
            (singles if len(folded) == 1 else expansions)[char] = folded
    return {"fold_from": "".join(singles), "fold_to": "".join(singles.values()),
            "fold_expansions": json.dumps(expansions, ensure_ascii=False),
            "fold_pattern": "[" + re.escape("".join(expansions)) + "]"}


def _casefold_sql(value: str, dialect: str) -> str:
    if dialect == "sqlite":
        return f"fitzen_casefold({value})"
    translated = f"translate({value}, :fold_from, :fold_to)"
    return f"""CASE WHEN {value} ~ :fold_pattern THEN (
        SELECT string_agg(COALESCE(CAST(:fold_expansions AS jsonb)->>c.ch, c.ch), '' ORDER BY c.ord)
        FROM regexp_split_to_table({translated}, '') WITH ORDINALITY AS c(ch, ord)
    ) ELSE {translated} END"""


def list_exercises(user_id: str = "demo-user", search: str = "", limit: int = 200) -> list[dict[str, Any]]:
    user_id = str(user_id or "demo-user").strip() or "demo-user"
    search = str(search or "").strip().casefold()
    limit = min(max(int(limit), 1), 300)
    if not (db_session.DB_READY and db_session.SessionLocal is not None):
        return _local_exercises(user_id, search, limit)
    today = date.today()
    params = {"user_id": user_id, "search": search, "limit": limit,
              "since": today - timedelta(days=29), "today": today, "starter_count": len(DEFAULT_EXERCISES)}
    defaults = []
    for i, item in enumerate(DEFAULT_EXERCISES):
        defaults.append(f"(:name{i}, :key{i}, :category{i}, :muscles{i}, {i})")
        params.update({f"name{i}": item["name"], f"key{i}": item["name"].casefold(),
                       f"category{i}": item["category"], f"muscles{i}": item["target_muscles"]})
    try:
        with db_session.SessionLocal() as session:
            dialect = session.get_bind().dialect.name
            if dialect == "sqlite":
                raw = session.connection().connection.driver_connection
                raw.create_function("fitzen_casefold", 1, lambda value: value.casefold(), deterministic=True)
                raw.create_function("fitzen_exercise_score", 2,
                    lambda count, days: round(math.log1p(count) * 0.65 + min(days / 30, 1.0) * 0.35, 6),
                    deterministic=True)
                score = "fitzen_exercise_score(usage_count, recent_use_days)"
            else:
                params.update(_postgres_casefold_parameters())
                score = "ROUND(CAST(LN(1.0+usage_count)*0.65 + LEAST(recent_use_days/30.0,1.0)*0.35 AS NUMERIC),6)"
            fold = lambda value: _casefold_sql(value, dialect)
            haystack = fold("name || ' ' || category || ' ' || target_muscles")
            search_predicate = (f"instr({haystack}, :search)>0" if dialect == "sqlite"
                                else f"strpos({haystack}, :search)>0")
            query = f"""WITH defaults(name, name_key, category, target_muscles, sort_order) AS (
                VALUES {','.join(defaults)}
            ), custom_ranked AS (
                SELECT id, exercise_name AS name, {fold('exercise_name')} AS name_key,
                       category, target_muscles,
                       ROW_NUMBER() OVER (PARTITION BY {fold('exercise_name')} ORDER BY id DESC) AS row_num
                FROM user_custom_exercises WHERE user_id=:user_id
            ), custom AS (SELECT * FROM custom_ranked WHERE row_num=1), library AS (
                SELECT c.name, c.name_key, COALESCE(NULLIF(c.category,''),d.category,'自定义') AS category,
                       COALESCE(NULLIF(c.target_muscles,''),d.target_muscles,'') AS target_muscles,
                       'custom' AS source, COALESCE(d.sort_order,:starter_count+c.id) AS sort_order
                FROM custom c LEFT JOIN defaults d ON d.name_key=c.name_key
                UNION ALL
                SELECT d.name, d.name_key, d.category, d.target_muscles, 'local', d.sort_order
                FROM defaults d WHERE NOT EXISTS(SELECT 1 FROM custom c WHERE c.name_key=d.name_key)
            ), candidates AS (
                SELECT * FROM library WHERE :search='' OR {search_predicate}
            ), events AS (
                SELECT {fold('TRIM(e.exercise_name)')} AS name_key, s.session_date AS workout_day
                FROM workout_session_exercises e JOIN workout_sessions s ON s.id=e.session_id
                WHERE s.user_id=:user_id
                UNION ALL
                SELECT {fold('TRIM(w.exercise)')}, w.workout_date FROM workout_records w
                WHERE w.user_id=:user_id AND NOT EXISTS(
                    SELECT 1 FROM workout_sessions linked WHERE linked.source_record_id=w.id)
            ), usage AS (
                SELECT e.name_key, COUNT(*) AS usage_count,
                       SUM(CASE WHEN workout_day>=:since AND workout_day<=:today THEN 1 ELSE 0 END) AS recent_use_count,
                       COUNT(DISTINCT CASE WHEN workout_day>=:since AND workout_day<=:today THEN workout_day END) AS recent_use_days
                FROM events e JOIN candidates c ON c.name_key=e.name_key GROUP BY e.name_key
            ), counts AS (
                SELECT c.*, COALESCE(u.usage_count,0) AS usage_count,
                       COALESCE(u.recent_use_count,0) AS recent_use_count,
                       COALESCE(u.recent_use_days,0) AS recent_use_days
                FROM candidates c LEFT JOIN usage u ON u.name_key=c.name_key
            ) SELECT name, category, target_muscles, source, usage_count, recent_use_count,
                     recent_use_days, {score} AS rank_score
              FROM counts ORDER BY rank_score DESC, recent_use_days DESC, usage_count DESC, sort_order
              LIMIT :limit"""
            result = session.execute(text(query), params).mappings().all()
            return [{**row, "rank_score": float(row["rank_score"])} for row in result]
    except Exception:
        return _local_exercises(user_id, search, limit)


def _local_exercises(user_id: str = "demo-user", search: str = "", limit: int = 200) -> list[dict[str, Any]]:
    user_id = str(user_id or "demo-user").strip() or "demo-user"
    search = str(search or "").strip().casefold()
    limit = min(max(int(limit), 1), 300)
    exercises: dict[str, dict[str, Any]] = {
        item["name"].casefold(): {**item, "source": "local", "_sort_order": index}
        for index, item in enumerate(DEFAULT_EXERCISES)
    }
    stats = {}
    output: list[dict[str, Any]] = []
    for key, item in exercises.items():
        if search and search not in " ".join((item["name"], item["category"], item.get("target_muscles", ""))).casefold():
            continue
        usage = stats.get(key, {})
        usage_count = usage.get("usage_count", 0)
        recent_use_count = usage.get("recent_use_count", 0)
        recent_use_days = usage.get("recent_use_days", 0)
        # Blend lifetime popularity with the share of days used in the recent
        # 30-day window. The components are returned so users can see why an
        # exercise was recommended.
        score = math.log1p(usage_count) * 0.65 + min(recent_use_days / 30, 1.0) * 0.35
        output.append({
            **item,
            "usage_count": usage_count,
            "recent_use_count": recent_use_count,
            "recent_use_days": recent_use_days,
            "rank_score": round(score, 6),
        })
    output.sort(key=lambda item: (
        -item["rank_score"], -item["recent_use_days"], -item["usage_count"],
        item["_sort_order"], item["name"].casefold(),
    ))
    return [{key: value for key, value in item.items() if key != "_sort_order"} for item in output[:limit]]


def save_custom_exercise(
    user_id: str,
    exercise_name: str,
    *,
    category: str = "自定义",
    target_muscles: str = "",
) -> dict[str, Any]:
    if not (db_session.DB_READY and db_session.SessionLocal is not None):
        raise RuntimeError("数据库暂时不可用，无法保存自定义动作")
    name = _normalize_name(exercise_name)
    if not name:
        raise ValueError("动作名称不能为空")
    normalized_user = str(user_id or "demo-user").strip() or "demo-user"
    with db_session.SessionLocal() as session:
        row = session.scalar(select(UserCustomExerciseModel).where(
            UserCustomExerciseModel.user_id == normalized_user,
            func.lower(UserCustomExerciseModel.exercise_name) == name.casefold(),
        ))
        if row is None:
            row = UserCustomExerciseModel(user_id=normalized_user, exercise_name=name)
            session.add(row)
        row.exercise_name = name
        row.category = _normalize_name(category) or "自定义"
        row.target_muscles = _normalize_name(target_muscles)
        session.commit()
        session.refresh(row)
        return {
            "id": int(row.id), "name": row.exercise_name, "category": row.category,
            "target_muscles": row.target_muscles, "source": "custom",
        }
