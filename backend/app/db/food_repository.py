from __future__ import annotations
from app.core import clock

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db import session as db_session
from app.db.storage import require_ready, storage_session, CommitUnknownError, IdempotencyConflict, recover_committed


CALCULATION_VERSION = "food-calc-v1"


@dataclass(frozen=True)
class FoodMatch:
    food_id: int
    food_name: str
    category: str
    preparation_state: str
    energy_kcal: float
    protein_g: float
    carb_g: float
    fat_g: float
    fiber_g: float
    match_method: str
    confidence: float
    source_version: str


SEED_FOODS = (
    ("熟大米", "谷薯类", "熟", 116, 2.60, 25.90, 0.30, 0.30),
    ("白粥", "谷薯类", "熟", 46, 1.10, 9.50, 0.10, 0.20),
    ("熟红薯", "谷薯类", "熟", 76, 1.10, 17.70, 0.20, 1.60),
    ("熟玉米", "谷薯类", "熟", 112, 4.00, 22.80, 1.20, 2.90),
    ("全麦面包", "谷薯类", "即食", 246, 8.50, 46.00, 3.50, 6.00),
    ("鸡蛋", "蛋类", "熟", 144, 12.60, 1.10, 9.50, 0),
    ("纯牛奶", "乳类", "液态", 54, 3.00, 3.40, 3.20, 0),
    ("鸡胸肉", "畜禽肉类", "熟", 167, 31.00, 0, 3.60, 0),
    ("瘦牛肉", "畜禽肉类", "熟", 194, 26.70, 0, 9.40, 0),
    ("三文鱼", "鱼虾类", "熟", 208, 22.10, 0, 12.40, 0),
    ("北豆腐", "豆制品", "熟", 84, 6.60, 3.40, 5.30, 0.40),
    ("燕麦", "谷薯类", "干", 377, 12.60, 66.30, 7.30, 5.30),
    ("香蕉", "水果类", "生鲜", 93, 1.40, 22.00, 0.20, 1.20),
    ("苹果", "水果类", "生鲜", 53, 0.40, 13.70, 0.20, 1.70),
    ("蓝莓", "水果类", "生鲜", 57, 0.70, 14.50, 0.30, 2.40),
    ("西兰花", "蔬菜类", "熟", 36, 4.10, 4.30, 0.60, 4.40),
    # Curated per-100g reference values for common foods. Preparation state is
    # explicit so cooked and raw values are not silently mixed.
    ("熟面条", "谷薯类", "熟", 137, 4.50, 25.30, 1.00, 1.80),
    ("馒头", "谷薯类", "熟", 223, 7.00, 47.00, 1.10, 1.30),
    ("普通面包", "谷薯类", "即食", 278, 8.30, 50.00, 5.00, 2.70),
    ("土豆", "薯芋类", "熟", 87, 1.90, 20.10, 0.10, 1.80),
    ("猪里脊肉", "畜禽肉类", "熟", 143, 26.70, 0, 4.70, 0),
    ("鸡腿肉", "畜禽肉类", "熟", 209, 26.00, 0, 10.90, 0),
    ("虾仁", "鱼虾类", "熟", 99, 23.00, 0.20, 0.30, 0),
    ("鳕鱼", "鱼虾类", "熟", 105, 23.00, 0, 0.90, 0),
    ("鸡蛋白", "蛋类", "熟", 52, 10.90, 0.70, 0.20, 0),
    ("低脂牛奶", "乳类", "液态", 46, 3.40, 4.80, 1.50, 0),
    ("无糖酸奶", "乳类", "即食", 62, 3.50, 5.00, 3.00, 0),
    ("无糖豆浆", "豆制品", "液态", 31, 3.00, 1.20, 1.60, 0.30),
    ("菠菜", "蔬菜类", "熟", 23, 2.90, 3.60, 0.40, 2.00),
    ("生菜", "蔬菜类", "生鲜", 15, 1.40, 2.90, 0.20, 1.30),
    ("黄瓜", "蔬菜类", "生鲜", 15, 0.70, 2.90, 0.10, 0.50),
    ("番茄", "蔬菜类", "生鲜", 18, 0.90, 3.90, 0.20, 1.20),
    ("胡萝卜", "蔬菜类", "熟", 35, 0.80, 8.20, 0.20, 3.00),
    ("鲜香菇", "蔬菜类", "生鲜", 27, 2.20, 5.00, 0.30, 3.30),
    ("橙子", "水果类", "生鲜", 47, 0.90, 11.75, 0.10, 2.40),
    ("梨", "水果类", "生鲜", 57, 0.40, 15.20, 0.10, 3.10),
    ("葡萄", "水果类", "生鲜", 69, 0.72, 18.10, 0.16, 0.90),
    ("草莓", "水果类", "生鲜", 32, 0.67, 7.68, 0.30, 2.00),
    ("食用植物油", "油脂类", "即食", 884, 0, 0, 100, 0),
    ("橄榄油", "油脂类", "即食", 884, 0, 0, 100, 0),
    ("花生油", "油脂类", "即食", 884, 0, 0, 100, 0),
    ("芝麻油", "油脂类", "即食", 884, 0, 0, 100, 0),
    ("花生", "坚果类", "熟", 567, 25.80, 16.10, 49.20, 8.50),
    ("杏仁", "坚果类", "熟", 579, 21.20, 21.60, 49.90, 12.50),
)
SEED_ALIASES = {
    "白饭": "熟大米", "米饭": "熟大米", "大米饭": "熟大米", "饭": "熟大米", "粥": "白粥", "白粥": "白粥",
    "红薯": "熟红薯", "地瓜": "熟红薯", "玉米": "熟玉米",
    "全麦吐司": "全麦面包", "面": "熟面条", "面条": "熟面条", "挂面": "熟面条", "拉面": "熟面条",
    "馒头": "馒头", "白面包": "普通面包", "面包": "普通面包", "土豆": "土豆", "马铃薯": "土豆",
    "蛋": "鸡蛋", "蛋白": "鸡蛋白", "牛奶": "纯牛奶", "奶": "纯牛奶", "低脂奶": "低脂牛奶", "酸奶": "无糖酸奶",
    "豆浆": "无糖豆浆", "鸡胸": "鸡胸肉", "牛肉": "瘦牛肉", "猪里脊": "猪里脊肉", "里脊肉": "猪里脊肉",
    "鸡腿": "鸡腿肉", "虾": "虾仁", "鳕鱼": "鳕鱼", "豆腐": "北豆腐", "燕麦片": "燕麦", "花椰菜": "西兰花",
    "菠菜": "菠菜", "生菜": "生菜", "黄瓜": "黄瓜", "西红柿": "番茄", "番茄": "番茄", "胡萝卜": "胡萝卜", "香菇": "鲜香菇",
    "橙": "橙子", "橙子": "橙子", "梨": "梨", "葡萄": "葡萄", "草莓": "草莓",
    "植物油": "食用植物油", "食用油": "食用植物油", "菜油": "食用植物油", "橄榄油": "橄榄油", "花生油": "花生油", "芝麻油": "芝麻油",
    "花生": "花生", "杏仁": "杏仁"
}
SEED_PORTIONS = {
    # 份量是常见食用规格的估算值；用户仍可在记录明细中改成准确克重。
    "熟大米": {"碗": 150, "份": 150, "盘": 200}, "白粥": {"碗": 250, "份": 250}, "熟红薯": {"个": 200, "份": 200},
    "熟玉米": {"根": 180, "个": 180, "份": 180}, "全麦面包": {"片": 30, "份": 60},
    "熟面条": {"碗": 200, "份": 200, "盘": 250}, "馒头": {"个": 100, "份": 100},
    "普通面包": {"片": 35, "份": 70}, "土豆": {"个": 150, "份": 150},
    "鸡蛋": {"个": 50, "枚": 50, "份": 50}, "鸡蛋白": {"个": 30, "枚": 30, "份": 30},
    "纯牛奶": {"杯": 250, "盒": 250, "瓶": 250}, "低脂牛奶": {"杯": 250, "盒": 250, "瓶": 250},
    "无糖酸奶": {"杯": 200, "盒": 200}, "无糖豆浆": {"杯": 250, "瓶": 250},
    "香蕉": {"根": 120, "个": 120}, "苹果": {"个": 200, "份": 200}, "橙子": {"个": 180, "份": 180},
    "梨": {"个": 200, "份": 200}, "葡萄": {"份": 150, "把": 80}, "草莓": {"份": 150, "盒": 250},
    "鸡胸肉": {"块": 150, "份": 150}, "猪里脊肉": {"份": 100, "块": 100},
    "鸡腿肉": {"块": 120, "份": 120}, "虾仁": {"份": 100, "盘": 150}, "鳕鱼": {"块": 150, "份": 150},
    "三文鱼": {"块": 150, "份": 150}, "瘦牛肉": {"份": 100, "块": 100}, "北豆腐": {"块": 100, "份": 100},
    "燕麦": {"勺": 30, "份": 40, "袋": 40}, "西兰花": {"份": 150, "盘": 150},
    "菠菜": {"份": 150, "盘": 150}, "生菜": {"份": 100, "盘": 100}, "黄瓜": {"根": 150, "个": 150},
    "番茄": {"个": 150, "份": 150}, "胡萝卜": {"根": 100, "个": 100}, "鲜香菇": {"份": 100, "盘": 150},
    "食用植物油": {"勺": 10, "份": 10}, "橄榄油": {"勺": 10, "份": 10},
    "花生油": {"勺": 10, "份": 10}, "芝麻油": {"勺": 10, "份": 10},
    "花生": {"把": 25, "份": 25}, "杏仁": {"把": 25, "份": 25}
}

def normalize_food_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = value.replace("公斤", "kg").replace("千克", "kg")
    return re.sub(r"\s+", "", value)


def _embedding(value: str, dimensions: int = 256) -> list[float]:
    normalized = normalize_food_name(value)
    tokens = list(normalized) + [normalized[index : index + 2] for index in range(len(normalized) - 1)]
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / norm for item in vector]


def migrate_food_schema(connection) -> None:
    """Create and seed the food tables during application startup only.

    Schema changes must not run from request handlers. PostgreSQL ALTER TABLE
    takes an exclusive lock and can make every nutrition request wait behind
    an unrelated read transaction.
    """
    statements = (
        """CREATE TABLE IF NOT EXISTS food_nutrition (
            id BIGSERIAL PRIMARY KEY, food_name VARCHAR(200) NOT NULL,
            category VARCHAR(100) NOT NULL DEFAULT '其他', preparation_state VARCHAR(50) NOT NULL DEFAULT '熟',
            energy_kcal NUMERIC(10,2) NOT NULL, protein_g NUMERIC(10,2) NOT NULL,
            carb_g NUMERIC(10,2) NOT NULL, fat_g NUMERIC(10,2) NOT NULL, fiber_g NUMERIC(10,2) NOT NULL DEFAULT 0,
            source_name VARCHAR(200) NOT NULL DEFAULT '公开汇总版中国食物成分表', source_version VARCHAR(100) NOT NULL DEFAULT 'public-summary-v1',
            is_active BOOLEAN NOT NULL DEFAULT TRUE, UNIQUE(food_name, preparation_state)
        )""",
        """CREATE TABLE IF NOT EXISTS food_alias_index (
            id BIGSERIAL PRIMARY KEY, food_id BIGINT NOT NULL REFERENCES food_nutrition(id),
            alias_text VARCHAR(200) NOT NULL, normalized_alias VARCHAR(200) NOT NULL,
            alias_type VARCHAR(50) NOT NULL DEFAULT 'manual', embedding vector(256), embedding_json TEXT,
            confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0, is_active BOOLEAN NOT NULL DEFAULT TRUE,
            UNIQUE(food_id, normalized_alias)
        )""",
        """CREATE TABLE IF NOT EXISTS food_portion_reference (
            id BIGSERIAL PRIMARY KEY, food_id BIGINT NOT NULL REFERENCES food_nutrition(id),
            portion_name VARCHAR(50) NOT NULL, weight_g NUMERIC(10,2) NOT NULL,
            source VARCHAR(200) NOT NULL DEFAULT 'public-portion-estimate', UNIQUE(food_id, portion_name)
        )""",
        """CREATE TABLE IF NOT EXISTS user_food_log (
            id BIGSERIAL PRIMARY KEY, user_id VARCHAR(100) NOT NULL, meal_group_id VARCHAR(100) NOT NULL,
            record_time TIMESTAMPTZ NOT NULL, meal_type VARCHAR(20), food_id BIGINT REFERENCES food_nutrition(id),
            raw_food_name VARCHAR(200) NOT NULL, food_name VARCHAR(200) NOT NULL, weight_g NUMERIC(10,2) NOT NULL,
            total_kcal NUMERIC(10,2) NOT NULL, total_protein NUMERIC(10,2) NOT NULL,
            total_carb NUMERIC(10,2) NOT NULL, total_fat NUMERIC(10,2) NOT NULL, total_fiber NUMERIC(10,2) NOT NULL DEFAULT 0,
            is_estimated BOOLEAN NOT NULL DEFAULT FALSE, weight_source VARCHAR(50) NOT NULL DEFAULT 'user_input',
            match_method VARCHAR(50) NOT NULL DEFAULT 'exact', match_confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            idempotency_key VARCHAR(150) NOT NULL, calculation_version VARCHAR(100) NOT NULL DEFAULT 'food-calc-v1',
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_id, idempotency_key)
        )""",
        """CREATE TABLE IF NOT EXISTS user_custom_food (
            id BIGSERIAL PRIMARY KEY,
            user_id VARCHAR(100) NOT NULL,
            food_name VARCHAR(200) NOT NULL,
            energy_kcal NUMERIC(10,2) NOT NULL,
            protein_g NUMERIC(10,2) NOT NULL,
            carb_g NUMERIC(10,2) NOT NULL,
            fat_g NUMERIC(10,2) NOT NULL,
            fiber_g NUMERIC(10,2) NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMPTZ,
            use_count INTEGER NOT NULL DEFAULT 0,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            UNIQUE(user_id, food_name)
        )""",
        "ALTER TABLE user_food_log ADD COLUMN IF NOT EXISTS meal_type VARCHAR(20)",
        "CREATE INDEX IF NOT EXISTS ix_user_custom_food_usage ON user_custom_food(user_id, use_count DESC, last_used_at DESC)",
    )
    for statement in statements:
        connection.execute(text(statement))
    _seed_catalog(connection)


def ensure_food_schema() -> bool:
    """Return whether the startup migration completed.

    Kept as a compatibility guard for existing callers. It intentionally
    performs no DDL and is safe to call from read/write requests.
    """
    return bool(db_session.DB_READY and db_session.engine is not None and db_session.SessionLocal is not None)


def _seed_catalog(connection) -> None:
    for food_name, category, state, kcal, protein, carb, fat, fiber in SEED_FOODS:
        connection.execute(text("""INSERT INTO food_nutrition (food_name, category, preparation_state, energy_kcal, protein_g, carb_g, fat_g, fiber_g) SELECT CAST(:name AS VARCHAR(200)),CAST(:category AS VARCHAR(100)),CAST(:state AS VARCHAR(50)),CAST(:kcal AS NUMERIC(10,2)),CAST(:protein AS NUMERIC(10,2)),CAST(:carb AS NUMERIC(10,2)),CAST(:fat AS NUMERIC(10,2)),CAST(:fiber AS NUMERIC(10,2)) WHERE NOT EXISTS (SELECT 1 FROM food_alias_index a JOIN food_nutrition f ON f.id=a.food_id WHERE a.alias_type='library_original' AND a.normalized_alias=:original AND f.preparation_state=CAST(:state AS VARCHAR(50))) ON CONFLICT (food_name, preparation_state) DO NOTHING"""), {"name": food_name, "category": category, "state": state, "kcal": kcal, "protein": protein, "carb": carb, "fat": fat, "fiber": fiber, "original": normalize_food_name(food_name)})
    rows = connection.execute(text("SELECT id, food_name FROM food_nutrition WHERE is_active=TRUE")).mappings().all()
    by_name = {row["food_name"]: row["id"] for row in rows}
    for alias, standard in SEED_ALIASES.items():
        food_id = by_name.get(standard)
        if not food_id:
            continue
        vector = _embedding(alias)
        vector_text = "[" + ",".join(str(item) for item in vector) + "]"
        connection.execute(text("""INSERT INTO food_alias_index (food_id, alias_text, normalized_alias, embedding, embedding_json) VALUES (:food_id,:alias,:normalized,CAST(:vector AS vector),:json) ON CONFLICT (food_id, normalized_alias) DO NOTHING"""), {"food_id": food_id, "alias": alias, "normalized": normalize_food_name(alias), "vector": vector_text, "json": json.dumps(vector)})
    for standard, portions in SEED_PORTIONS.items():
        food_id = by_name.get(standard)
        if not food_id:
            continue
        for portion_name, weight_g in portions.items():
            connection.execute(text("INSERT INTO food_portion_reference (food_id, portion_name, weight_g) VALUES (:food_id,:portion,:weight) ON CONFLICT (food_id, portion_name) DO NOTHING"), {"food_id": food_id, "portion": portion_name, "weight": weight_g})


def _to_match(row: dict[str, Any], method: str, confidence: float) -> FoodMatch:
    return FoodMatch(int(row["id"]), row["food_name"], row["category"], row["preparation_state"], float(row["energy_kcal"]), float(row["protein_g"]), float(row["carb_g"]), float(row["fat_g"]), float(row["fiber_g"]), method, float(confidence), row["source_version"])


def resolve_food(food_name: str) -> FoodMatch | None:
    requested = normalize_food_name(food_name)
    if db_session.DB_READY and db_session.SessionLocal is not None:
        ensure_food_schema()
        try:
            with storage_session() as session:
                row = session.execute(text("SELECT f.* FROM food_nutrition f WHERE f.is_active=TRUE AND (lower(f.food_name)=:name OR lower(replace(f.food_name,' ',''))=:name) LIMIT 1"), {"name": requested}).mappings().first()
                if row:
                    return _to_match(row, "exact", 1.0)
                alias = session.execute(text("SELECT f.*, a.confidence FROM food_alias_index a JOIN food_nutrition f ON f.id=a.food_id WHERE a.is_active=TRUE AND f.is_active=TRUE AND a.normalized_alias=:name LIMIT 1"), {"name": requested}).mappings().first()
                if alias:
                    return _to_match(alias, "alias_exact", float(alias.get("confidence", 1.0)))
                vector_text = "[" + ",".join(str(item) for item in _embedding(food_name)) + "]"
                candidates = session.execute(text("SELECT f.*, a.confidence, 1 - (a.embedding <=> CAST(:vector AS vector)) AS similarity FROM food_alias_index a JOIN food_nutrition f ON f.id=a.food_id WHERE a.is_active=TRUE AND f.is_active=TRUE AND a.embedding IS NOT NULL ORDER BY a.embedding <=> CAST(:vector AS vector) LIMIT 2"), {"vector": vector_text}).mappings().all()
                if candidates:
                    best = candidates[0]
                    similarity = float(best["similarity"])
                    margin = similarity - float(candidates[1]["similarity"]) if len(candidates) > 1 else 1.0
                    if similarity >= 0.70 and margin >= 0.03:
                        return _to_match(best, "alias_vector", similarity)
        except Exception:
            pass
    return _fallback_match(food_name)


def _fallback_match(food_name: str) -> FoodMatch | None:
    requested = normalize_food_name(food_name)
    standard = SEED_ALIASES.get(food_name) or SEED_ALIASES.get(requested, food_name)
    standard_norm = normalize_food_name(standard)
    for index, item in enumerate(SEED_FOODS, start=1):
        if normalize_food_name(item[0]) == requested or normalize_food_name(item[0]) == standard_norm:
            return FoodMatch(index, item[0], item[1], item[2], *map(float, item[3:]), "fallback_exact", 1.0, "public-summary-v1")
    return None


def portion_weight(food_id: int, portion_name: str) -> float | None:
    portion_name = {"大碗": "碗", "小碗": "碗", "汤匙": "勺", "茶匙": "勺"}.get(portion_name, portion_name)
    if db_session.DB_READY and db_session.SessionLocal is not None:
        ensure_food_schema()
        try:
            with storage_session() as session:
                row = session.execute(text("SELECT weight_g FROM food_portion_reference WHERE food_id=:food_id AND portion_name=:portion LIMIT 1"), {"food_id": food_id, "portion": portion_name}).first()
                if row:
                    return float(row[0])
        except Exception:
            pass
    return None


def portion_references(food_name: str) -> list[dict[str, float | str]]:
    """Return the default non-gram units used by the assistant and UI."""
    match = resolve_food(food_name)
    if match is None:
        return []
    result: dict[str, float] = {}
    if db_session.DB_READY and db_session.SessionLocal is not None and match.food_id:
        try:
            with storage_session() as session:
                rows = session.execute(
                    text("SELECT portion_name, weight_g FROM food_portion_reference WHERE food_id=:food_id ORDER BY id"),
                    {"food_id": match.food_id},
                ).mappings().all()
                result.update({str(row["portion_name"]): float(row["weight_g"]) for row in rows})
        except Exception:
            pass
    if not result:
        result.update({str(unit): float(weight) for unit, weight in SEED_PORTIONS.get(match.food_name, {}).items()})
    return [{"unit": unit, "weight_g": weight} for unit, weight in result.items()]


def list_food_catalog(search: str = "", limit: int = 200, user_id: str = "") -> list[dict[str, Any]]:
    query = normalize_food_name(search)
    bounded_limit = min(max(int(limit), 1), 500)
    if db_session.DB_READY and db_session.SessionLocal is not None and ensure_food_schema():
        try:
            from app.db.food_log_listing import query_food_catalog
            with storage_session() as session:
                return query_food_catalog(session, user_id or "", clock.utc_now() - timedelta(days=90),
                                          bounded_limit, query)
        except Exception:
            pass
    usage: dict[str, tuple[int, str]] = {}
    result = [
        {"name": item[0], "category": item[1], "preparation_state": item[2], "kcal": float(item[3]),
         "protein": float(item[4]), "carbs": float(item[5]), "fat": float(item[6]), "fiber": float(item[7])}
        for item in SEED_FOODS
        if not query or query in normalize_food_name(item[0]) or query in normalize_food_name(item[1])
    ]
    for item in result:
        count, latest = usage.get(normalize_food_name(item["name"]), (0, ""))
        item["usage_count"] = count
        item["_last_used"] = latest
    result.sort(key=lambda item: (item["category"], item["name"]))
    result.sort(key=lambda item: item["_last_used"], reverse=True)
    result.sort(key=lambda item: int(item["usage_count"]), reverse=True)
    for item in result:
        item.pop("_last_used", None)
    return result[:bounded_limit]


def _insert_food_items(user_id: str, items: list[dict[str, Any]], totals: dict[str, float], meal_group_id: str, idempotency_key: str, record_time: datetime | None = None, meal_type: str | None = None) -> list[int]:
    record_time = record_time or clock.utc_now()
    require_ready()
    with storage_session() as session:
        if getattr(getattr(db_session, "engine", None), "dialect", None) and db_session.engine.dialect.name == "postgresql":
            session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
                {"scope": f"{user_id}:{idempotency_key}"},
            )
        existing = _find_food_items(session, user_id, idempotency_key, len(items))
        if existing is not None:
            _assert_food_key_matches(
                _find_food_rows(session, user_id, idempotency_key, len(items)), items,
                idempotency_key)
            return existing
        ids: list[int] = []
        for index, item in enumerate(items):
            item_key = f"{idempotency_key}:{index}"
            row = session.execute(
                text("""INSERT INTO user_food_log (user_id,meal_group_id,record_time,meal_type,food_id,raw_food_name,food_name,weight_g,total_kcal,total_protein,total_carb,total_fat,total_fiber,is_estimated,weight_source,match_method,match_confidence,idempotency_key,calculation_version) VALUES (:user_id,:meal_group_id,:record_time,:meal_type,:food_id,:raw_food_name,:food_name,:weight_g,:kcal,:protein,:carb,:fat,:fiber,:estimated,:weight_source,:match_method,:confidence,:key,:version) ON CONFLICT DO NOTHING RETURNING id"""),
                {"user_id": user_id, "meal_group_id": meal_group_id, "record_time": record_time, "meal_type": meal_type, "food_id": item["food_id"], "raw_food_name": item["raw_food_name"], "food_name": item["food_name"], "weight_g": item["weight_g"], "kcal": item["energy_kcal"], "protein": item["protein_g"], "carb": item["carb_g"], "fat": item["fat_g"], "fiber": item["fiber_g"], "estimated": item["is_estimated"], "weight_source": item["weight_source"], "match_method": item["match_method"], "confidence": item["match_confidence"], "key": item_key, "version": CALCULATION_VERSION},
            ).first()
            if row is None:
                row = session.execute(
                    text("SELECT id FROM user_food_log WHERE user_id=:user_id AND idempotency_key=:key"),
                    {"user_id": user_id, "key": item_key},
                ).first()
            if row is None:
                raise RuntimeError("幂等写入未返回饮食明细 ID")
            ids.append(int(row[0]))
        session.commit()
        return ids






def _find_food_rows(session, user_id, key, count):
    from sqlalchemy import bindparam
    keys = [f"{key}:{index}" for index in range(count)]
    rows = session.execute(text("SELECT id, idempotency_key, food_name, weight_g, meal_type FROM user_food_log WHERE user_id=:user_id AND idempotency_key IN :keys").bindparams(bindparam("keys", expanding=True)), {"user_id": user_id, "keys": keys}).all()
    if not rows:
        return None
    found = {row[1]: row for row in rows}
    if set(found) != set(keys):
        raise CommitUnknownError("Incomplete food batch for idempotency key")
    return [found[item] for item in keys]


def _find_food_items(session, user_id, key, count):
    rows = _find_food_rows(session, user_id, key, count)
    return None if rows is None else [int(row[0]) for row in rows]


def _assert_food_key_matches(rows, items, idempotency_key):
    for row, item in zip(rows, items, strict=True):
        if str(row[2]).strip() != str(item.get("food_name") or "").strip():
            raise IdempotencyConflict(
                f"idempotency key {idempotency_key} reused with different food")
        if round(float(row[3] or 0), 4) != round(float(item.get("weight_g") or 0), 4):
            raise IdempotencyConflict(
                f"idempotency key {idempotency_key} reused with different weight")


def log_food_items(user_id: str, items: list[dict[str, Any]], totals: dict[str, float], meal_group_id: str, idempotency_key: str, record_time: datetime | None = None, meal_type: str | None = None) -> list[int]:
    require_ready()
    if not items or not idempotency_key:
        raise ValueError("Food items and idempotency key are required")
    try:
        return _insert_food_items(user_id, items, totals, meal_group_id, idempotency_key, record_time, meal_type)
    except CommitUnknownError as exc:
        def lookup():
            with storage_session() as session:
                return _find_food_items(session, user_id, idempotency_key, len(items))
        return recover_committed(lookup, exc)

def resolve_food_in_session(session, food_name: str) -> FoodMatch | None:
    """C2-2 session primitive: active exact / alias_exact only.

    No global engine, schema seed, embedding or semantic fallback; storage
    errors propagate to the caller instead of being swallowed.
    """
    requested = normalize_food_name(food_name)
    if not requested:
        return None
    row = session.execute(
        text("SELECT f.* FROM food_nutrition f WHERE f.is_active=TRUE AND "
             "(lower(f.food_name)=:name OR lower(replace(f.food_name,' ',''))=:name) LIMIT 1"),
        {"name": requested},
    ).mappings().first()
    if row:
        return _to_match(row, "exact", 1.0)
    alias = session.execute(
        text("SELECT f.*, a.confidence FROM food_alias_index a JOIN food_nutrition f ON f.id=a.food_id "
             "WHERE a.is_active=TRUE AND f.is_active=TRUE AND a.normalized_alias=:name LIMIT 1"),
        {"name": requested},
    ).mappings().first()
    if alias:
        return _to_match(alias, "alias_exact", float(alias.get("confidence", 1.0)))
    return None


def portion_weight_in_session(session, food_id: int, portion_name: str) -> float | None:
    """C2-2 session primitive: real reference table only, no silent seed fallback."""
    if isinstance(food_id, bool) or not isinstance(food_id, int) or food_id <= 0:
        return None
    if not isinstance(portion_name, str) or not portion_name.strip():
        return None
    row = session.execute(
        text("SELECT weight_g FROM food_portion_reference WHERE food_id=:food_id AND portion_name=:portion"),
        {"food_id": food_id, "portion": portion_name.strip()},
    ).first()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def log_food_items_in_session(session, user_id: str, items: list[dict], totals: dict,
                              meal_group_id: str, idempotency_key: str, record_time, meal_type: str) -> dict:
    """C2-2 session primitive: insert every item under ``key:<index>``; flush only.

    ``created`` is strict True only when every row was newly inserted in this
    transaction; a conflict leaves created False without touching existing
    rows. The caller owns the transaction.
    """
    ids: list[int] = []
    created = True
    for index, item in enumerate(items):
        item_key = f"{idempotency_key}:{index}"
        row = session.execute(
            text("""INSERT INTO user_food_log (user_id,meal_group_id,record_time,meal_type,food_id,raw_food_name,"""
                 """food_name,weight_g,total_kcal,total_protein,total_carb,total_fat,total_fiber,is_estimated,"""
                 """weight_source,match_method,match_confidence,idempotency_key,calculation_version) """
                 """VALUES (:user_id,:meal_group_id,:record_time,:meal_type,:food_id,:raw_food_name,:food_name,"""
                 """:weight_g,:kcal,:protein,:carb,:fat,:fiber,:estimated,:weight_source,:match_method,:confidence,"""
                 """:key,:version) ON CONFLICT DO NOTHING RETURNING id"""),
            {"user_id": user_id, "meal_group_id": meal_group_id, "record_time": record_time,
             "meal_type": meal_type, "food_id": item["food_id"], "raw_food_name": item["raw_food_name"],
             "food_name": item["food_name"], "weight_g": item["weight_g"], "kcal": item["energy_kcal"],
             "protein": item["protein_g"], "carb": item["carb_g"], "fat": item["fat_g"],
             "fiber": item["fiber_g"], "estimated": item["is_estimated"],
             "weight_source": item["weight_source"], "match_method": item["match_method"],
             "confidence": item["match_confidence"], "key": item_key, "version": CALCULATION_VERSION},
        ).first()
        if row is None:
            created = False
            continue
        ids.append(int(row[0]))
    return {"created": created and len(ids) == len(items), "log_ids": ids}


def find_food_items_in_session(session, user_id: str, idempotency_key: str) -> list[dict]:
    """C2-2 session primitive: complete existing row set for the exact group
    prefix. Used for foreign-key detection only; matched rows are never a
    bridge proof."""
    rows = session.execute(
        text("SELECT id, idempotency_key, food_name, weight_g, meal_group_id FROM user_food_log "
             "WHERE user_id=:user_id AND (idempotency_key=:key OR idempotency_key LIKE :prefix) ORDER BY id"),
        {"user_id": user_id, "key": idempotency_key, "prefix": idempotency_key + ":%"},
    ).mappings().all()
    return [dict(row) for row in rows]


def delete_catalog_food(food_id: int) -> bool:
    """Hide a library entry without changing saved nutrition snapshots."""
    if not (db_session.DB_READY and db_session.SessionLocal is not None and ensure_food_schema()):
        raise RuntimeError("数据库暂时不可用，无法删除食物")
    try:
        with db_session.SessionLocal() as session:
            row = session.execute(
                text("UPDATE food_nutrition SET is_active=FALSE WHERE id=:food_id RETURNING id"),
                {"food_id": food_id},
            ).first()
            session.commit()
            return row is not None
    except SQLAlchemyError as exc:
        raise RuntimeError("删除食物失败，请稍后重试") from exc
