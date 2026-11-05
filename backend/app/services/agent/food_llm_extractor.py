"""LLM-based food entity extraction, replacing fragile regex parsing."""
from __future__ import annotations
import json
import logging
from app.services.agent.food_parser import FoodInput, normalize_food_name

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM = """你是饮食记录实体抽取器，不是聊天助手。
任务：从用户消息中提取所有食物条目，输出严格 JSON。
只输出 JSON，不要解释，不要 markdown 代码块。

输出格式：
{"foods":[{"name":"食物名称","quantity":数量数字,"unit":"单位","weight_g":重量克数或null,"meal_type":"早餐/午餐/晚餐/加餐/其他或null"}]}

规则：
- "两个鸡蛋" → {"name":"鸡蛋","quantity":2,"unit":"个","weight_g":null}
- "鸡胸肉150g" → {"name":"鸡胸肉","quantity":1,"unit":"g","weight_g":150}
- "一碗米饭" → {"name":"米饭","quantity":1,"unit":"碗","weight_g":null}
- "一杯牛奶" → {"name":"牛奶","quantity":1,"unit":"杯","weight_g":null}
- 没有提到重量时 weight_g 填 null
- 没有明确数量时 quantity 填 1
- meal_type 从"早餐/午餐/晚餐/早上/晚上/加餐"等推断，没有就填 null
- 用户否定的食物（"不吃鸡蛋"）不要输出
- 用户没有提到食物时输出 {"foods":[]}"""

async def extract_foods_with_llm(llm, message: str) -> tuple[FoodInput, ...]:
    """Use LLM to extract food entities. Returns empty tuple on failure."""
    if not llm or not llm.configured():
        return ()
    try:
        raw = await llm.complete([
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": message},
        ], temperature=0.1)
    except Exception:
        logger.warning("llm_food_extract_failed", exc_info=True)
        return ()
    if not raw:
        return ()
    # strip markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # try to find JSON in text
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end+1])
            except json.JSONDecodeError:
                return ()
        else:
            return ()
    foods = data.get("foods") or []
    entries: list[FoodInput] = []
    for item in foods:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        name = normalize_food_name(name)
        try:
            qty = float(item.get("quantity") or 1)
        except (TypeError, ValueError):
            qty = 1.0
        unit = str(item.get("unit") or "").strip()
        try:
            wg = item.get("weight_g")
            weight_g = float(wg) if wg is not None else None
        except (TypeError, ValueError):
            weight_g = None
        meal_type = str(item.get("meal_type") or "").strip() or None
        entries.append(FoodInput(
            raw_text=f"{qty:g}{unit}{name}",
            requested_name=name,
            quantity=qty,
            unit=unit or "个",
            weight_g=weight_g,
            portion_name=None if weight_g is not None else unit,
            meal_type=meal_type,
        ))
    return tuple(entries)
