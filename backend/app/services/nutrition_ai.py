from __future__ import annotations

import json
import re
from typing import Any

from app.services.llm.client import LLMClient


def _json_object(raw: str) -> dict[str, Any] | None:
    fence = chr(96) * 3
    text = raw.strip().replace(fence + "json", "").replace(fence, "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


async def estimate_food_nutrition(query: str, llm: LLMClient | None = None) -> dict[str, Any] | None:
    client = llm or LLMClient()
    system = """你是食物营养数据估算器。根据用户描述，估算该食物每100克可食部分的营养。
区分生熟状态、烹饪方式和品牌信息；信息不足时采用常见版本并在 notes 中说明。
只输出一个合法 JSON 对象，不要 Markdown，不要额外解释：
{"food_name":"名称","preparation_state":"生/熟/即食/液态","kcal":0,"protein_g":0,"carbs_g":0,"fat_g":0,"fiber_g":0,"notes":"估算依据","confidence":"low|medium|high"}
所有营养字段必须是每100克的非负数字。"""
    raw = await client.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": query}],
        temperature=0.0,
    )
    if not raw:
        return None
    payload = _json_object(raw)
    if payload is None:
        return None
    try:
        values = {
            "kcal": float(payload["kcal"]),
            "protein_g": float(payload["protein_g"]),
            "carbs_g": float(payload["carbs_g"]),
            "fat_g": float(payload["fat_g"]),
            "fiber_g": float(payload.get("fiber_g", 0)),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if not str(payload.get("food_name") or "").strip():
        return None
    if not (0 <= values["kcal"] <= 1200):
        return None
    if any(not 0 <= values[key] <= 100 for key in ("protein_g", "carbs_g", "fat_g", "fiber_g")):
        return None
    if values["protein_g"] + values["carbs_g"] + values["fat_g"] + values["fiber_g"] > 110:
        return None
    confidence = str(payload.get("confidence") or "medium").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    return {
        "food_name": str(payload["food_name"]).strip()[:200],
        "preparation_state": str(payload.get("preparation_state") or "未说明").strip()[:50],
        "per_100g": values,
        "notes": str(payload.get("notes") or "AI 根据描述估算，请在保存前核对。").strip()[:500],
        "confidence": confidence,
        "source": "ai_estimate",
    }
