from __future__ import annotations
from app.core import clock

import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterable

from app.db.food_repository import normalize_food_name
from app.services.agent.workout_extractor import chinese_number_to_value


NUMBER = r"(?:\d+(?:\.\d+)?|半|一半|[零〇一两二三四五六七八九十百千万]+)"
WEIGHT_UNIT = r"(?:克|g|公斤|千克|kg)"
PORTION_UNIT = r"(?:大碗|小碗|汤匙|茶匙|碗|盘|杯|盒|瓶|袋|包|个|枚|根|块|勺|片|份|把)"

PORTION_UNIT_ALIASES = {
    "大碗": "碗",
    "小碗": "碗",
    "汤匙": "勺",
    "茶匙": "勺",
}

MEAL_TYPE_ALIASES = (
    ("早上加餐", "早上加餐"),
    ("上午加餐", "早上加餐"),
    ("下午加餐", "下午加餐"),
    ("下午茶", "下午加餐"),
    ("晚上加餐", "晚上加餐"),
    ("夜宵", "晚上加餐"),
    ("早餐", "早餐"),
    ("早饭", "早餐"),
    ("早上", "早餐"),
    ("早晨", "早餐"),
    ("午餐", "午餐"),
    ("午饭", "午餐"),
    ("中午", "午餐"),
    ("晚餐", "晚餐"),
    ("晚饭", "晚餐"),
    ("晚上", "晚餐"),
    ("今晚", "晚餐"),
    ("加餐", "加餐"),
)


@dataclass(frozen=True)
class FoodInput:
    raw_text: str
    requested_name: str
    quantity: float
    unit: str
    weight_g: float | None
    portion_name: str | None
    meal_type: str | None = None
    record_date: str | None = None
    is_negated: bool = False


def _number(value: str) -> float | None:
    if value in {"半", "一半"}:
        return 0.5
    result = chinese_number_to_value(value)
    return float(result) if result is not None else None


def _strip_prefix(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^(?:(?:记录一下|帮我记录|记录|添加|录入|统计)|(?:今天|今日|今晚|昨天|昨日|昨晚|前天|明天|明日|明晚|翌日|后天)|(?:(?:上|下|本|这)?(?:周|星期)[一二三四五六日天]))+", "", text)

    text = re.sub(r"^(?:记录一下|帮我记录|记录|添加|录入|统计)", "", text)
    text = re.sub(r"^(?:今天|今日|昨天|昨日|前天)", "", text)
    text = re.sub(r"^的", "", text)
    text = re.sub(r"^(?:我)?(?:今天|今日|昨天|昨日|前天)?(?:的)?(?:饮食记录|饮食|吃饭|吃的)[:：]?", "", text)
    text = re.sub(r"^(?:今天|今日|昨天|昨日|前天)(?:我)?(?:吃了|吃过|喝了|摄入了|有)[:：]?", "", text)
    text = re.sub(r"^(?:今天|今日|昨天|昨日|前天)?(?:早上加餐|上午加餐|下午加餐|晚上加餐|下午茶|夜宵|早餐|午餐|晚餐|加餐|早饭|午饭|晚饭)[:：]?", "", text)
    text = re.sub(r"^(?:我)?(?:今天|今日|昨天|昨日|前天)?(?:吃了|吃过|喝了|摄入了|有)[:：]?", "", text)
    text = re.sub(r"^(?:早上加餐|上午加餐|下午加餐|晚上加餐|下午茶|夜宵|早餐|午餐|晚餐|加餐|早饭|午饭|晚饭)(?:我)?(?:今天|今日)?(?:吃了|吃过|喝了|摄入了)[:：]?", "", text)
    return text.strip(" ：:，,、")


def _extract_meal_type(text: str) -> tuple[str | None, str]:
    """Extract an explicit meal period without treating it as a food name."""
    for phrase, meal_type in MEAL_TYPE_ALIASES:
        match = re.search(re.escape(phrase), text)
        if match:
            remaining = text[:match.start()] + text[match.end():]
            return meal_type, remaining
    return None, text


def _split_items(text: str) -> list[str]:
    text = re.sub(r"(?:以及|和)", "，", text)
    return [part for part in re.split(r"[，,、;；+]+", text) if part.strip()]


def _clean_name(name: str) -> str:
    name = re.sub(r"^(?:我今天|今天我|今天|今日)", "", name)
    name = re.sub(r"^(?:一份|一些|适量|少量|一大份|一小份)", "", name)
    name = re.sub(r"^(?:吃了|吃过|喝了|摄入了)", "", name)
    name = re.sub(r"(?:的)?(?:食物|食品)$", "", name)
    return normalize_food_name(name.strip(" ：:，,、。！？!?"))


def _canonical_portion_unit(unit: str) -> str:
    return PORTION_UNIT_ALIASES.get(unit, unit)


def _parse_one(raw: str, meal_type: str | None = None) -> FoodInput | None:
    text = raw.strip()
    # This fallback extracts entities only; the task/model decides write semantics.
    is_negated = False
    text = re.sub(r"^(?:我)?(?:今天|昨天|前天)?(?:没有|没|未|不要|不吃|不喝)(?:吃了|吃过|吃|喝了|喝)?", "", text)
    text = re.sub(r"^(?:只|仅)(?:吃了|吃过|吃|喝了|喝)", "", text)
    direct_before = re.match(rf"^(?P<num>{NUMBER})\s*(?P<unit>{WEIGHT_UNIT})\s*(?P<name>.+)$", text, re.I)
    direct_after = re.match(rf"^(?P<name>.+?)(?P<num>{NUMBER})\s*(?P<unit>{WEIGHT_UNIT})$", text, re.I)
    if direct_before or direct_after:
        match = direct_before or direct_after
        number = _number(match.group("num"))
        name = _clean_name(match.group("name"))
        if number is None or not name:
            return None
        unit = match.group("unit").lower()
        grams = number * 1000 if unit in {"kg", "公斤", "千克"} else number
        return FoodInput(raw.strip(), name, 1.0, unit, grams, None, meal_type, None, is_negated)
    portion_before = re.match(rf"^(?P<num>{NUMBER})\s*(?P<unit>{PORTION_UNIT})\s*(?P<name>.+)$", text)
    portion_after = re.match(rf"^(?P<name>.+?)(?P<num>{NUMBER})\s*(?P<unit>{PORTION_UNIT})$", text)
    if portion_before or portion_after:
        match = portion_before or portion_after
        number = _number(match.group("num"))
        name = _clean_name(match.group("name"))
        if number is None or not name:
            return None
        unit = _canonical_portion_unit(match.group("unit"))
        return FoodInput(raw.strip(), name, number, unit, None, unit, meal_type, None, is_negated)
    return None


def _parse_segment(text: str, meal_type: str | None) -> list[FoodInput]:
    cleaned = _strip_prefix(text)
    parsed: list[FoodInput] = []
    for item in _split_items(cleaned):
        record = _parse_one(item, meal_type)
        if record:
            parsed.append(record)
    return parsed


def parse_food_message(message: str) -> tuple[FoodInput, ...]:
    normalized = unicodedata.normalize("NFKC", message).lower()
    normalized = re.sub(r"\s+", "", normalized)

    # A single message may contain several meal periods.  Keeping the meal
    # marker attached to each segment prevents the first marker from being
    # incorrectly applied to every food item.
    meal_pattern = "|".join(re.escape(phrase) for phrase, _ in MEAL_TYPE_ALIASES)
    meal_matches = list(re.finditer(meal_pattern, normalized))
    if len(meal_matches) > 1:
        parsed: list[FoodInput] = []
        for index, match in enumerate(meal_matches):
            phrase = match.group(0)
            meal_type = next((alias for alias, value in MEAL_TYPE_ALIASES if alias == phrase), None)
            meal_type = next((value for alias, value in MEAL_TYPE_ALIASES if alias == phrase), meal_type)
            end = meal_matches[index + 1].start() if index + 1 < len(meal_matches) else len(normalized)
            parsed.extend(_parse_segment(normalized[match.end():end], meal_type))
        return _attach_record_date(tuple(parsed), normalized)

    meal_type, without_meal_type = _extract_meal_type(normalized)
    return _attach_record_date(tuple(_parse_segment(without_meal_type, meal_type)), normalized)


def _attach_record_date(entries: tuple[FoodInput, ...], message: str) -> tuple[FoodInput, ...]:
    day = clock.parse_date(message)
    if day is None:
        return entries
    return tuple(replace(entry, record_date=day.isoformat()) for entry in entries)


def food_names(entries: Iterable[FoodInput]) -> tuple[str, ...]:
    return tuple(entry.requested_name for entry in entries)


# 显式词典映射：模型/解析器输出的规范名 → 允许出现的原文别名表面。
# 只有列在这里的别名才被实体来源校验接受；不做模糊包含匹配。
FOOD_NAME_ALIASES = {
    "饭": "米饭",
}

_CN_DIGIT_VALUES = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _quantity_evidence(quantity: float, source: str) -> bool:
    """数量必须有原文依据：默认 1 之外，数字或中文数词须出现在原文。"""
    if quantity == 1.0:
        return True
    if quantity == 0.5:
        return "半" in source or "一半" in source
    if not float(quantity).is_integer():
        return str(quantity) in source
    number = int(quantity)
    if str(number) in source:
        return True
    if 2 <= number <= 9:
        return any(char in source for char, value in _CN_DIGIT_VALUES.items() if value == number)
    if number == 10:
        return "十" in source
    tens, units = divmod(number, 10)
    if 10 < number < 100:
        for t_char, t_value in _CN_DIGIT_VALUES.items():
            if t_value != tens:
                continue
            if units == 0 and f"{t_char}十" in source:
                return True
            for u_char, u_value in _CN_DIGIT_VALUES.items():
                if u_value == units and f"{t_char}十{u_char}" in source:
                    return True
    return False


def _weight_evidence(weight_g: float, source: str) -> bool:
    """重量必须有原文依据：数字字面量或“数字+重量单位”换算结果一致。"""
    literal = str(int(weight_g)) if float(weight_g).is_integer() else str(weight_g)
    if literal in source:
        return True
    for match in re.finditer(rf"({NUMBER})\s*({WEIGHT_UNIT})", source, re.IGNORECASE):
        value = _number(match.group(1))
        if value is None:
            continue
        unit = match.group(2).lower()
        grams = value * 1000 if unit in {"kg", "公斤", "千克"} else value
        if abs(grams - weight_g) < 1e-6:
            return True
    return False


def _unit_known(unit: str) -> bool:
    """单位要么是已知份量/重量单位（词典化份量估算），要么原样出现在原文。"""
    if re.fullmatch(WEIGHT_UNIT, unit, re.IGNORECASE) or re.fullmatch(PORTION_UNIT, unit):
        return True
    return _canonical_portion_unit(unit) != unit and re.fullmatch(PORTION_UNIT, _canonical_portion_unit(unit)) is not None


def filter_entries_to_source(
    entries: Iterable[FoodInput], source: str
) -> tuple[tuple[FoodInput, ...], list[str]]:
    """Keep only food entries whose evidence exists inside the authorized span.

    R003: a model (or any non-deterministic extractor) may not invent food
    names, quantities, weights or dates, and may not borrow entities from
    other tasks. Evidence rules:
    - name: verbatim in the span, or an explicit FOOD_NAME_ALIASES surface
      (饭→米饭); normalization stays NFKC/lower/space-stripped.
    - quantity: default 1, otherwise a numeral present in the span.
    - weight: digits present in the span, or a “数字+重量单位” conversion
      that matches exactly (0.5公斤 → 500).
    - unit: present in the span, or a known portion/weight unit (词典化份量
      估算，交由食物计算器打 is_estimated 标记)。
    - meal_type / record_date without span evidence are cleared, never kept
      as a guess.
    Returns (kept entries, dropped names) for observability.
    """
    normalized_source = normalize_food_name(source)
    meal_values = {value for alias, value in MEAL_TYPE_ALIASES if alias in source}
    kept: list[FoodInput] = []
    dropped: list[str] = []
    for entry in entries:
        name = normalize_food_name(entry.requested_name)
        name_ok = bool(name) and name in normalized_source
        if not name_ok:
            name_ok = any(
                normalize_food_name(surface) in normalized_source
                for surface, canonical in FOOD_NAME_ALIASES.items()
                if normalize_food_name(canonical) == name
            )
        if not name_ok:
            dropped.append(entry.requested_name)
            continue
        if entry.weight_g is not None:
            if not _weight_evidence(float(entry.weight_g), source):
                dropped.append(entry.requested_name)
                continue
        else:
            if not _quantity_evidence(float(entry.quantity), source):
                dropped.append(entry.requested_name)
                continue
            unit = (entry.unit or "").strip()
            if unit and unit not in source and not _unit_known(unit):
                dropped.append(entry.requested_name)
                continue
        updates: dict[str, object] = {}
        if entry.meal_type and entry.meal_type not in meal_values:
            updates["meal_type"] = None
        if entry.record_date and entry.record_date not in source:
            updates["record_date"] = None
        kept.append(replace(entry, **updates) if updates else entry)
    return tuple(kept), dropped
