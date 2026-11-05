from __future__ import annotations
from app.core import clock

import math
import re
import unicodedata
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import ValidationError

from app.schemas.workout import WorkoutRecordPayload


NUMBER_TOKEN = r"(?:\d+(?:\.\d+)?|[零〇一两二三四五六七八九十百千万]+)"
UNIT_PATTERN = r"(?:kg|公斤|千克|斤|lb|磅)"
REPS_UNIT_PATTERN = r"(?:次|个|下|回|reps?)"
SETS_UNIT_PATTERN = r"(?:组|套)"

ACTION_ALIASES: tuple[tuple[str, str], ...] = (
    ("哑铃卧推", "哑铃卧推"),
    ("杠铃卧推", "杠铃卧推"),
    ("上斜卧推", "上斜卧推"),
    ("下斜卧推", "下斜卧推"),
    ("哑铃深蹲", "哑铃深蹲"),
    ("杠铃深蹲", "杠铃深蹲"),
    ("保加利亚分腿蹲", "保加利亚分腿蹲"),
    ("单臂哑铃划船", "单臂哑铃划船"),
    ("坐姿划船", "坐姿划船"),
    ("俯身划船", "俯身划船"),
    ("哑铃肩推", "哑铃肩推"),
    ("杠铃肩推", "杠铃肩推"),
    ("哑铃弯举", "哑铃弯举"),
    ("杠铃弯举", "杠铃弯举"),
    ("哑铃飞鸟", "哑铃飞鸟"),
    ("引体向上", "引体向上"),
    ("俯卧撑", "俯卧撑"),
    ("卷腹", "卷腹"),
    ("腿举", "腿举"),
    ("硬拉", "硬拉"),
    ("跑步", "跑步"),
    ("深蹲", "深蹲"),
    ("卧推", "卧推"),
    ("划船", "划船"),
    ("肩推", "肩推"),
    ("弯举", "弯举"),
    ("飞鸟", "飞鸟"),
)

BODYWEIGHT_ACTIONS = {"俯卧撑", "引体向上", "卷腹", "跑步"}
BODYWEIGHT_TOKENS = ("自重", "徒手")
ACTION_PATTERN = re.compile("|".join(re.escape(alias) for alias, _ in ACTION_ALIASES))
CHINESE_DIGITS = {"零": 0, "〇": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


@dataclass(frozen=True)
class WorkoutParseResult:
    status: str
    record: dict[str, Any] | None
    missing_fields: tuple[str, ...]
    inferred_fields: tuple[str, ...]
    normalized_text: str
    number_mentions: tuple[dict[str, Any], ...]
    reason: str = ""
    trace: tuple[dict[str, Any], ...] = ()

    def to_state(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "record": self.record,
            "missing_fields": list(self.missing_fields),
            "inferred_fields": list(self.inferred_fields),
            "normalized_text": self.normalized_text,
            "number_mentions": list(self.number_mentions),
            "reason": self.reason,
            "trace": list(self.trace),
        }


_WORKOUT_FIELD_LABELS = {
    "exercise": "动作名称",
    "weight_kg": "重量（kg；自重动作可填写自重）",
    "sets": "组数",
    "reps": "每组次数",
    "set_details": "逐组明细",
    "date": "训练日期",
    "动作拆分": "动作拆分",
}


def format_missing_workout_fields(fields: Iterable[str]) -> str:
    """Turn internal workout field keys into explicit user-facing requirements."""
    labels: list[str] = []
    for field in fields:
        label = _WORKOUT_FIELD_LABELS.get(str(field), str(field))
        if label and label not in labels:
            labels.append(label)
    return "、".join(labels)


def normalize_workout_text(message: str) -> str:
    text = unicodedata.normalize("NFKC", message).lower()
    text = text.replace("×", "x").replace("＊", "*")
    text = re.sub(r"\s+", "", text)
    return text.replace("公斤", "kg").replace("千克", "kg")


def split_workout_messages(message: str) -> tuple[str, ...]:
    """Split one workout command into action-scoped messages."""
    normalized = normalize_workout_text(message)
    matches = list(ACTION_PATTERN.finditer(normalized))
    if len(matches) <= 1:
        return (message,)
    segments: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        segment = normalized[match.start():end].strip(" ，,、。；;：:！？!?")
        if segment:
            segments.append(segment)
    return tuple(segments) or (message,)


def chinese_number_to_value(token: str) -> int | float | None:
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token) if "." in token else int(token)
    if not token or any(char not in CHINESE_DIGITS and char not in CHINESE_UNITS for char in token):
        return None
    if all(char in CHINESE_DIGITS for char in token):
        return int("".join(str(CHINESE_DIGITS[char]) for char in token))
    total = section = number = 0
    for char in token:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
            continue
        unit = CHINESE_UNITS[char]
        if unit == 10000:
            section += number
            total += section * unit
            section = number = 0
        else:
            section += (number or 1) * unit
            number = 0
    return total + section + number


def _as_positive_int(value: int | float | None) -> int | None:
    if value is None or value <= 0 or int(value) != value:
        return None
    return int(value)


def _number_value(match: re.Match[str]) -> int | float | None:
    return chinese_number_to_value(match.group(1))


def _find_action(text: str) -> str | None:
    for alias, canonical in ACTION_ALIASES:
        if alias in text:
            return canonical
    return None


def _extract_number_mentions(text: str) -> tuple[dict[str, Any], ...]:
    pattern = re.compile(rf"(?P<number>{NUMBER_TOKEN})\s*(?P<unit>{UNIT_PATTERN}|{REPS_UNIT_PATTERN}|{SETS_UNIT_PATTERN})?", re.IGNORECASE)
    mentions: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        value = chinese_number_to_value(match.group("number"))
        if value is None:
            continue
        unit = (match.group("unit") or "").lower()
        if unit in {"kg", "公斤", "千克"}:
            normalized_unit = "kg"
        elif unit == "斤":
            normalized_unit = "jin"
        elif unit in {"lb", "磅"}:
            normalized_unit = "lb"
        elif unit in {"组", "套"}:
            normalized_unit = "sets"
        elif unit in {"次", "个", "下", "回", "rep", "reps"}:
            normalized_unit = "reps"
        else:
            normalized_unit = "bare"
        mentions.append({"raw": match.group(0), "value": value, "unit": normalized_unit})
    return tuple(mentions)


def number_values(text: str) -> set[float]:
    """Normalized numeric values in text, including jin/lb weight conversions.

    Shared source/quantity check for record evidence and read-only reference
    rewrites: a number introduced by a model rewrite must already appear in the
    user's original words or a cited source.
    """
    values: set[float] = set()
    for item in _extract_number_mentions(text):
        value = float(item["value"])
        values.add(value)
        if item["unit"] == "jin":
            values.add(value / 2)
        if item["unit"] == "lb":
            values.add(round(value * .45359237, 6))
    return values


def _extract_weight(text: str) -> float | None:
    match = re.search(rf"({NUMBER_TOKEN})\s*({UNIT_PATTERN})", text, re.IGNORECASE)
    if not match:
        # 自重/徒手 is an explicit weight statement, not a missing number.
        return 0.0 if any(token in text for token in BODYWEIGHT_TOKENS) else None
    value = chinese_number_to_value(match.group(1))
    if value is None:
        return None
    unit = match.group(2).lower()
    if unit == "斤":
        return float(value) * 0.5
    if unit in {"lb", "磅"}:
        return float(value) * 0.45359237
    return float(value)


def _extract_sets(text: str) -> tuple[int | None, bool]:
    match = re.search(rf"({NUMBER_TOKEN})\s*{SETS_UNIT_PATTERN}", text)
    if match:
        return _as_positive_int(_number_value(match)), False
    match = re.search(rf"(?:做了|完成了?|共|一共|练了)({NUMBER_TOKEN})(?=组|套|[,，、]|$)", text)
    if match:
        return _as_positive_int(_number_value(match)), True
    return None, False


def _extract_reps(text: str, sets_end: int | None = None) -> tuple[int | None, bool]:
    explicit = re.search(rf"({NUMBER_TOKEN})\s*{REPS_UNIT_PATTERN}(?!\w)", text, re.IGNORECASE)
    if explicit:
        return _as_positive_int(_number_value(explicit)), False
    weighted = re.search(rf"{NUMBER_TOKEN}\s*{UNIT_PATTERN}\s*[x*]\s*({NUMBER_TOKEN})", text, re.IGNORECASE)
    if weighted:
        return _as_positive_int(chinese_number_to_value(weighted.group(1))), False
    every_set = re.search(rf"每组({NUMBER_TOKEN})(?!\s*(?:kg|斤|lb|磅))", text)
    if every_set:
        return _as_positive_int(_number_value(every_set)), True
    if sets_end is not None:
        after_sets = text[sets_end:]
        adjacent = re.match(r"(?:[,，、/]|x|\*)?" + rf"({NUMBER_TOKEN})", after_sets)
        if adjacent:
            return _as_positive_int(chinese_number_to_value(adjacent.group(1))), True
    return None, False


def _extract_x_pattern(text: str) -> tuple[int | None, int | None]:
    match = re.search(rf"({NUMBER_TOKEN})\s*[x*]\s*({NUMBER_TOKEN})", text, re.IGNORECASE)
    if not match:
        return None, None
    return _as_positive_int(chinese_number_to_value(match.group(1))), _as_positive_int(chinese_number_to_value(match.group(2)))


def preprocess_workout(message: str) -> WorkoutParseResult:
    normalized = normalize_workout_text(message)
    mentions = _extract_number_mentions(normalized)
    exercise = _find_action(normalized)
    from app.services.agent.workout_syntax import source_values

    syntax = source_values(normalized)
    x_sets, x_reps = _extract_x_pattern(normalized)
    sets, sets_inferred = _extract_sets(normalized)
    if sets is None and len(syntax["sets"]) == 1:
        sets = syntax["sets"][0]
        sets_inferred = True
    if sets is None:
        sets = x_sets
    reps, reps_inferred = _extract_reps(normalized)
    if reps is None and len(syntax["reps"]) == 1:
        reps = syntax["reps"][0]
        reps_inferred = True
    if reps is None:
        reps = x_reps
    weight_kg = _extract_weight(normalized)
    if weight_kg is None and len(syntax["weight_kg"]) == 1:
        weight_kg = syntax["weight_kg"][0]

    inferred: list[str] = []
    if sets_inferred or (x_sets is not None and sets == x_sets):
        inferred.append("sets")
    if reps_inferred or (x_reps is not None and reps == x_reps):
        inferred.append("reps")
    if weight_kg is None and exercise in BODYWEIGHT_ACTIONS:
        weight_kg = 0.0
        inferred.append("weight_kg")

    record_date = _extract_record_date(message)
    values = {"exercise": exercise, "sets": sets, "reps": reps, "weight_kg": weight_kg}
    missing = tuple(field for field, value in values.items() if value is None)
    if missing:
        return WorkoutParseResult("incomplete", None, missing, tuple(dict.fromkeys(inferred)), normalized, mentions, "required workout entities could not be resolved deterministically")

    record = {
        "exercise": str(exercise),
        "sets": int(sets),
        "reps": int(reps),
        "weight_kg": float(weight_kg),
        "reps_inferred": "reps" in inferred,
        "note": message,
    }
    if record_date is not None:
        record["date"] = record_date
    try:
        WorkoutRecordPayload.model_validate({key: value for key, value in record.items() if key in WorkoutRecordPayload.model_fields})
    except ValidationError as exc:
        return WorkoutParseResult("invalid", None, (), tuple(dict.fromkeys(inferred)), normalized, mentions, str(exc))
    return WorkoutParseResult("valid", record, (), tuple(dict.fromkeys(inferred)), normalized, mentions)


def _action_clause_text(text: str, exercise: str) -> str:
    """The source clause that literally contains this action's name.

    Clause splitting is punctuation-only. It prevents a bodyweight marker from
    a sibling action's comma-separated clause from supplying this action's
    weight; it never classifies intent or action vocabulary.
    """
    normalized = normalize_workout_text(text)
    name = normalize_workout_text(exercise)
    if not name:
        return normalized
    position = normalized.find(name)
    if position < 0:
        return normalized
    separators = "，,；;。\n"
    starts = [normalized.rfind(sep, 0, position) for sep in separators]
    start = max(starts) + 1 if starts else 0
    ends = [normalized.find(sep, position + len(name)) for sep in separators]
    ends = [end for end in ends if end >= 0]
    end = min(ends) if ends else len(normalized)
    return normalized[start:end]


def _bodyweight_contract_applies(exercise: str, source: str) -> bool:
    """True when the weight is explicitly bodyweight for this action.

    Known bodyweight actions keep their existing zero-weight contract. For any
    other literal action name, 自重/徒手 must occur inside that action's own
    source segment so one action's bodyweight marker cannot fill another's
    missing weight.
    """
    if exercise in BODYWEIGHT_ACTIONS:
        return True
    if not exercise or not source:
        return False
    normalized = normalize_workout_text(source)
    name = normalize_workout_text(exercise)
    if not name or name not in normalized:
        return False
    clause = _action_clause_text(source, exercise)
    return any(token in clause for token in BODYWEIGHT_TOKENS)


def validate_workout_candidate(candidate: dict[str, Any] | None, message: str, preprocessed: dict[str, Any] | None = None, *, reference_date: date | None = None) -> WorkoutParseResult:
    """Validate model output without allowing it to invent unresolved values."""
    candidate = candidate if isinstance(candidate, dict) else {}
    # R011: raw model types are checked before Pydantic normalization. bool is
    # never a number; sets/reps must be whole; weight must be a finite raw
    # int/float (numeric strings are rejected); per-set values must be present.
    if workout_candidate_type_problems(candidate):
        return WorkoutParseResult("invalid", None, (), (), normalize_workout_text(message), (), "invalid model value types")
    base_record = (preprocessed or {}).get("record") or {}
    merged: dict[str, Any] = {}
    for field in ("exercise", "sets", "reps", "weight_kg"):
        merged[field] = candidate.get(field) if candidate.get(field) is not None else base_record.get(field)
    # R011: explicit per-set facts are part of the record and must survive
    # validation instead of being dropped before the save dispatch.
    candidate_details = candidate.get("set_details")
    if isinstance(candidate_details, list) and candidate_details:
        merged["set_details"] = candidate_details
    elif isinstance(base_record.get("set_details"), list) and base_record.get("set_details"):
        merged["set_details"] = base_record["set_details"]
    merged["note"] = message
    source_date = clock.parse_date(message, base_date=reference_date)
    raw_candidate_date = candidate.get("date") or candidate.get("record_date")
    candidate_date: date | None = None
    if isinstance(raw_candidate_date, date):
        candidate_date = raw_candidate_date
    elif isinstance(raw_candidate_date, str) and raw_candidate_date:
        try:
            parsed_candidate_date = date.fromisoformat(raw_candidate_date)
            if parsed_candidate_date.isoformat() == raw_candidate_date:
                candidate_date = parsed_candidate_date
        except ValueError:
            pass

    trace: list[dict[str, Any]] = []
    if source_date is not None:
        merged["date"] = source_date
        if raw_candidate_date is not None and candidate_date != source_date:
            trace.append({
                "code": "model_date_overridden",
                "model_date": str(raw_candidate_date),
                "source_date": source_date.isoformat(),
            })
    elif candidate_date is not None:
        merged["date"] = candidate_date
    else:
        # Neither the user text nor the model supplied a date. Leaving it unset
        # matches every other record entry: the write path stamps the local
        # workout date. Code and model must not invent a different day.
        merged["date"] = base_record.get("date")
    inferred = list((preprocessed or {}).get("inferred_fields") or [])
    inferred.extend(str(item) for item in candidate.get("inferred_fields", []) if item)
    if candidate.get("reps_inferred"):
        inferred.append("reps")
    # An explicit 自重/徒手 (or a known bodyweight action) supplies weight 0.
    # The model is not required to invent a numeric weight the user never gave.
    if merged.get("weight_kg") is None:
        exercise_name = str(merged.get("exercise") or "").strip()
        if _bodyweight_contract_applies(exercise_name, message):
            merged["weight_kg"] = 0.0
            inferred.append("weight_kg")
    missing = tuple(field for field in ("exercise", "sets", "reps", "weight_kg") if merged.get(field) is None)
    normalized = str((preprocessed or {}).get("normalized_text", normalize_workout_text(message)))
    mentions = tuple((preprocessed or {}).get("number_mentions", []))
    if missing:
        return WorkoutParseResult("incomplete", None, tuple(dict.fromkeys(missing)), tuple(dict.fromkeys(inferred)), normalized, mentions, "LLM extraction still has unresolved required entities", tuple(trace))
    try:
        payload = WorkoutRecordPayload.model_validate(merged)
    except ValidationError as exc:
        return WorkoutParseResult("invalid", None, (), tuple(dict.fromkeys(inferred)), normalized, mentions, str(exc), tuple(trace))
    record = payload.model_dump()
    record["reps_inferred"] = "reps" in inferred
    return WorkoutParseResult("valid", record, (), tuple(dict.fromkeys(inferred)), normalized, mentions, trace=tuple(trace))


def _extract_record_date(message: str) -> date | None:
    return clock.parse_date(message)


# ---------------------------------------------------------------------------
# R011: workout entities must trace to the authorized task span
# ---------------------------------------------------------------------------

def _finite_number(value: Any) -> float | None:
    """Raw values only: finite int/float; bool, str, null, NaN and inf rejected."""
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _whole_number(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or number != int(number):
        return None
    return int(number)


def workout_candidate_type_problems(candidate: Any) -> list[str]:
    """Deterministic raw-type check shared by plan persistence and the worker.

    R011: model/cache values are checked before Pydantic, int/float or
    fingerprint conversion. Only raw int/float are numbers (bool, strings such
    as ``"60"``, null, NaN/inf are rejected); sets/reps must be whole; per-set
    entries must be objects with a finite weight and whole reps.
    """
    if not isinstance(candidate, dict):
        return ["record"]
    problems: list[str] = []
    for field in ("sets", "reps"):
        value = candidate.get(field)
        if value is not None and _whole_number(value) is None:
            problems.append(field)
    weight = candidate.get("weight_kg")
    if weight is not None and _finite_number(weight) is None:
        problems.append("weight_kg")
    details = candidate.get("set_details")
    if details is not None:
        if not isinstance(details, list) or not details:
            problems.append("set_details")
        else:
            for item in details:
                if (not isinstance(item, dict)
                        or _finite_number(item.get("weight_kg")) is None
                        or _whole_number(item.get("reps")) is None):
                    problems.append("set_details")
                    break
    return problems


class InvalidWorkoutCandidate(ValueError):
    """A structured workout candidate failed raw-type validation (R011-H).

    Raised before a plan containing the candidate is persisted so the request
    is answered as an explicit input rejection instead of a ledger failure.
    """
    code = "invalid_workout_candidate"

    def __init__(self, problems: list[str]) -> None:
        super().__init__("invalid workout candidate: " + ", ".join(problems))
        self.problems = tuple(problems)


def invalid_structured_workout_candidates(tasks: Any) -> list[str]:
    """Problems of every structured ``workouts`` candidate about to be persisted."""
    problems: list[str] = []
    for task in tasks or ():
        if not isinstance(task, dict):
            continue
        payload = task.get("payload")
        if not isinstance(payload, dict):
            continue
        workouts = payload.get("workouts")
        if not isinstance(workouts, list) or not workouts:
            continue
        for index, candidate in enumerate(workouts):
            fields = workout_candidate_type_problems(candidate)
            if fields:
                problems.append(f"workouts[{index}].{','.join(fields)}")
    return problems


def _action_segments(source: str, exercises: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Split the authorized span into action-scoped evidence segments.

    Literal model-reported names and known aliases supply boundaries. A single
    action owns the whole span (numbers may precede the action
    name); with several actions, each occurrence owns the text up to the next
    occurrence so another action's numbers can never be borrowed. Positions are
    kept so consumed intervals are never taken again by a later record.
    """
    normalized = normalize_workout_text(source)
    matches = [(m.start(), m.end(), m.group(0)) for m in ACTION_PATTERN.finditer(normalized)]
    # Model-reported names must literally occur in the source, but need not
    # belong to a local exercise dictionary. Include all their occurrences so
    # mixed known/custom actions have separate evidence and consumption slots.
    for name in {normalize_workout_text(str(item)) for item in exercises if item}:
        position = normalized.find(name)
        while position >= 0:
            matches.append((position, position + len(name), name))
            position = normalized.find(name, position + len(name))
    selected = []
    for match in sorted(set(matches), key=lambda item: (item[0], -(item[1] - item[0]))):
        if not selected or match[0] >= selected[-1][1]:
            selected.append(match)
    matches = selected
    if not matches:
        return []
    if len(matches) == 1:
        return [{"action": matches[0][2], "text": normalized, "start": 0, "end": len(normalized)}]
    segments: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = 0 if index == 0 else match[0]
        end = matches[index + 1][0] if index + 1 < len(matches) else len(normalized)
        segments.append({"action": match[2], "text": normalized[start:end], "start": start, "end": end})
    return segments


def _segment_matches_action(segment_action: str, exercise: str) -> bool:
    canonical = normalize_workout_text(segment_action)
    name = normalize_workout_text(exercise)
    if not name:
        return False
    return canonical == name or name in canonical


def _interval_consumed(consumed: list[tuple[int, int]], position: int) -> bool:
    return any(start <= position < end for start, end in consumed)


def _take_segment(
    segments: list[dict[str, Any]], consumed: list[tuple[int, int]], exercise: str, normalized_source: str
) -> dict[str, Any] | None:
    """Consume the next unconsumed segment for this action, preferring exact.

    A known/alias occurrence that was already consumed is never re-taken by a
    whole-span find fallback; unknown names need exactly one unconsumed
    occurrence or the record is left without evidence (clarify).
    """
    name = normalize_workout_text(exercise)
    for segment in segments:
        if not _interval_consumed(consumed, segment["start"]) and normalize_workout_text(segment["action"]) == name:
            consumed.append((segment["start"], segment["end"]))
            return segment
    for segment in segments:
        if not _interval_consumed(consumed, segment["start"]) and _segment_matches_action(segment["action"], exercise):
            consumed.append((segment["start"], segment["end"]))
            return segment
    positions: list[int] = []
    if name:
        position = normalized_source.find(name)
        while position != -1:
            if not _interval_consumed(consumed, position):
                positions.append(position)
            position = normalized_source.find(name, position + 1)
    if len(positions) != 1:
        return None
    position = positions[0]
    next_positions = [match.start() for match in ACTION_PATTERN.finditer(normalized_source)
                      if match.start() > position and not _interval_consumed(consumed, match.start())]
    end = min(next_positions) if next_positions else len(normalized_source)
    consumed.append((position, end))
    return {"action": exercise, "text": normalized_source[position:end], "start": position, "end": end}


def _segment_weight_values(segment_text: str) -> list[float]:
    """Only weight-role numbers: a unit conversion decides the kg value."""
    from app.services.agent.workout_syntax import source_values
    return source_values(segment_text)["weight_kg"]


def _segment_sets_values(segment_text: str) -> list[int]:
    from app.services.agent.workout_syntax import source_values
    return source_values(segment_text)["sets"]


def _segment_reps_values(segment_text: str) -> list[int]:
    from app.services.agent.workout_syntax import source_values
    return source_values(segment_text)["reps"]


def _weight_evidenced(weight_kg: float, segment_text: str, source: str, exercise: str) -> bool:
    if weight_kg == 0.0:
        # Known bodyweight actions keep their zero-weight contract. Any other
        # literal action needs its own bodyweight marker in its own clause; a
        # sibling action's 自重/徒手 must never supply this action's weight.
        return (exercise in BODYWEIGHT_ACTIONS
                or any(token in _action_clause_text(segment_text, exercise)
                       for token in BODYWEIGHT_TOKENS))
    return any(abs(candidate - weight_kg) <= 0.01 for candidate in _segment_weight_values(segment_text))


_SET_MARKER = re.compile(rf"第\s*({NUMBER_TOKEN})\s*{SETS_UNIT_PATTERN}")


def _per_set_facts(segment_text: str) -> list[dict[str, Any]] | None:
    """Numbered per-set facts from explicit ``第N组…`` expressions.

    The group number is part of the fact identity: callers order by number
    instead of trusting marker appearance order. ``None`` means the numbering
    itself is invalid (non-integer, <1); ``[]`` means no numbered markers.
    """
    markers = list(_SET_MARKER.finditer(segment_text))
    facts: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        number = chinese_number_to_value(marker.group(1))
        if number is None or float(number) != int(number) or int(number) < 1:
            return None
        end = markers[index + 1].start() if index + 1 < len(markers) else len(segment_text)
        span = segment_text[marker.end():end]
        facts.append({"number": int(number), "weight": _segment_weight_values(span), "reps": _segment_reps_values(span)})
    return facts


def _numbered_fact_value(values: list[float] | list[int], target) -> bool:
    distinct = {round(float(value), 4) for value in values}
    return len(distinct) == 1 and abs(next(iter(distinct)) - target) <= 0.01


def _numbered_details_problems(pairs: list[tuple[float, int]], numbered: list[dict[str, Any]]) -> list[str]:
    numbers = [fact["number"] for fact in numbered]
    if len(set(numbers)) != len(numbers) or sorted(numbers) != list(range(1, len(pairs) + 1)):
        return ["set_details"]
    problems: list[str] = []
    ordered = sorted(numbered, key=lambda fact: fact["number"])
    for index, (item_weight, item_reps) in enumerate(pairs):
        fact = ordered[index]
        if not _numbered_fact_value(fact["weight"], item_weight) or not _numbered_fact_value(fact["reps"], item_reps):
            problems.append(f"set_details[{index}]")
    return problems


def _detail_pairs(details: Any) -> list[tuple[float, int]]:
    """Strict per-set pairs; any non-object/non-numeric entry invalidates all."""
    if not isinstance(details, list) or not details:
        return []
    pairs: list[tuple[float, int]] = []
    for item in details:
        if not isinstance(item, dict):
            return []
        item_weight = _finite_number(item.get("weight_kg"))
        item_reps = _whole_number(item.get("reps"))
        if item_weight is None or item_reps is None:
            return []
        pairs.append((item_weight, item_reps))
    return pairs


def validate_workout_records_against_source(
    records: Iterable[dict[str, Any]], source: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only workout records whose facts exist in the authorized span (R011).

    Mirrors the food R003 rule with action-scoped evidence: a model (or any
    non-deterministic extractor) may not invent or borrow an action, weight,
    set count, rep count or per-set fact from another action's numbers.
    Evidence rules:
    - exercise: the record's action has its own segment in the span; records
      for the same action consume occurrences in source order and a consumed
      occurrence is never re-taken by a whole-span find fallback.
    - weight: a kg/斤/磅 number inside that segment converting to the record
      value, or the bodyweight zero-weight contract. Sets/reps numbers can
      never serve as a weight.
    - sets/reps: explicit labels, two-factor sets × reps, complete three-factor
      weight × reps × sets, or 每组N inside the action's own segment. Explicit
      role labels override positional order; no unrelated number is borrowed.
    - set_details: explicit ``第N组`` facts are matched by group number (the
      array position is the number order, not the marker order); duplicate,
      missing, out-of-range or conflicting numbers are rejected. Uniform
      candidates are only accepted without numbered facts (uniform shorthand),
      never against explicit heterogeneous numbered facts.
    Returns ``(kept, dropped)``; each dropped item carries the exercise and the
    unsourced field keys for an explicit clarification answer.
    """
    normalized_source = normalize_workout_text(str(source or ""))
    raw_source = str(source or "")
    records = list(records)
    segments = _action_segments(raw_source, (record.get("exercise") for record in records if isinstance(record, dict)))
    consumed: list[tuple[int, int]] = []
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            dropped.append({"exercise": "", "fields": ["record"]})
            continue
        exercise = str(record.get("exercise") or "").strip()
        segment = _take_segment(segments, consumed, exercise, normalized_source)
        if segment is None:
            dropped.append({"exercise": exercise or "训练记录", "fields": ["exercise", "weight_kg", "sets", "reps"]})
            continue
        segment_text = str(segment["text"])
        problems: list[str] = []
        sets = _whole_number(record.get("sets"))
        reps = _whole_number(record.get("reps"))
        weight = _finite_number(record.get("weight_kg"))
        if sets is None or sets not in _segment_sets_values(segment_text):
            problems.append("sets")
        if reps is None or reps not in _segment_reps_values(segment_text):
            problems.append("reps")
        if weight is None or not _weight_evidenced(weight, segment_text, raw_source, exercise):
            problems.append("weight_kg")
        details = record.get("set_details")
        numbered = _per_set_facts(segment_text)
        if numbered is None:
            # Invalid group numbering in the authorized span is corrupt source
            # evidence regardless of whether the candidate carried details.
            problems.append("set_details")
        elif numbered:
            numbers = [fact["number"] for fact in numbered]
            if len(set(numbers)) != len(numbers) or sorted(numbers) != list(range(1, len(numbered) + 1)):
                problems.append("set_details")
            elif details is None:
                # R011-B/D: the numbered source is authoritative. A candidate
                # without details can only represent a uniform numbered source;
                # it must never summarize explicit heterogeneous groups.
                if (sets is None or reps is None or weight is None or sets != len(numbered)
                        or not all(_numbered_fact_value(fact["weight"], weight)
                                   and _numbered_fact_value(fact["reps"], reps)
                                   for fact in numbered)):
                    problems.append("set_details")
            else:
                pairs = _detail_pairs(details)
                if not pairs:
                    problems.append("set_details")
                elif sets is not None and len(pairs) != sets:
                    problems.append("set_details")
                else:
                    problems.extend(_numbered_details_problems(pairs, numbered))
        elif details is not None:
            pairs = _detail_pairs(details)
            if not pairs:
                problems.append("set_details")
            elif sets is not None and len(pairs) != sets:
                problems.append("set_details")
            elif all(pair == pairs[0] for pair in pairs):
                if not _weight_evidenced(pairs[0][0], segment_text, raw_source, exercise) or pairs[0][1] not in _segment_reps_values(segment_text):
                    problems.append("set_details")
            else:
                # Heterogeneous per-set facts need explicit numbered evidence;
                # a uniform segment cannot prove them.
                problems.append("set_details")
        if problems:
            dropped.append({"exercise": exercise or "训练记录", "fields": list(dict.fromkeys(problems))})
        else:
            kept.append(record)
    return kept, dropped


def format_unsourced_workout_records(dropped: Iterable[dict[str, Any]]) -> str:
    """User-facing summary of the fields that could not be traced to the span."""
    parts: list[str] = []
    for item in dropped:
        labels: list[str] = []
        for field in item.get("fields", []):
            key = "set_details" if str(field).startswith("set_details") else str(field)
            label = _WORKOUT_FIELD_LABELS.get(key, str(field))
            if label not in labels:
                labels.append(label)
        exercise = str(item.get("exercise") or "训练记录")
        parts.append(f"{exercise}（{'、'.join(labels)}）" if labels else exercise)
    return "；".join(parts)
