"""Validate limited training schedules before rendering their time allocation.

Durations are proposed estimates, never measured user facts. The input limit
comes only from the already source-checked planning information. This module
does not call a provider, repair model output, or invent a missing phase.
"""
from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


_NUM = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+|[零〇一二两三四五六七八九十百]+|半)"
_NUMBER_BOUNDARY = r"(?<![0-9.零〇一二两三四五六七八九十百点半])"
_QUANTITY = re.compile(
    rf"{_NUMBER_BOUNDARY}(?P<low>{_NUM})(?:\s*(?:到|至|[-~～—])\s*(?P<high>{_NUM}))?"
    r"\s*(?P<classifier>个)?(?P<half>半)?\s*"
    r"(?P<unit>小时|钟头|分钟|秒钟|分|秒|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)(?![a-z])(?P<after_half>半)?",
    re.IGNORECASE,
)
_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))


def _number(value: str) -> Decimal:
    if value == "半":
        return Decimal("0.5")
    if value.isascii():
        return Decimal(value)
    if not re.fullmatch(r"[零〇一二两三四五六七八九十百]+", value):
        raise ValueError("invalid quantity")
    if "十" not in value and "百" not in value:
        if len(value) != 1:
            raise ValueError("ambiguous Chinese quantity")
        return Decimal("".join(str(_DIGITS[c]) for c in value))
    tens = r"(?:[一二两三四五六七八九]?十[一二三四五六七八九]?)"
    if not re.fullmatch(rf"{tens}|[一二两三四五六七八九]百(?:[零〇][一二三四五六七八九]|{tens})?", value):
        raise ValueError("ambiguous Chinese quantity")
    total = digit = 0
    last_unit = 1000
    for char in value:
        if char in _DIGITS:
            digit = _DIGITS[char]
        else:
            unit = {"十": 10, "百": 100}[char]
            if unit >= last_unit:
                raise ValueError("invalid quantity order")
            total += (digit or 1) * unit
            digit, last_unit = 0, unit
    return Decimal(total + digit)


def duration_limit_seconds(source_quote: str) -> int | None:
    """Read one explicit numeric duration expression, never a default.

    Only a number/unit expression (optionally composed units or an explicit
    range) is accepted. Leading/trailing words, negation, alternatives and
    unexplained text fail closed; this function never labels intent. An explicit
    range uses its upper bound.
    """
    text = unicodedata.normalize("NFKC", source_quote).strip().lower()
    matches = list(_QUANTITY.finditer(text))
    if not matches:
        return None
    if text[:matches[0].start()].strip() or text[matches[-1].end():].strip():
        return None
    values, units = [], []
    try:
        for match in matches:
            unit = match["unit"]
            scale = 3600 if unit in {"小时", "钟头", "h", "hr", "hrs", "hour", "hours"} else 60 if unit in {"分钟", "分", "m", "min", "mins", "minute", "minutes"} else 1
            low = _number(match["low"])
            value = _number(match["high"]) if match["high"] else low
            if value < low or ((match["half"] or match["after_half"]) and match["high"]) or (match["half"] and match["after_half"]):
                return None
            if (match["half"] and not match["classifier"]) or ((match["half"] or match["after_half"]) and (low == Decimal("0.5") or low != int(low))):
                return None
            if match["half"] or match["after_half"]:
                value += Decimal("0.5")
            values.append(value * scale)
            units.append(scale)
        if len(matches) == 2 and re.fullmatch(r"\s*(?:到|至|[-~～—])\s*", text[matches[0].end():matches[1].start()]):
            if values[1] < values[0]:
                return None
            total = values[1]
        else:
            # A grounded field may include a total and a component/preference.
            # Only adjacent additive units form one duration expression.
            if any(not re.fullmatch(r"\s*(?:又|加|and)?\s*", text[a.end():b.start()]) for a, b in zip(matches, matches[1:])):
                return None
            if units != sorted(set(units), reverse=True):
                return None
            total = sum(values)
        # Exact integer seconds, rounded down to respect sub-second maxima.
        seconds = int(total)
        return seconds if 0 < seconds <= 86400 else None
    except (ValueError, InvalidOperation, KeyError, OverflowError):
        return None


def count_quantity(source_quote: str) -> int | None:
    """Parse only a bounded numeric field; its business meaning is typed upstream."""
    text = unicodedata.normalize("NFKC", source_quote).strip()
    try:
        if not re.fullmatch(_NUM, text):
            return None
        value = _number(text)
        return int(value) if value == int(value) and 1 <= value <= 21 else None
    except (ValueError, InvalidOperation, OverflowError):
        return None


def weekly_sessions(source_quote: str) -> int | None:
    text = unicodedata.normalize("NFKC", source_quote).strip()
    try:
        if re.fullmatch(_NUM, text):
            numbers = [_number(text)]
        else:
            numbers = [_number(m[1]) for m in re.finditer(rf"{_NUMBER_BOUNDARY}({_NUM})\s*(?:天|次|回|练)", text)]
        if len(numbers) != 1 or numbers[0] != int(numbers[0]):
            days = weekday_list(text)
            return len(days) if days else None
        result = int(numbers[0])
        return result if 1 <= result <= 21 else None
    except (ValueError, InvalidOperation, OverflowError):
        return None


_WEEKDAY_NAMES = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7}
_WEEKDAY_SEPARATORS = "、,，和与及跟"


def weekday_list(source_quote: str) -> list[int] | None:
    """Parse a finite Chinese weekday list such as 周二和周六.

    This is a format parser over a closed vocabulary, not an intent classifier.
    It returns the unique weekday numbers in source order, or ``None`` when the
    text contains anything outside weekdays/separators or repeats a weekday.
    """
    text = unicodedata.normalize("NFKC", source_quote).strip()
    text = text.replace("星期", "周").replace("礼拜", "周")
    if not text:
        return None
    parts = re.split(rf"[{re.escape(_WEEKDAY_SEPARATORS)}]", text)
    values: list[int] = []
    for part in parts:
        token = part.strip()
        match = re.fullmatch(r"周([一二三四五六日天])", token)
        if not match:
            return None
        value = _WEEKDAY_NAMES[match.group(1)]
        if value in values:
            return None
        values.append(value)
    return values if 1 <= len(values) <= 7 else None


def expected_session_count(scope: str, available_days_count: int | None,
                           cycle_sessions: int | None = None) -> int | None:
    """One plan target may span several sessions when a grounded weekday count exists."""
    if scope == "cycle":
        return cycle_sessions
    if isinstance(available_days_count, int) and not isinstance(available_days_count, bool) \
            and available_days_count > 0:
        return min(available_days_count, 21)
    return None if scope == "weekly" else 1


def minimum_session_seconds(phase_limits: list[dict] | None) -> int:
    """Lower bound on a session's total seconds given source-bound phase budgets.

    Every phase must be a positive integer. An unconstrained phase or an
    ``at_most`` bound reserves 1 second; ``exact``/``at_least`` reserve their
    bound. Duplicate same-phase bounds use the strongest lower bound so they
    never double-count. Callers compare this against the total limit to reject
    impossible allocations before retrieval or generation.
    """
    lower: dict[str, int] = {}
    for limit in phase_limits or []:
        if not isinstance(limit, dict):
            continue
        phase = limit.get("phase")
        relation = limit.get("relation")
        seconds = limit.get("seconds")
        if phase not in {"warmup", "main", "breaks", "cooldown"} or relation not in {"exact", "at_most", "at_least"}:
            continue
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
            continue
        if relation in {"exact", "at_least"}:
            lower[phase] = max(lower.get(phase, 0), seconds)
    return sum(lower.get(phase, 1) for phase in ("warmup", "main", "breaks", "cooldown"))


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Phase(_Output):
    seconds: StrictInt = Field(gt=0, le=86400, description="该阶段累计用时，正整数秒；与其余三阶段相加不得超过单次上限。")
    activity: str = Field(min_length=1, max_length=1400, description="写动作、组数、次数或活动方法；自然语言可用‘若干秒’等不定量描述，但任何明确数字时长都必须落在本阶段seconds内，用时统一由seconds展示。")


class Rest(_Output):
    seconds: StrictInt = Field(gt=0, le=86400, description="组间休息和动作转换的累计用时，只填正整数秒，不提供activity或额外说明。")


class Exercise(_Output):
    name: str = Field(min_length=1, max_length=150, description="动作名称，不含计时或额外指令。")
    sets: StrictInt = Field(gt=0, le=100)
    reps: StrictInt | None = Field(default=None, gt=0, le=1000, description="每组次数；与hold_seconds恰选一个。")
    hold_seconds: StrictInt | None = Field(default=None, gt=0, le=86400, description="计时动作每组保持的秒数；与reps恰选一个。")


class Main(_Output):
    seconds: StrictInt = Field(gt=0, le=86400)
    exercises: list[Exercise] = Field(min_length=1, max_length=12)


class Session(_Output):
    label: str = Field(min_length=1, max_length=100, description="训练日或训练序号；自然语言可含不定量时间描述，但明确数字时长不得超过本次总上限。")
    warmup: Phase
    main: Main
    breaks: Rest
    cooldown: Phase
    time_consistency: Literal["consistent"] = Field(
        description="结构化自检：确认activity/name/label中的自然语言时间描述没有引入额外安排，所有明确时间均落在对应seconds/hold_seconds内；不一致时不要返回该结果，模型必须显式返回该字段。")

    @property
    def total_seconds(self) -> int:
        return sum(getattr(self, name).seconds for name in ("warmup", "main", "breaks", "cooldown"))


class TimedTrainingPlan(_Output):
    sessions: list[Session] = Field(min_length=1, max_length=21)


def output_contract() -> str:
    return (
        "\n本次时间安排必须通过指定的具名结果函数返回单个JSON参数对象，不输出Markdown或工具参数之外的正文。"
        "若planning_information未给time_limit_seconds或session_limits_seconds，表示用户没有指定单次上限，直接给出合理时间估算，不追问上限，不编造用户上限。"
        "按planning_information中的time_limit_seconds及expected_sessions安排，不能修改这两个约束。"
        "若planning_information提供了session_limits_seconds，按数组顺序让每个session的总秒数不超过对应上限；不得把所有训练日改成同一个上限。"
        "若planning_information提供了phase_limits（热身/主体/间歇/收尾的固定或上下限秒数），每个阶段的seconds必须严格满足其关系：exact恰好等于、at_most不超过、at_least不低于；不得改变固定值或编造秒数。"
        "每次训练都必须显式填写warmup准备热身、main主体动作、breaks组间休息与动作转换、cooldown收尾放松四项。"
        "每项seconds是正整数秒；有对应上限时四项全部相加不得超限，不要把收尾留作未分配余量或另加在上限外。"
        "breaks是穿插在主体中的累计休息转换时间，不是主体结束后再休息。"
        "main.exercises逐项写动作name、sets以及reps或hold_seconds，两者恰选一个；计时动作只能用hold_seconds，不要在名称中写用时。"
        "所有计时动作sets乘hold_seconds之和不得超过main.seconds。warmup/cooldown.activity只写阶段活动。"
        "breaks只填写累计seconds，不填activity或逐组休息秒数；休息转换说明由应用统一展示。"
        "activity、name和label可以用自然语言（包括‘若干秒’等不定量描述），但任何明确数字时长都必须落在对应阶段seconds或hold_seconds内，不能借描述另加时间或形成第二套时间表；"
        "每个session必须返回time_consistency='consistent'，确认描述与结构化秒数一致；不一致时先自行修正，不要返回矛盾结果。"
        "顶层只有sessions，不要introduction、notes或其他额外字段；应用会统一展示上限、总计、间歇说明、来源与估算标记。"
        "所有历史事实与知识来源规则仍适用。JSON schema："
        + json.dumps(TimedTrainingPlan.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    )


def _time(seconds: int) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}分{rest}秒" if minutes and rest else f"{minutes}分钟" if minutes else f"{rest}秒"


def parse_timed_plan(raw: str | TimedTrainingPlan, *, limit_seconds: int | None,
                     expected_sessions: int, phase_limits: list[dict] | None = None,
                     expected_weekdays: list[str] | None = None,
                     session_limits: list[int | None] | None = None) -> TimedTrainingPlan:
    """Validate and return the structured plan; rendering is a separate step."""
    def unique_object(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_plan_field")
            result[key] = value
        return result
    if isinstance(raw, TimedTrainingPlan):
        plan = raw.model_copy(deep=True)
    else:
        plan = TimedTrainingPlan.model_validate(json.loads(raw, object_pairs_hook=unique_object))
    if len(plan.sessions) != expected_sessions:
        raise ValueError("session_count_mismatch")
    if len({session.label for session in plan.sessions}) != len(plan.sessions):
        raise ValueError("duplicate_session_label")
    limits = (list(session_limits) if session_limits is not None
              else [limit_seconds if limit_seconds is not None else 86400] * len(plan.sessions))
    if len(limits) != len(plan.sessions):
        raise ValueError("session_limit_count_mismatch")
    if any(value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0)
           for value in limits):
        raise ValueError("invalid_session_limit")
    # An existing preview keeps each unchanged day at its own confirmed
    # duration; only the edited day takes the new limit.
    if any(session.total_seconds > (limit if limit is not None else 86400)
           for session, limit in zip(plan.sessions, limits)):
        raise ValueError("session_time_exceeded")
    for session in plan.sessions:
        for exercise in session.main.exercises:
            if (exercise.reps is None) == (exercise.hold_seconds is None):
                raise ValueError("exercise_measure_must_be_unique")
        if sum(exercise.sets * (exercise.hold_seconds or 0) for exercise in session.main.exercises) > session.main.seconds:
            raise ValueError("timed_exercises_exceed_main_phase")
    # Explicit source-bound phase budgets are checked per session, never edited.
    for limit in phase_limits or []:
        phase = limit.get("phase")
        relation = limit.get("relation")
        seconds = limit.get("seconds")
        if phase not in {"warmup", "main", "breaks", "cooldown"} or relation not in {"exact", "at_most", "at_least"}:
            raise ValueError("invalid_phase_limit")
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
            raise ValueError("invalid_phase_limit")
        for session in plan.sessions:
            actual = getattr(session, phase).seconds
            if relation == "exact" and actual != seconds:
                raise ValueError("phase_limit_violation")
            if relation == "at_most" and actual > seconds:
                raise ValueError("phase_limit_violation")
            if relation == "at_least" and actual < seconds:
                raise ValueError("phase_limit_violation")
    if expected_weekdays:
        if len(plan.sessions) != len(expected_weekdays):
            raise ValueError("session_weekday_mismatch")
        for session, weekday in zip(plan.sessions, expected_weekdays):
            if weekday not in session.label:
                raise ValueError("session_weekday_mismatch")
    return plan


def render_timed_plan(raw: str, *, limit_seconds: int | None, expected_sessions: int,
                      phase_limits: list[dict] | None = None,
                      expected_weekdays: list[str] | None = None,
                      session_limits: list[int | None] | None = None) -> str:
    """Reject malformed/missing/over-limit phases and render one source of time."""
    plan = parse_timed_plan(raw, limit_seconds=limit_seconds,
                            expected_sessions=expected_sessions,
                            phase_limits=phase_limits,
                            expected_weekdays=expected_weekdays,
                            session_limits=session_limits)
    limits = (list(session_limits) if session_limits is not None
              else [limit_seconds] * len(plan.sessions))
    if session_limits is None and limit_seconds is None:
        intro = "以下为时间安排示例；热身、主体、间歇转换与收尾全部计入，未指定单次时长上限。"
    elif session_limits is None:
        intro = f"以下为示例时间分配，每次上限{_time(limit_seconds)}；热身、主体、间歇转换与收尾全部计入。"
    else:
        intro = "以下为示例时间分配，各次上限按对应训练日标明；热身、主体、间歇转换与收尾全部计入。"
    lines = ["已根据确认的条件生成训练安排示例。", "", intro]
    labels = (("warmup", "准备／热身"), ("main", "主体动作"), ("breaks", "组间休息与转换（穿插进行）"), ("cooldown", "收尾／放松"))
    clean = lambda value: value.replace("|", "／").replace("\n", "；").replace("\r", "")
    for session, session_limit in zip(plan.sessions, limits):
        lines += ["", clean(session.label), "", "| 阶段 | 用时 | 安排 |", "|---|---|---|"]
        for name, label in labels:
            phase = getattr(session, name)
            if name == "main":
                activity = "；".join(f"{exercise.name}：{exercise.sets}组，每组" + (f"{exercise.reps}次" if exercise.reps is not None else _time(exercise.hold_seconds)) for exercise in phase.exercises)
            elif name == "breaks":
                activity = "穿插在主体中的累计休息和动作转换，不在总量外另加。"
            else:
                activity = phase.activity
            lines.append(f"| {label} | {_time(phase.seconds)} | {clean(activity)} |")
        lines.append(f"合计：{_time(session.total_seconds)}。" if session_limit is None else
                     f"合计：{_time(session.total_seconds)}，不超过{_time(session_limit)}。")
    if expected_sessions > 1:
        lines += ["", f"以上{expected_sessions}次训练累计{_time(sum(s.total_seconds for s in plan.sessions))}。"]
    lines += ["", "用时为建议估算，并非实测；不在上述总量外追加热身、休息或收尾。"]
    return "\n".join(lines)
