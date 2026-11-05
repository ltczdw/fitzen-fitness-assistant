"""C2-3 写入领域适配器（phaseI 实施中）。

``write_catalog`` 返回每请求冻结的九工具目录副本（不改全局 TOOL_CATALOG）；
``WriteAdapters`` 由应用生成 canonical records/batches、冻结 C1 任务/操作并调用
两个已闭单事务桥。AI 有界单次估算只用于 protected pending；其余执行分支显式
返回 blocked_capability/rejected，绝不伪造保存成功或提交回执。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

from app.services.agent.react_models import DomainObservation, ToolStatus

logger = logging.getLogger(__name__)

_NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
_QUANTITY_TOKEN = r"[0-9]+(?:\.[0-9]+)?\s*(?:克|g|G|kg|千克|毫升|ml|ML)"
_LEADING_QUANTITY = re.compile(r"^\s*" + _QUANTITY_TOKEN + r"\s*")
_TRAILING_QUANTITY = re.compile(r"\s*" + _QUANTITY_TOKEN + r"\s*$")
_MACRO_LABELS = {
    "calories": ("热量", "千卡"),
    "protein_g": ("蛋白质", "克"),
    "carbs_g": ("碳水", "克"),
    "fat_g": ("脂肪", "克"),
    "fiber_g": ("纤维", "克"),
}
_MEALS = ("早餐", "午餐", "晚餐", "加餐")


def write_catalog() -> dict:
    """每请求冻结的九工具目录副本；模型可见与执行校验共用同一目录。"""
    from app.services.agent.react_tools import TOOL_CATALOG

    return dict(TOOL_CATALOG)
_DATE_RE = re.compile(r"(20[0-9]{2})-([0-9]{1,2})-([0-9]{1,2})")
_DATE_CN = re.compile(r"([0-9]{1,2})月([0-9]{1,2})日")


def _candidate_binding(context: dict[str, Any], source_text: str,
                       start: int, end: int) -> tuple[dict | None, str | None]:
    """本次真实匹配的授权候选区间必须存在、严格合法且完整包含本次 ref。

    缺失、格式/范围/身份不合法（含 bool 冒充 int）、未完整包含 ref、或无法唯一定位
    时返回错误：调用方必须明确拒绝/澄清，绝不从整句/全消息找近似替代授权。
    """
    from app.services.agent.react_models import is_int

    span = context.get("candidate_span")
    if not isinstance(span, dict):
        return None, "candidate_binding_required"
    span_start, span_end = span.get("start"), span.get("end")
    if not is_int(span_start) or not is_int(span_end):
        return None, "candidate_binding_invalid"
    if not (0 <= span_start <= span_end <= len(source_text)):
        return None, "candidate_binding_invalid"
    if not (span_start <= start and end <= span_end):
        return None, "candidate_binding_mismatch"
    return {"start": span_start, "end": span_end,
            "goal_id": str(span.get("goal_id") or "")}, None


def _meal_type_in(text: str) -> str | None:
    """仅接受规范餐次词本身（闭桥要求 meal_type 出现在真实任务原文中）。"""
    for meal in _MEALS:
        if meal in text:
            return meal
    return None


_QUANTITY_WITH_NAME = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(克|g|G|kg|千克|毫升|ml|ML)\s*(.+?)\s*$")
_NAME_WITH_QUANTITY = re.compile(
    r"^\s*(.+?)\s*([0-9]+(?:\.[0-9]+)?)\s*(克|g|G|kg|千克|毫升|ml|ML)\s*$")


_UNIT_ALIASES = {
    "g": ("克", "g", "G"),
    "克": ("克", "g", "G"),
    "kg": ("千克", "kg", "KG"),
    "千克": ("千克", "kg", "KG"),
    "ml": ("毫升", "ml", "mL", "ML"),
    "毫升": ("毫升", "ml", "mL", "ML"),
}


def _source_unit(unit: Any, span: str) -> str:
    """Return the unit token that actually appears in the item's own source.

    The parser may emit ``g`` while the user wrote ``克``. Only the closed
    equivalence family is considered, and a token must be present verbatim;
    no unit is invented when none appears.
    """
    value = str(unit or "").strip()
    if not value:
        return value
    for token in _UNIT_ALIASES.get(value.lower(), ()):
        if token in span:
            return token
    return value


def _fallback_food_entries(span: str) -> list[dict[str, Any]]:
    """Deterministic food entries when the structured food parser yields none.

    Uses the existing message parser first, then one closed quantity+unit+name
    shape. It never invents a food without a name or an amount.
    """
    from app.services.agent.food_parser import parse_food_message

    try:
        parsed = parse_food_message(span)
    except Exception:
        parsed = ()
    foods = [{"source_text": entry.raw_text, "name": entry.requested_name,
              "quantity": entry.quantity, "unit": entry.unit,
              "weight_g": entry.weight_g, "portion_name": entry.portion_name}
             for entry in parsed
             if entry.requested_name and (entry.weight_g is not None or entry.unit)]
    if foods:
        return foods
    match = _QUANTITY_WITH_NAME.match(span)
    if match:
        quantity, unit, name = match.group(1), match.group(2), match.group(3)
    else:
        match = _NAME_WITH_QUANTITY.match(span)
        if not match:
            return []
        name, quantity, unit = match.group(1), match.group(2), match.group(3)
    name = _normalized_food_name(name)
    if not name:
        return []
    weight = float(quantity) * (1000.0 if unit.lower() in {"kg", "千克"} else 1.0)
    return [{"source_text": span, "name": name, "quantity": float(quantity),
             "unit": unit, "weight_g": weight, "portion_name": None}]


def _normalized_food_name(name: Any) -> str:
    """Strip one leading/trailing quantity+unit from a parsed food name.

    Closed-vocabulary format normalization (``45克燕麦`` → ``燕麦``); it never
    classifies intent or invents a food. Names without a unit are unchanged.
    """
    value = str(name or "").strip()
    cleaned = _TRAILING_QUANTITY.sub("", _LEADING_QUANTITY.sub("", value))
    return cleaned.strip()


def _workout_clarification(exercise: str, fields: list[str]) -> str:
    """One action-scoped clarification, never a generic "need more info".

    Only field labels come from the existing workout syntax formatter; the
    exercise is the model/source's own literal name.
    """
    from app.services.agent.workout_extractor import format_missing_workout_fields

    name = str(exercise or "").strip() or "该训练动作"
    # A full span can be long; keep the action identity, not the whole command.
    if len(name) > 40:
        name = name[:40].rstrip() + "…"
    missing = format_missing_workout_fields(fields)
    if not missing:
        return f"{name}还缺少可确认的训练数值，请补充后再保存。"
    return f"{name}还缺{missing}，请补充后再保存。"


def _resolve_workout_sources(raw_sources: Any, source_text: str,
                             start: int, end: int) -> list[dict[str, Any]]:
    """Resolve model-provided action quotes inside this call's own span.

    The semantic model owns the action range; this helper only maps each exact
    quote to source coordinates and rejects overlaps/foreign text. It never
    derives actions from keywords or a local action dictionary.
    """
    from app.services.agent.workout_extractor import normalize_workout_text

    resolved: list[dict[str, Any]] = []
    cursor = start
    for item in raw_sources if isinstance(raw_sources, list) else []:
        if not isinstance(item, dict):
            continue
        exercise = str(item.get("exercise") or "").strip()
        quote = str(item.get("quote") or "")
        if not exercise or not quote:
            continue
        normalized_exercise = normalize_workout_text(exercise)
        normalized_quote = normalize_workout_text(quote)
        if not normalized_exercise or normalized_exercise not in normalized_quote:
            continue
        position = source_text.find(quote, cursor, end)
        if position < 0:
            position = source_text.find(quote, start, end)
        if position < 0 or position + len(quote) > end:
            continue
        if any(position < existing["end"] and existing["start"] < position + len(quote)
               for existing in resolved):
            continue
        resolved.append({
            "exercise": exercise,
            "quote": quote,
            "start": position,
            "end": position + len(quote),
        })
        cursor = position + len(quote)
    resolved.sort(key=lambda source: source["start"])
    return resolved


def _clause_span(source: str, start: int, end: int) -> tuple[int, int]:
    boundaries = [match.end() for match in re.finditer(r"[。！？!?；;\n]", source[:start])]
    begin = boundaries[-1] if boundaries else 0
    after = re.search(r"[。！？!?；;\n]", source[end:])
    finish = end + after.start() if after else len(source)
    return begin, finish


def _resolve_date(text: str, frozen_date: str) -> tuple[str | None, int, bool]:
    """真实原文日期词 → (日期, 相对冻结日天数, 是否未来)。无日期词用冻结日。"""
    from datetime import date as _date, timedelta

    base = _date.fromisoformat(frozen_date)
    if "前天" in text:
        return (base - timedelta(days=2)).isoformat(), -2, False
    if "昨天" in text:
        return (base - timedelta(days=1)).isoformat(), -1, False
    if "今天" in text or "今晚" in text or "今早" in text:
        return base.isoformat(), 0, False
    if "明天" in text or "后天" in text:
        return None, 0, True
    match = _DATE_RE.search(text)
    if match is not None:
        try:
            value = _date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None, 0, True
        if value > base:
            return None, 0, True
        return value.isoformat(), (value - base).days, False
    match = _DATE_CN.search(text)
    if match is not None:
        try:
            value = _date(base.year, int(match.group(1)), int(match.group(2)))
        except ValueError:
            return None, 0, True
        if value > base:
            return None, 0, True
        return value.isoformat(), (value - base).days, False
    return base.isoformat(), 0, False


def _shift_time(record_time: str, day_delta: int) -> str:
    from datetime import datetime, timedelta

    if not day_delta:
        return record_time
    return (datetime.fromisoformat(record_time) + timedelta(days=day_delta)).isoformat()


def _macro(text: str, key: str) -> float | None:
    label, unit = _MACRO_LABELS[key]
    match = re.search(re.escape(label) + r"\s*" + _NUMBER + r"\s*" + re.escape(unit), text)
    if match is None:
        return None
    return float(match.group(1))


def _custom_span(text: str) -> dict | None:
    """每 100 克自制食物：数量/名称/每100克营养均需原文证据。"""
    if "每100克" not in text and "每100g" not in text:
        return None
    weight_match = re.match(r"\s*" + _NUMBER + r"\s*克", text)
    if weight_match is None:
        return None
    weight = float(weight_match.group(1))
    name_part = text[:text.find("，") if "，" in text else len(text)]
    name = re.sub(r"^\s*" + _NUMBER + r"\s*克", "", name_part).strip()
    if not name:
        return None
    tail = text[text.find("每100克") + 4:] if "每100克" in text else text[text.find("每100g") + 5:]
    per100 = {key: _macro(tail, key) for key in ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")}
    if per100["calories"] is None:
        return None
    return {"weight_g": weight, "name": name,
            "per100": {"kcal": per100["calories"], "protein_g": per100["protein_g"] or 0.0,
                       "carbs_g": per100["carbs_g"] or 0.0, "fat_g": per100["fat_g"] or 0.0,
                       "fiber_g": per100["fiber_g"] or 0.0}}


def _manual_group(text: str, source_ref: dict, db_date: str, *, meal: str) -> dict | None:
    """显式热量/三大营养素的手动营养记录（描述取第一个逗号前的原文）。"""
    if "每100克" in text or "每100g" in text:
        return None
    calories = _macro(text, "calories")
    protein = _macro(text, "protein_g")
    carbs = _macro(text, "carbs_g")
    fat = _macro(text, "fat_g")
    if calories is None and protein is None and carbs is None and fat is None:
        return None
    head = text.split("，")[0].split(",")[0].strip()
    description = head or text.strip()
    return {"source_ref": dict(source_ref), "meal_type": meal, "log_date": db_date,
            "description": description, "calories": calories, "protein_g": protein,
            "carbs_g": carbs, "fat_g": fat}


def _span_of(source: str, text: str, lower: int = 0, upper: int | None = None) -> dict | None:
    start = source.find(text, lower, upper)
    if start < 0:
        return None
    return {"start": start, "end": start + len(text), "exact_text": text}


class WriteAdapters:
    """应用拥有的 canonical 载荷/goal 映射与真实桥派发入口。"""

    def __init__(self, runtime):
        self.runtime = runtime

    # ------------------------------------------------------------- executor

    async def execute(self, name: str, args: Any, context: dict[str, Any]) -> DomainObservation:
        if name == "correct_workout":
            return await self.runtime.correct_workout(args, context)
        if name == "record_workout":
            return await self.record_workout(args, context)
        if name == "record_nutrition":
            return await self.record_nutrition(args, context)
        # 只读/建议/问答在写运行时内复用真实只读 worker（同一 request scope 计量）。
        from app.services.agent import react_adapters, react_readonly_workers

        handler = react_readonly_workers.ANSWER_WORKERS.get(name) or react_adapters.READONLY_HANDLERS.get(name)
        if handler is None:
            return DomainObservation(status=ToolStatus.UNAVAILABLE, error="unknown_tool")
        value = handler(args, context)
        if hasattr(value, "__await__"):
            value = await value
        return DomainObservation.normalize(value)

    # -------------------------------------------------------------- workout

    async def record_workout(self, args: Any, context: dict[str, Any]) -> DomainObservation:
        ref = args.get("source_ref") if isinstance(args, dict) else None
        source_text = str(context.get("source_text") or "")
        if not isinstance(ref, dict):
            return DomainObservation(status=ToolStatus.REJECTED, error="source_ref_missing")
        start, end = ref.get("start"), ref.get("end")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(source_text)):
            return DomainObservation(status=ToolStatus.REJECTED, error="source_ref_out_of_bounds")
        span = source_text[start:end]
        candidate_span, binding_error = _candidate_binding(context, source_text, start, end)
        if binding_error is not None:
            return DomainObservation(status=ToolStatus.NEEDS_CLARIFICATION,
                                     missing_fields=["candidate_binding"], error=binding_error)
        reference_date = date.fromisoformat(self.runtime.db_date())
        from app.services.agent.workout_graph import WorkoutExtractionWorkflow
        records = await self.runtime.agent.graph.input_parser.workout_extractor.extract(
            span, reference_date=reference_date,
            task_source=source_text[candidate_span["start"]:candidate_span["end"]])
        source_span = context.get("workout_source_span")
        if not (isinstance(source_span, (list, tuple)) and len(source_span) == 2
                and isinstance(source_span[0], int) and isinstance(source_span[1], int)):
            source_span = (start, end)
        resolved_sources = _resolve_workout_sources(
            context.get("workout_sources"), source_text,
            int(source_span[0]), int(source_span[1]))
        if len(resolved_sources) >= 2 and records:
            return await self._record_multi_action_workout(
                resolved_sources, records, context, ref)
        if not records or len(records) != 1:
            if records and len(records) > 1:
                return DomainObservation(
                    status=ToolStatus.NEEDS_CLARIFICATION,
                    missing_fields=["workout_action"],
                    error="multiple_workout_actions",
                    summary="这段原文包含多个动作，请分别给出每个动作的保存引用，不能合并保存。",
                )
            return DomainObservation(
                status=ToolStatus.NEEDS_CLARIFICATION,
                missing_fields=["workout_record"], error="workout_semantics_incomplete",
                summary="这段引用里没有可确认的训练动作和数值，请补充动作、重量、组数和次数。",
            )
        raw_exercise = str(records[0].get("exercise") or "").strip() if isinstance(records[0], dict) else ""
        parsed = WorkoutExtractionWorkflow._model_record_state(
            records[0], span, reference_date=reference_date)
        record = parsed.get("record") or {}
        exercise, sets, reps, weight = (record.get(k) for k in ("exercise", "sets", "reps", "weight_kg"))
        if not record:
            missing = list(parsed.get("missing_fields") or ["workout_record"])
            return DomainObservation(
                status=ToolStatus.NEEDS_CLARIFICATION,
                missing_fields=missing,
                error="workout_parse_incomplete",
                summary=_workout_clarification(raw_exercise or span, missing),
                result={"exercise": raw_exercise, "missing_fields": missing},
            )
        # The model interprets the action and numbers, but every saved value must
        # still be evidenced inside this call's own source span. This prevents a
        # sibling action's weight/sets/reps from being borrowed through a broader
        # request or model rewrite.
        from app.services.agent.workout_extractor import (
            format_missing_workout_fields,
            validate_workout_records_against_source,
        )
        kept, dropped = validate_workout_records_against_source([record], span)
        if not kept:
            fields = list((dropped[0] if dropped else {}).get("fields") or ["workout_record"])
            return DomainObservation(
                status=ToolStatus.NEEDS_CLARIFICATION,
                missing_fields=fields,
                error="workout_source_mismatch",
                summary=_workout_clarification(exercise or raw_exercise or span, fields),
                result={"exercise": exercise or raw_exercise,
                        "missing_fields": fields,
                        "missing_text": format_missing_workout_fields(fields)},
            )
        record = kept[0]
        exercise, sets, reps, weight = (record.get(k) for k in ("exercise", "sets", "reps", "weight_kg"))
        db_date = self._workout_date(record, reference_date)
        canonical = {
            "exercise": exercise.strip(), "sets": int(sets), "reps": int(reps),
            "weight_kg": float(weight), "note": "", "difficulty": "未记录",
            "completed": True, "date": db_date, "set_details": None,
        }
        return await self._dispatch_workout(
            span, canonical, {**context, "source_ref": ref, "candidate_span": candidate_span})

    @staticmethod
    def _workout_date(record: dict, reference_date: date) -> str:
        """Keep the source-validated per-record date; only absence defaults locally."""
        value = record.get("date")
        if value is None:
            return reference_date.isoformat()
        return value.isoformat() if isinstance(value, date) else date.fromisoformat(value).isoformat()

    async def _dispatch_workout(self, span: str, canonical: dict, context: dict) -> DomainObservation:
        observation = await self.runtime.dispatch_workout(span, canonical, context)
        if (observation.status == ToolStatus.COMPLETED
                and observation.commit_state == "committed" and observation.receipt_refs
                and (observation.result or {}).get("proof_type") == "workout_db_v1"):
            observation.summary = (
                f"已保存训练记录：{canonical['date']}，{canonical['exercise']}，"
                f"{canonical['weight_kg']:g} kg × {canonical['reps']} 次 × {canonical['sets']} 组。")
        return observation

    async def _record_multi_action_workout(self, units: list[dict[str, Any]],
                                           records: list[dict[str, Any]],
                                           context: dict[str, Any],
                                           ref: dict[str, Any]) -> DomainObservation:
        """Save every model-confirmed action unit, not just the first one.

        Each unit gets its own source span, canonical record, C1 task, operation,
        receipt and idempotency identity. The source units come from the
        validated semantic model; this method only maps records and reads finite
        numeric syntax. A missing unit never blocks an earlier valid unit, but
        the aggregate result is not completed and names the missing action.
        """
        from app.services.agent.workout_graph import WorkoutExtractionWorkflow
        from app.services.agent.workout_extractor import (
            validate_workout_records_against_source,
        )
        from app.services.agent.workout_syntax import source_values

        pending = list(records)
        saved: list[dict[str, Any]] = []
        clarifications: list[dict[str, Any]] = []
        unavailable: list[dict[str, Any]] = []
        unit_results: list[dict[str, Any]] = []
        receipts: list[str] = []
        goal_id = str((context.get("candidate_span") or {}).get("goal_id") or "")
        reference_date = date.fromisoformat(self.runtime.db_date())

        def take_record(exercise: str) -> dict[str, Any] | None:
            from app.services.agent.workout_extractor import normalize_workout_text
            wanted = normalize_workout_text(exercise)
            for index, candidate in enumerate(pending):
                name = normalize_workout_text(str(candidate.get("exercise") or ""))
                if name and wanted and (name == wanted or name in wanted or wanted in name):
                    return pending.pop(index)
            return None

        for unit in units:
            exercise = str(unit["exercise"]).strip()
            quote = str(unit["quote"])
            candidate = take_record(exercise)
            if candidate is None:
                values = source_values(quote)
                def one(field: str):
                    items = values.get(field) or []
                    return items[0] if len(items) == 1 else None
                candidate = {
                    "exercise": exercise,
                    "weight_kg": one("weight_kg"),
                    "sets": one("sets"),
                    "reps": one("reps"),
                    "inferred_fields": [],
                    "record_date": None,
                }
            parsed = WorkoutExtractionWorkflow._model_record_state(
                candidate, quote, reference_date=reference_date)
            record = parsed.get("record") or {}
            if not record:
                missing = list(parsed.get("missing_fields") or ["workout_record"])
                clarifications.append({
                    "exercise": exercise, "missing_fields": missing,
                    "summary": _workout_clarification(exercise, missing),
                })
                unit_results.append({
                    "exercise": exercise, "status": ToolStatus.NEEDS_CLARIFICATION.value,
                    "missing_fields": missing,
                })
                continue
            kept, dropped = validate_workout_records_against_source([record], quote)
            if not kept:
                fields = list((dropped[0] if dropped else {}).get("fields")
                              or ["workout_record"])
                clarifications.append({
                    "exercise": exercise, "missing_fields": fields,
                    "summary": _workout_clarification(
                        record.get("exercise") or exercise, fields),
                })
                unit_results.append({
                    "exercise": exercise, "status": ToolStatus.NEEDS_CLARIFICATION.value,
                    "missing_fields": fields,
                })
                continue
            record = kept[0]
            unit_ref = {
                "start": unit["start"], "end": unit["end"],
                "exact_text": context["source_text"][unit["start"]:unit["end"]],
            }
            unit_span = {"start": unit["start"], "end": unit["end"], "goal_id": goal_id}
            canonical = {
                "exercise": str(record["exercise"]).strip(),
                "sets": int(record["sets"]), "reps": int(record["reps"]),
                "weight_kg": float(record["weight_kg"]), "note": "",
                "difficulty": "未记录", "completed": True,
                "date": self._workout_date(record, reference_date),
                "set_details": None,
            }
            try:
                observation = await self._dispatch_workout(
                    quote, canonical,
                    {**context, "source_ref": unit_ref, "candidate_span": unit_span})
            except Exception as exc:  # noqa: BLE001 - one unit must not erase others
                unavailable.append({"exercise": exercise, "error": type(exc).__name__})
                unit_results.append({
                    "exercise": exercise, "status": ToolStatus.UNAVAILABLE.value,
                    "error": type(exc).__name__,
                })
                continue
            if observation.status == ToolStatus.COMPLETED \
                    and observation.commit_state == "committed" and observation.receipt_refs:
                receipts.extend(str(item) for item in observation.receipt_refs)
                saved.append({
                    "exercise": canonical["exercise"],
                    "receipt_refs": list(observation.receipt_refs),
                    "proof": observation.result,
                    "summary": observation.summary,
                })
                unit_results.append({
                    "exercise": exercise, "status": ToolStatus.COMPLETED.value,
                    "receipt_refs": list(observation.receipt_refs),
                })
            else:
                unavailable.append({
                    "exercise": exercise,
                    "status": observation.status.value,
                    "error": observation.error,
                })
                unit_results.append({
                    "exercise": exercise, "status": ToolStatus.UNAVAILABLE.value,
                    "error": observation.error,
                })

        summary_parts: list[str] = []
        if saved:
            summary_parts.extend(str(item["summary"]) for item in saved if item.get("summary"))
        summary_parts.extend(str(item["summary"]) for item in clarifications)
        if unavailable:
            summary_parts.append("部分动作的保存状态未确认，未作为全部完成处理")
        missing_fields = list(dict.fromkeys(
            field for item in clarifications for field in item["missing_fields"]))
        if unavailable:
            missing_fields = list(dict.fromkeys([*missing_fields, "workout_record"]))
        result = {
            "saved": saved,
            "clarifications": clarifications,
            "unavailable": unavailable,
            "workout_units": unit_results,
            "receipt_refs": list(receipts),
        }
        if saved and not clarifications and not unavailable:
            return DomainObservation(
                status=ToolStatus.COMPLETED, commit_state="committed",
                receipt_refs=list(receipts), result=result,
                summary="；".join(summary_parts) or None,
            )
        if clarifications:
            return DomainObservation(
                status=ToolStatus.NEEDS_CLARIFICATION,
                missing_fields=missing_fields or ["workout_record"],
                error="workout_action_incomplete",
                receipt_refs=list(receipts), result=result,
                summary="；".join(summary_parts),
            )
        return DomainObservation(
            status=ToolStatus.PARTIAL,
            missing_fields=missing_fields or ["workout_record"],
            error="workout_action_unconfirmed",
            receipt_refs=list(receipts), result=result,
            summary="；".join(summary_parts) or "部分动作的保存状态未确认。",
        )

    # ------------------------------------------------------------ nutrition

    async def record_nutrition(self, args: Any, context: dict[str, Any]) -> DomainObservation:
        ref = args.get("source_ref") if isinstance(args, dict) else None
        source_text = str(context.get("source_text") or "")
        if not isinstance(ref, dict):
            return DomainObservation(status=ToolStatus.REJECTED, error="source_ref_missing")
        start, end = ref.get("start"), ref.get("end")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(source_text)):
            return DomainObservation(status=ToolStatus.REJECTED, error="source_ref_out_of_bounds")
        span = source_text[start:end]
        record_time, frozen_date = self.runtime.record_facts()
        context = {**context, "source_ref": ref}
        # 应用元数据（餐次/日期）只能来自本次真实匹配的授权候选区间：图在执行前
        # 显式传入该候选的完整真实区间；缺失/不合法/未包含 ref 一律拒绝（零写），
        # 绝不回退到整句或其它候选的近似事实。
        candidate_span, binding_error = _candidate_binding(context, source_text, start, end)
        if binding_error is not None:
            return DomainObservation(status=ToolStatus.NEEDS_CLARIFICATION,
                                     missing_fields=["candidate_binding"], error=binding_error)
        prefix = source_text[candidate_span["start"]:start]
        meal = _meal_type_in(prefix) or _meal_type_in(span)
        if meal is None:
            return DomainObservation(status=ToolStatus.NEEDS_CLARIFICATION,
                                     missing_fields=["meal_type"], error="meal_type_required")
        date_text = prefix + "\n" + span
        db_date, day_delta, future = _resolve_date(date_text, frozen_date)
        if future or db_date is None:
            return DomainObservation(status=ToolStatus.REJECTED, error="future_date_rejected")
        record_time = _shift_time(record_time, day_delta)
        context = {**context, "candidate_span": candidate_span}
        custom = _custom_span(span)
        if custom is not None:
            batch = self._custom_batch(span, ref, custom, db_date, start=start, end=end,
                                       record_time=record_time, meal=meal)
            return await self.runtime.dispatch_nutrition(span, batch, context, has_ai=False)
        manual = _manual_group(span, ref, db_date, meal=meal)
        if manual is not None:
            batch = {"route": "manual_nutrition_log", "timezone": "Asia/Shanghai", "groups": [manual]}
            return await self.runtime.dispatch_nutrition(span, batch, context, has_ai=False)
        items, unresolved = await self._local_items(source_text, start, end, span, db_date, meal=meal)
        if items and not unresolved:
            batch = {"route": "food_items", "timezone": "Asia/Shanghai",
                     "groups": [{"source_ref": dict(ref), "meal_type": meal,
                                 "record_time": record_time, "foods": items}]}
            return await self.runtime.dispatch_nutrition(span, batch, context, has_ai=False)
        if len(unresolved) != 1:
            # 多个未知食物或无法解析：不猜、不合并；明确澄清，零写。
            return DomainObservation(status=ToolStatus.NEEDS_CLARIFICATION,
                                     missing_fields=["food_resolution"], error="food_resolution_required")
        # AI：有界单次估算仅用于 protected pending（真实付费调用在请求 scope 内）。
        estimate = await self.runtime.estimate_food(unresolved[0])
        if estimate is None:
            if getattr(self.runtime, "policy_stopped", False):
                # 真实预算停止：明确不可用（不是可澄清的估算失败），后续同批调用由
                # 统一 policy-stop 阻断。
                return DomainObservation(status=ToolStatus.UNAVAILABLE, error="model_budget_exhausted")
            return DomainObservation(status=ToolStatus.NEEDS_CLARIFICATION,
                                     missing_fields=["food_estimate"], error="food_estimate_unavailable")
        if self.runtime._policy_stop_check():
            # 估算过程期间出现的真实取消/停止/期限：进入桥 claim 之前零派发。
            return DomainObservation(status=ToolStatus.UNAVAILABLE, error="policy_stop")
        ai_item = unresolved[0]["item"]
        ai_item["source"] = "ai"
        ai_item["nutrition_per_100g"] = estimate
        items.append(ai_item)
        batch = {"route": "food_items", "timezone": "Asia/Shanghai",
                 "groups": [{"source_ref": dict(ref), "meal_type": meal,
                             "record_time": record_time, "foods": items}]}
        return await self.runtime.dispatch_nutrition(span, batch, context, has_ai=True)

    def _custom_batch(self, span: str, ref: dict, custom: dict, db_date: str,
                      *, start: int, end: int, record_time: str, meal: str) -> dict:
        # 完整冻结 source 坐标：item 与 group 同一真实全 source 区间（左闭右开）。
        item_span = {"start": start, "end": end, "exact_text": span}
        entry = {"raw_text": span, "requested_name": custom["name"], "quantity": 1.0,
                 "unit": "克", "weight_g": custom["weight_g"], "portion_name": None,
                 "meal_type": meal, "record_date": db_date, "is_negated": False}
        return {"route": "food_items", "timezone": "Asia/Shanghai",
                "groups": [{"source_ref": dict(ref), "meal_type": meal,
                            "record_time": record_time,
                            "foods": [{"source_ref": item_span, "entry": entry, "source": "custom",
                                       "nutrition_per_100g": dict(custom["per100"])}]}]}

    async def _local_items(self, source_text: str, start: int, end: int, span: str, db_date: str,
                     *, meal: str):
        from app.db.food_repository import resolve_food_in_session
        from app.services.agent.multi_agent_graph import _food_inputs_from_structured_payload
        from app.services.agent.deepseek_input_parser import ParsedFoodResponse
        parser = self.runtime.agent.graph.input_parser
        entities = await parser._call_json(parser._food_prompt, ParsedFoodResponse, span)
        if entities is None or not (isinstance(entities, dict) and entities.get("foods")):
            # A parser/format failure must not turn "45克燕麦" into "需要补充
            # 食物信息": try the existing deterministic parser and one closed
            # quantity+unit+name shape before any AI estimate.
            entities = {"foods": _fallback_food_entries(span)}
        foods = entities.get("foods") or []
        if len(foods) == 1 and not foods[0].get("source_text"):
            foods[0] = {**foods[0], "source_text": span}
        kept = _food_inputs_from_structured_payload({"foods": foods}, span) or ()
        items: list[dict] = []
        unresolved: list[dict] = []
        with self.runtime.factory() as session:
            for entry in kept:
                raw = str(getattr(entry, "raw_text", "") or "")
                item_span = _span_of(source_text, raw, start, end)
                if item_span is None:
                    name = str(getattr(entry, "requested_name", "") or "")
                    item_span = _span_of(source_text, name, start, end)
                if item_span is None:
                    continue
                payload = {
                    "raw_text": raw, "requested_name": str(getattr(entry, "requested_name", "") or ""),
                    "quantity": float(getattr(entry, "quantity", 1.0) or 1.0),
                    "unit": _source_unit(getattr(entry, "unit", ""), span),
                    "weight_g": getattr(entry, "weight_g", None),
                    "portion_name": getattr(entry, "portion_name", None),
                    "meal_type": meal, "record_date": db_date, "is_negated": False,
                }
                if not payload["requested_name"]:
                    continue
                normalized = _normalized_food_name(payload["requested_name"])
                if normalized and normalized != payload["requested_name"]:
                    # The parser may keep the quantity in the name ("45克燕麦").
                    # Try the closed-vocabulary normalized name before any AI
                    # estimate; this is not a semantic substitution.
                    payload["requested_name"] = normalized
                try:
                    resolved = resolve_food_in_session(session, payload["requested_name"]) is not None
                except Exception:
                    resolved = False
                item = {"source_ref": item_span, "entry": payload, "source": "local",
                        "nutrition_per_100g": None}
                if resolved:
                    items.append(item)
                else:
                    unresolved.append({"item": item, "entry": payload, "span": raw})
        return items, unresolved
