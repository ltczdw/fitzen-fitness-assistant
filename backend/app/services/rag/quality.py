from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# MEWHELP/core/confidence.py four-signal structure and weights.
# Domain terms adapted to public fitness evidence; threshold remains unchanged.
KEY_CLAUSES=('每周','每天','每组','分钟','蛋白质','禁忌','停止','就医','逐步增加','循序渐进')

# 纯词法降级模式（vector 不可用）的置信可达上限：0.30*lexical_strength + 0.25*coverage。
# 安全任务阈值上浮若超过该值即不可达（G004），上浮保持基线阈值。
_DEGRADED_ACHIEVABLE_MAX = 0.55

_STOP_TERMS = {
    "怎么", "如何", "怎样", "请问", "帮我", "告诉", "可以", "建议", "推荐",
    "一下", "什么", "有没有", "是否", "请教", "想问", "我要",
}


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _query_terms(query: str) -> set[str]:
    # G003: 覆盖率与候选过滤、BM25 索引共用同一查询词项规则（重叠二元词），
    # 再套用本模块的停用词与最小长度约束。
    from app.services.rag.bm25 import query_terms
    terms = set(query_terms(query))
    return {term for term in terms if term not in _STOP_TERMS and len(term) > 1}


def _field(hit: Any, name: str, default: Any = None) -> Any:
    if isinstance(hit, dict):
        return hit.get(name, default)
    return getattr(hit, name, default)


def _metadata(hit: Any) -> dict[str, Any]:
    value = _field(hit, "metadata", {})
    return value if isinstance(value, dict) else {}


def _query_coverage(query: str, hits: list[Any]) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.5 if hits else 0.0
    evidence = " ".join(
        f"{_field(hit, 'title', '')} {_field(hit, 'content', '')}" for hit in hits[:3]
    ).lower()
    return sum(term in evidence for term in terms) / len(terms)


@dataclass(frozen=True)
class RetrievalAssessment:
    confidence: float
    threshold: float
    mode: str
    answerable: bool
    reason: str
    signals: dict[str, float | int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_rag_evidence(query: str, hits: list[Any], *, task_type: str = "training_qa") -> RetrievalAssessment:
    """Estimate whether retrieved evidence can support a generated answer.

    Model reranker scores and degraded lexical/vector scores use separate formulas;
    the RRF rank score is never interpreted as a probability or relevance score.
    """
    base_threshold = float(settings.rag_answer_confidence_threshold)
    if not hits:
        threshold = base_threshold
        if task_type in {"safety_triage", "symptom_triage"}:
            threshold = min(0.95, threshold + 0.10)
        result = RetrievalAssessment(0.0, threshold, "none", False, "no_hits", {
            "top1_score": 0.0, "valid_count": 0, "margin": 0.0,
            "query_coverage": 0.0,
        })
        _record_assessment(result, 0)
        return result

    coverage = _query_coverage(query, hits)
    model_scores = [
        _clip01(_number(_metadata(hit).get("rerank_score")))
        for hit in hits
        if _metadata(hit).get("rerank_status") == "model"
    ]
    mode = "model_rerank" if model_scores else "degraded_rules"

    threshold = base_threshold
    if task_type in {"safety_triage", "symptom_triage"}:
        bumped = min(0.95, base_threshold + 0.10)
        # G004：上浮必须落在当前模式的可达分数范围内。纯词法降级的置信上限
        # 为 0.30*lexical_strength + 0.25*coverage = 0.55，基线 0.48 上浮到
        # 0.58 后安全问题必然全部拒答（不可达组合）；此时保持基线阈值，让
        # 安全任务与普通任务在同一可达标尺上判定。模型重排可达 1.0，照常上浮。
        if mode == "model_rerank" or bumped <= _DEGRADED_ACHIEVABLE_MAX:
            threshold = bumped
    if model_scores:
        scores = model_scores
        top1 = scores[0]
        margin = max(0.0, top1 - scores[1]) if len(scores) > 1 else top1
        valid_count = sum(score >= 0.30 for score in scores)
        key_hit=any(any(term in f"{_field(hit,'title','')} {_field(hit,'content','')}" for term in KEY_CLAUSES) for hit in hits[:3])
        confidence = (
            0.50 * top1
            + 0.20 * min(valid_count, 3) / 3
            + 0.20 * _clip01(margin)
            + 0.10 * float(key_hit)
        )
        mode = "model_rerank"
        signals = {
            "top1_score": round(top1, 4),
            "valid_count": valid_count,
            "margin": round(margin, 4),
            "query_coverage": round(coverage, 4),
            "key_clause_hit": int(key_hit),
        }
    else:
        vector_top = max(
            (_clip01(_number(_metadata(hit).get("vector_score"))) for hit in hits),
            default=0.0,
        )
        lexical_top = max(
            (_number(_metadata(hit).get("lexical_score")) for hit in hits),
            default=0.0,
        )
        lexical_strength = 1.0 - math.exp(-max(0.0, lexical_top))
        confidence = 0.45 * vector_top + 0.30 * lexical_strength + 0.25 * _clip01(coverage)
        mode = "degraded_rules"
        signals = {
            "vector_top": round(vector_top, 4),
            "lexical_strength": round(lexical_strength, 4),
            "query_coverage": round(coverage, 4),
        }

    confidence = round(_clip01(confidence), 4)
    answerable = confidence >= threshold
    reason = "evidence_sufficient" if answerable else "low_evidence_confidence"
    result = RetrievalAssessment(confidence, threshold, mode, answerable, reason, signals)
    _record_assessment(result, len(hits))
    return result


def _record_assessment(result: RetrievalAssessment, hit_count: int) -> None:
    event = {
        "confidence": result.confidence,
        "threshold": result.threshold,
        "mode": result.mode,
        "answerable": result.answerable,
        "reason": result.reason,
        "hit_count": hit_count,
        "signals": result.signals,
    }
    from app.services.telemetry import mark_degraded
    if not result.answerable:
        mark_degraded("insufficient_evidence", failure=True)
    logger.info("rag_quality_gate", extra={"rag_quality": event})
    try:
        from app.services.rag.retrieval import retrieval_traces

        traces = retrieval_traces.get()
        if traces is not None:
            traces.append({"corpus": "knowledge", "quality_gate": event})
    except Exception:
        logger.debug("rag_quality_trace_unavailable", exc_info=True)


def insufficient_evidence_reply(task_type: str, assessment: RetrievalAssessment | None = None,
                                *, filtered: bool = False) -> str:
    # A retrieval miss does not prove that the corpus lacks a topic. Report
    # only what this request's actual quality/filter checks established.
    reason = assessment.reason if assessment is not None else None
    detail = ("缺少：适用于当前问题及条件的知识依据；检索结果经适用性检查后仍不足。"
              if filtered else
              "缺少：与当前问题匹配的知识依据；本次检索未找到资料。"
              if reason == "no_hits" else
              "缺少：可靠性达到回答要求的知识依据；已检索到资料，但质量检查未通过。"
              if reason == "low_evidence_confidence" else
              "缺少：支持当前回答的可靠知识依据；具体不足类型尚未确认。")
    if task_type in {"safety_triage", "symptom_triage"}:
        return detail + "暂不能据此判断。先停止引发不适的训练；如果症状明显或持续加重，请联系专业医疗人员。"
    if task_type in {"nutrition_advice", "nutrition"}:
        return detail + "这些依据不足以支持本次营养建议，暂未生成建议。需要补充或重新检索相关可靠资料，补充个人信息不能替代知识依据。"
    if task_type in {"training_plan", "plan_adjust"}:
        return detail + "暂未生成计划。需要补充或重新检索相关可靠资料，已有记录仍然保留。"
    return detail + "暂未生成建议。需要补充或重新检索相关可靠资料。"
