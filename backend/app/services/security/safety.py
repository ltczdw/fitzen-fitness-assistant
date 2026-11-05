"""Source-bound structured safety judgment contract.

Emergency meaning is a semantic decision. The model returns a bounded
``SafetyAssessment`` inside the existing request-level ``RequestUnderstanding``
call; the application only verifies enum values, source identity and the
verbatim evidence span. No keyword list or regular expression in this module
decides normal/urgent.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

URGENT_SAFETY_ANSWER = (
    "检测到可能存在需要及时处理的危险信号。请立即停止训练；"
    "如有胸痛、呼吸困难、意识异常、明显变形或持续加重，请尽快联系急救或到急诊就医。"
)

SafetyLevel = Literal["normal", "medical", "urgent"]
SafetyCategory = Literal[
    "current_symptom",
    "medical_history",
    "consultation",
    "prevention",
    "negated",
    "medication",
    "mental_health",
    "other",
]


class SafetyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: SafetyLevel = Field(
        description="normal=无当前医疗风险；medical=有医疗背景但非当前急症；urgent=当前可能出现需立即处理的安全风险。"
    )
    evidence: str = Field(
        default="", max_length=4000,
        description="逐字引用当前query中支撑该判断的连续片段；normal可为空。不能引用历史或改写。",
    )
    categories: list[SafetyCategory] = Field(default_factory=list, max_length=12)
    reason: str = Field(default="", max_length=400,
                        description="简短说明语境（当前/否定/历史/咨询/预防），不进行诊断。")

    def bound_to(self, source: str) -> bool:
        """Evidence must be a real current-query span for non-normal judgments."""
        if self.level == "normal":
            return not self.evidence or self.evidence in source
        return bool(self.evidence.strip()) and self.evidence in source


__all__ = ["SafetyAssessment", "SafetyCategory", "SafetyLevel", "URGENT_SAFETY_ANSWER"]
