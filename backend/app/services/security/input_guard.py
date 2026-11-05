from __future__ import annotations

import re
from dataclasses import dataclass


# Level 1: identifiers/secrets. Level 2: health/medical context. These are
# technical privacy/format classifications. Emergency meaning is *not* decided
# here: a source-bound structured model judgment owns that semantic contract.
LEVEL1_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "<已脱敏手机号>"),
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "<已脱敏邮箱>"),
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "<已脱敏证件号>"),
    ("secret", re.compile(r"(?i)\b(?:api[_ -]?key|token|secret|password)\s*[:=]\s*\S+"), "<已脱敏凭证>"),
    ("address", re.compile(r"(?:住址|家庭住址|详细地址)\s*[:：]?\s*[^，。；;\n]{4,60}"), "<已脱敏地址>"),
)

LEVEL2_TERMS = (
    "病历",
    "诊断",
    "处方",
    "用药",
    "药物",
    "手术",
    "住院",
    "就诊",
    "化验单",
    "检查报告",
    "医保号",
)

@dataclass(frozen=True)
class InputGuardResult:
    level: str
    categories: tuple[str, ...]
    sanitized_text: str
    matched_terms: tuple[str, ...]


def redact_phi(text: str) -> str:
    sanitized = text
    for _, pattern, replacement in LEVEL1_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def inspect_input(text: str) -> InputGuardResult:
    """Technical privacy classification only.

    ``level`` never becomes ``"urgent"``. Emergency/medical meaning is decided
    solely from the model's source-bound SafetyAssessment (see agent.semantics);
    this function must not be used as a lexical safety classifier.
    """
    matched: list[str] = []
    categories: list[str] = []
    sanitized = redact_phi(text)
    for name, pattern, _ in LEVEL1_PATTERNS:
        if pattern.search(text):
            matched.append(name)
            categories.append("level1_sensitive")
    for term in LEVEL2_TERMS:
        if term in text:
            matched.append(term)
            categories.append("level2_phi")
    if "level2_phi" in categories:
        level = "phi"
    elif "level1_sensitive" in categories:
        level = "sensitive"
    else:
        level = "normal"
    return InputGuardResult(level, tuple(dict.fromkeys(categories)), sanitized,
                            tuple(dict.fromkeys(matched)))
