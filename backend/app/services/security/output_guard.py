from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str


# Technical redaction and markup neutralisation only. Medical/medication/
# absolute-wording/unsafe-activity meaning is not rewritten here: the generating
# model follows the shared safety instruction and the source-bound safety
# judgment. Rewriting negated or quoted sentences with fixed strings corrupted
# real replies ("不要忍痛训练", "我不能保证训练效果", "请勿自行停药").
OUTPUT_RULES: tuple[OutputRule, ...] = (
    OutputRule(
        "secret_leak",
        re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret|password)\s*[:=]\s*\S+"),
        "<已隐藏凭证>",
    ),
    OutputRule(
        "pii_leak",
        re.compile(r"(?:1[3-9]\d{9}|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|\d{17}[\dXx])", re.I),
        "<已隐藏个人信息>",
    ),
    OutputRule(
        "unsafe_markup",
        re.compile(r"(?is)<script[^>]*>.*?</script>|javascript\s*:"),
        "",
    ),
)


@dataclass(frozen=True)
class OutputGuardResult:
    text: str
    triggered_rules: tuple[str, ...]
    blocked: bool


def apply_output_guardrail(text: str) -> OutputGuardResult:
    current = text or ""
    triggered: list[str] = []
    for rule in OUTPUT_RULES:
        updated, count = rule.pattern.subn(rule.replacement, current)
        if count:
            triggered.append(rule.name)
            current = updated
    return OutputGuardResult(current, tuple(triggered), bool(triggered))
