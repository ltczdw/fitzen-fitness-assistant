"""Backward-compatible imports for the former local-Qwen summary client."""

from app.services.memory.deepseek_summary import (
    DeepSeekSummaryClient,
    MessageImportance,
    SummaryItem,
    SummaryResult,
    deepseek_summary,
)

QwenSummaryClient = DeepSeekSummaryClient
qwen_summary = deepseek_summary

__all__ = [
    "DeepSeekSummaryClient", "MessageImportance", "SummaryItem", "SummaryResult",
    "deepseek_summary", "QwenSummaryClient", "qwen_summary",
]
