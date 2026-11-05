"""Backward-compatible imports for the former local-Qwen intent classifier.

Intent classification now uses the shared DeepSeek LLMClient and deterministic
safety fallback.
"""

from app.services.agent.deepseek_intent import (
    DeepSeekIntentClient,
    classify_intent_with_deepseek,
    deepseek_intent,
)

QwenIntentClient = DeepSeekIntentClient
qwen_intent = deepseek_intent
classify_intent_with_qwen = classify_intent_with_deepseek

__all__ = [
    "DeepSeekIntentClient", "classify_intent_with_deepseek", "deepseek_intent",
    "QwenIntentClient", "qwen_intent", "classify_intent_with_qwen",
]
