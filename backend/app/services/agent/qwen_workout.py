"""Backward-compatible imports for the former local-Qwen workout parser.

The implementation now calls DeepSeek through the shared LLMClient.
"""

from app.services.agent.deepseek_workout import (
    DeepSeekWorkoutExtractor,
    DeepSeekWorkoutItem,
    DeepSeekWorkoutResponse,
)

QwenWorkoutExtractor = DeepSeekWorkoutExtractor
QwenWorkoutItem = DeepSeekWorkoutItem
QwenWorkoutResponse = DeepSeekWorkoutResponse

__all__ = [
    "DeepSeekWorkoutExtractor",
    "DeepSeekWorkoutItem",
    "DeepSeekWorkoutResponse",
    "QwenWorkoutExtractor",
    "QwenWorkoutItem",
    "QwenWorkoutResponse",
]
