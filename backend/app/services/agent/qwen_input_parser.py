"""Backward-compatible import for the former local-Qwen input parser."""

from app.services.agent.deepseek_input_parser import (
    DeepSeekInputParser,
    ParsedFood,
    ParsedFoodResponse,
    ParsedOutline,
    ParsedRequest,
    ParsedTask,
    ParsedTaskOutline,
    ParsedWorkout,
    QwenInputParser,
    TaskAction,
    TaskType,
)

__all__ = [
    "DeepSeekInputParser", "QwenInputParser", "ParsedFood", "ParsedFoodResponse",
    "ParsedOutline", "ParsedRequest", "ParsedTask", "ParsedTaskOutline",
    "ParsedWorkout", "TaskAction", "TaskType",
]
