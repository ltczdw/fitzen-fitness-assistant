from __future__ import annotations

import asyncio

from app.services.agent.intent import IntentResult, classify_intent


class DeepSeekIntentClient:
    """Async wrapper around the shared DeepSeek intent gateway."""

    async def classify(self, message: str) -> IntentResult:
        # classify_intent owns the deterministic safety gate and LLM fallback.
        return await asyncio.to_thread(classify_intent, message)


deepseek_intent = DeepSeekIntentClient()


async def classify_intent_with_deepseek(message: str) -> IntentResult:
    return await deepseek_intent.classify(message)
