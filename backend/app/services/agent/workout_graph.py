from __future__ import annotations

import logging
from typing import Any, TypedDict

from app.services.agent.workout_extractor import validate_workout_candidate
from app.services.agent.deepseek_workout import DeepSeekWorkoutExtractor
from app.services.llm.client import LLMClient
from app.services.llm.budget import model_stage, key


logger = logging.getLogger(__name__)


class WorkoutGraphState(TypedDict, total=False):
    message: str
    preprocessed: dict[str, Any]
    llm_output: dict[str, Any] | None
    llm_attempts: int
    retry_count: int
    record: dict[str, Any] | None
    status: str
    missing_fields: list[str]
    inferred_fields: list[str]
    normalized_text: str
    number_mentions: list[dict[str, Any]]
    reason: str


class WorkoutExtractionWorkflow:
    """Model-only entity extraction with bounded retries and typed validation."""

    def __init__(self, llm: LLMClient, model_extractor: DeepSeekWorkoutExtractor | None = None) -> None:
        self.llm = llm
        self.model_extractor = model_extractor

    async def run(self, message: str) -> dict[str, Any]:
        with model_stage("workout_extract:" + key(message), 2):
            return await self._run_impl(message)

    @staticmethod
    def _unavailable(message: str) -> dict[str, Any]:
        return {"message": message, "status": "incomplete", "record": None,
                "missing_fields": ["动作名称、重量、组数和次数"], "inferred_fields": [],
                "normalized_text": message, "number_mentions": [], "reason": "semantic_model_unavailable"}

    async def _run_impl(self, message: str) -> dict[str, Any]:
        records = await self.run_many(message)
        return records[0] if len(records) == 1 else self._unavailable(message)

    @staticmethod
    def _rule_state(message: str) -> dict[str, Any]:
        """Deterministic entities for the no-model path, validated from source."""
        from app.services.agent.workout_extractor import preprocess_workout
        state = preprocess_workout(message).to_state()
        state.update({"message": message, "llm_attempts": 0, "parser": "rule"})
        return state

    @staticmethod
    def _llm_configured(llm: Any) -> bool:
        configured = getattr(llm, "configured", None)
        try:
            return bool(configured()) if callable(configured) else False
        except Exception:
            return False

    async def run_many(self, message: str) -> list[dict[str, Any]]:
        """Model interprets exercise entities; deterministic parsing only when no model exists."""
        if self.model_extractor is not None:
            records = await self.model_extractor.extract(message)
            if records:
                return [self._model_record_state(record, message) for record in records]
            if not self._llm_configured(self.llm):
                return [self._rule_state(message)]
            return [self._unavailable(message)]
        if not self._llm_configured(self.llm):
            return [self._rule_state(message)]
        with model_stage("workout_extract:" + key(message), 2):
            for attempt in range(2):
                try:
                    candidate = await self.llm.extract_workout_json(
                        message=message, preprocessed={}, enhanced=attempt > 0)
                except Exception as exc:
                    from app.services.llm.budget import is_budget_policy_stop
                    if is_budget_policy_stop(exc):
                        raise
                    logger.warning("workout_model_extraction_unavailable: %s", type(exc).__name__)
                    candidate = None
                state = self._model_record_state(candidate, message) if candidate else self._unavailable(message)
                state.update(llm_attempts=attempt + 1, retry_count=attempt)
                if state.get("record"):
                    return [state]
            return [state]

    @staticmethod
    def _model_record_state(candidate: dict[str, Any], message: str, *, reference_date=None) -> dict[str, Any]:
        result = validate_workout_candidate(
            candidate,
            message,
            {"normalized_text": message, "inferred_fields": [], "number_mentions": []},
            reference_date=reference_date,
        )
        state = result.to_state()
        state.update({"message": message, "llm_attempts": 1, "parser": "deepseek"})
        return state
