import unittest
from contextlib import contextmanager
from unittest.mock import patch

from app.core.config import settings
from app.services.context.budget import BudgetedMessages, ContextBudgetExceeded
from app.services.llm.client import LLMClient, LLMErrorCategory


class _Span:
    def end(self, outputs=None, error=None):
        return None


@contextmanager
def _trace(*args, **kwargs):
    yield _Span()


class LLMContextGateTests(unittest.TestCase):
    def test_payload_reserves_configured_output_tokens(self):
        messages = BudgetedMessages([{"role": "user", "content": "问题"}], {"used_tokens": 10})
        with patch("app.services.llm.client.trace_span", _trace):
            payload = LLMClient._payload(messages, 0.2, True)
        self.assertEqual(payload["max_tokens"], settings.llm_max_output_tokens)
        self.assertTrue(payload["stream"])

    def test_context_overflow_is_non_retryable_protocol_error(self):
        messages = [{"role": "user", "content": "问题"}]
        with patch("app.services.llm.client.context_budget.prepare_request",
                   side_effect=ContextBudgetExceeded(101, 100)), \
             patch("app.services.llm.client.trace_span", _trace):
            with self.assertRaises(ContextBudgetExceeded):
                LLMClient._payload(messages, 0.2, False)
        error = LLMClient._classify_exception(ContextBudgetExceeded(101, 100))
        self.assertEqual(error.category, LLMErrorCategory.PROTOCOL)
        self.assertFalse(error.retryable)


if __name__ == "__main__":
    unittest.main()
