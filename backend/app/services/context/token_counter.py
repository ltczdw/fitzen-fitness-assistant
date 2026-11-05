from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class TokenCounter:
    def __init__(self, tokenizer_path: str | None = None) -> None:
        self.tokenizer = None
        path = settings.context_tokenizer_path if tokenizer_path is None else tokenizer_path
        if path:
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(str(Path(path)))
            except (ImportError, OSError, ValueError):
                logger.warning("Configured tokenizer unavailable; using conservative UTF-8 budget")
        if tokenizer_path is None and not path:
            from app.services.context.request_tokens import native_plain_tokenizer
            self.tokenizer = native_plain_tokenizer()
        # An unrelated tokenizer must not be presented as exact for DeepSeek.
        # UTF-8 bytes are a conservative bound for byte-level text tokenization.
        self.mode = "model-tokenizer" if self.tokenizer is not None else "conservative-utf8"

    @lru_cache(maxsize=2048)
    def text(self, content: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(content, add_special_tokens=False).ids)
        return len(content.encode("utf-8"))

    def messages(self, messages: list[dict[str, str]]) -> int:
        return 8 + sum(16 + self.text(item.get("role", "")) + self.text(item.get("content", ""))
                       for item in messages)
