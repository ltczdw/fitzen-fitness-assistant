"""One complete-chat-request estimate for both window and durable reservation.

Official pinned V4.1 vocabulary and prompt renderer run locally. Input is still
an estimate: literal control-token spellings are encoded as plain text, and a
protocol allowance is retained. Other providers keep the UTF-8 upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from app.core.config import settings

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "deepseek_v41"
FLASH_MODELS = frozenset({"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp", "deepseek-v4.1-flash"})


@dataclass(frozen=True)
class RequestTokens:
    input_tokens: int
    mode: str
    protocol_allowance: int = 0


@lru_cache(maxsize=1)
def official_plain_tokenizer():
    from tokenizers import Tokenizer
    manifest = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))
    for name in ("tokenizer.json", "encoding.py"):
        raw = (VENDOR / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError("tokenizer_asset_hash_mismatch")
    data = json.loads((VENDOR / "tokenizer.json").read_text(encoding="utf-8"))
    # User-typed special token spellings must count as text, not control IDs.
    data["added_tokens"] = []
    data["post_processor"] = None
    return Tokenizer.from_str(json.dumps(data, ensure_ascii=False))


def matching_native_model(model: str | None = None) -> bool:
    return (urlsplit(settings.llm_base_url).hostname == "api.deepseek.com"
            and (settings.llm_model if model is None else model) in FLASH_MODELS)


def native_plain_tokenizer():
    if not matching_native_model():
        return None
    try:
        return official_plain_tokenizer()
    except (ImportError, OSError, ValueError, KeyError):
        return None


def count_request(payload: dict) -> RequestTokens:
    if matching_native_model(payload.get("model")):
        try:
            # Verify the pinned assets before importing the vendored renderer.
            tokenizer = official_plain_tokenizer()
            from app.vendor.deepseek_v41.encoding import encode_messages
            messages = [dict(message) for message in payload.get("messages") or []]
            # Text-only is this application's contract. Never undercount unseen
            # image/tool content by silently dropping unsupported input shapes.
            if any(not isinstance(m.get("content", ""), (str, type(None))) for m in messages):
                raise ValueError("unsupported_request_content")
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": ""})
            if payload.get("tools"):
                messages[0]["tools"] = payload["tools"]
            if payload.get("response_format"):
                messages[0]["response_format"] = payload["response_format"]
            thinking = (payload.get("thinking") or {}).get("type") != "disabled"
            text, media = encode_messages(messages, thinking_mode="thinking" if thinking else "chat",
                                         reasoning_effort=payload.get("reasoning_effort"),
                                         return_multi_modal_data=True)
            if media.get("images"):
                raise ValueError("unsupported_image_estimation")
            # Control markers are conservatively encoded as plain text too.
            # Unknown API protocol options remain counted as serialized input.
            extras = {k:v for k,v in payload.items() if k not in {
                "model", "messages", "tools", "response_format", "thinking", "reasoning_effort",
                "temperature", "stream", "stream_options", "max_tokens"}}
            if extras:
                text += json.dumps(extras, ensure_ascii=False, sort_keys=True)
            allowance = 128 + 16 * len(messages)
            return RequestTokens(len(tokenizer.encode(text, add_special_tokens=False).ids) + allowance,
                                 "deepseek-v41-estimate", allowance)
        except (ImportError, OSError, ValueError, KeyError, TypeError, AssertionError):
            # Missing assets/unsupported protocol retain the conservative path;
            # never use an unrelated tokenizer or a guessed characters ratio.
            pass
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return RequestTokens(len(encoded), "conservative-utf8")


class PreparedPayload(dict):
    """Gateway-owned immutable count accompanies the exact serialized payload."""
    def __init__(self, payload: dict, tokens: RequestTokens, report: dict):
        super().__init__(payload)
        self.tokens = tokens
        self.report = report
        self.fingerprint = hashlib.sha256(json.dumps(self, ensure_ascii=False, sort_keys=True, default=str).encode()).digest()

    def counted_tokens(self) -> RequestTokens:
        fingerprint = hashlib.sha256(json.dumps(self, ensure_ascii=False, sort_keys=True, default=str).encode()).digest()
        return self.tokens if fingerprint == self.fingerprint else count_request(self)
