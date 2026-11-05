import logging
from pathlib import Path
from urllib.parse import urlsplit
from typing import Literal
from dotenv import dotenv_values

from pydantic import field_validator, model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, JsonConfigSettingsSource


PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
PROJECT_RUNTIME_SETTINGS_FILE = Path(__file__).resolve().parents[2] / "runtime-settings.json"


class Settings(BaseSettings):
    app_env: str = "development"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    backend_host: str = "127.0.0.1"
    backend_port: int = Field(default=8001, ge=1, le=65535)
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=9000, ge=1, le=65535)
    database_url: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 60.0
    llm_stream_timeout_seconds: float = 30.0
    llm_retry_delays: str = "2,4,10"
    # Physical model attempts: shared across stages, retries and threads.
    model_http_max_attempts: int = Field(default=2, ge=1, le=3)
    model_stage_max_calls: int = Field(default=2, ge=1, le=3)
    # New chat requests aggregate real root usage but do not enforce this cap.
    # Old/frozen requests without a per-task policy marker keep the old shared
    # request semantics, and low-level explicit model_scope() calls retain them.
    model_request_max_calls: int = Field(default=12, ge=1, le=64)
    model_request_max_tokens: int = Field(default=64000, ge=1000, le=1000000)
    model_request_deadline_seconds: float = Field(default=120, gt=0, le=600)
    # One durable allowance for every new business task, regardless of task
    # type or route. chat/embedding/rerank/retries/corrections all share it.
    model_task_max_calls: int = Field(default=12, ge=1, le=32)
    model_task_max_tokens: int = Field(default=64000, ge=1000, le=1000000)
    # Deprecated compatibility fields. They exist only so older .env files
    # still load; the task bridge never reads them and no task type selects a
    # different allowance. An old MODEL_TASK_STANDARD_MAX_CALLS=8 is ignored.
    model_task_standard_max_calls: int = Field(
        default=8, ge=1, le=32,
        deprecated="Deprecated and ignored; use model_task_max_calls.")
    model_task_plan_max_calls: int = Field(
        default=12, ge=1, le=32,
        deprecated="Deprecated and ignored; use model_task_max_calls.")
    model_prepare_max_calls: int = Field(default=8, ge=1, le=32)
    model_daily_max_calls: int = Field(default=200, ge=1, le=10000)
    model_daily_max_tokens: int = Field(default=1000000, ge=1000, le=100000000)
    model_background_daily_max_calls: int = Field(default=20, ge=0, le=200)
    model_background_max_calls: int = Field(default=5, ge=1, le=10)
    model_background_max_tokens: int = Field(default=24000, ge=1000, le=1000000)
    model_cli_max_calls: int = Field(default=50, ge=1, le=200)
    model_cli_max_tokens: int = Field(default=500000, ge=1000, le=10000000)
    model_cli_deadline_seconds: float = Field(default=1800, gt=0, le=7200)
    model_retry_backoff_seconds: float = Field(default=1, ge=0, le=30)
    memory_outbox_max_attempts: int = Field(default=3, ge=1, le=10)
    memory_outbox_ttl_seconds: float = Field(default=1800, gt=0, le=86400)
    # Shared per process and upstream configuration; all retries share a deadline.
    llm_max_concurrency: int = Field(default=4, ge=1, le=128)
    llm_requests_per_second: float = Field(default=2.0, gt=0, le=1000)
    llm_rate_burst: int = Field(default=4, ge=1, le=1000)
    llm_total_deadline_seconds: float = Field(default=90.0, gt=0, le=600)
    llm_circuit_failure_threshold: int = 3
    llm_circuit_recovery_seconds: float = 30.0
    llm_context_window_tokens: int = 65536
    llm_max_output_tokens: int = 4096
    context_safety_margin: float = 0.15
    context_tokenizer_path: str = ""
    mcp_url: str = "http://127.0.0.1:9000/mcp"
    mcp_timeout_seconds: float = 8.0
    embedding_model: str = "qwen3.7-text-embedding"
    retrieval_embedding_base_url: str = ""
    retrieval_embedding_api_key: str = ""
    retrieval_embedding_model: str = "qwen3.7-text-embedding"
    retrieval_embedding_dimensions: int = 1024
    retrieval_embedding_index_version: str = "qwen37_1024_v1"
    retrieval_embedding_timeout_seconds: float = 12.0
    retrieval_max_workers: int = Field(default=4, ge=1, le=32)
    retrieval_vector_threshold: float = 0.50
    rag_answer_confidence_threshold: float = 0.48
    memory_read_backend: str = "compact_v1"
    memory_short_window_turns: int = Field(default=10, ge=0, le=50)
    memory_recent_max_tokens: int = Field(default=12000, ge=1000, le=30000)
    memory_turn_soft_tokens: int = 1200
    memory_outbox_interval_seconds: float = 5.0
    # Shadow Planner calls are diagnostic only and default off.
    planner_shadow_enabled: bool = False
    intent_router: str = "planner_llm"
    rag_query_rewrite_enabled: bool = False
    rerank_model: str = "qwen3.7-text-rerank"
    rerank_enabled: bool = True
    rerank_provider: Literal["dashscope", "jina_cohere"] = "dashscope"
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_legacy_embedding_config: bool = True
    rerank_timeout_seconds: float = 15.0
    rerank_min_score: float = 0.10
    wiki_enabled: bool = True
    # Bounds for one business request, independent of the legacy archive window.
    wiki_memory_context_fraction: float = Field(default=0.60, gt=0, le=0.85)
    wiki_catalog_max_tokens: int = Field(default=3000, ge=256, le=16000)
    wiki_catalog_batch_size: int = Field(default=32, ge=1, le=128)
    wiki_memory_max_rounds: int = Field(default=3, ge=1, le=8)
    # One bounded structured extraction per request for selected originals.
    # A failure keeps the complete page; it never narrows the safety line.
    wiki_semantic_units_enabled: bool = True
    wiki_semantic_unit_max_calls: int = Field(default=2, ge=1, le=4)
    wiki_semantic_unit_max_pages: int = Field(default=8, ge=1, le=32)
    wiki_semantic_unit_max_input_tokens: int = Field(default=16000, ge=1000, le=60000)
    # Independent source-read budget: a page may be read and inspected for
    # bounded extraction even when its complete text cannot enter the final
    # context. It is not the final answer context budget.
    wiki_source_read_max_tokens: int = Field(default=14000, ge=1000, le=60000)
    # Optional unit ranking is an explicit ablation switch, not a truth rule.
    # "density" uses value/full incremental closure cost; "value" disables the
    # density denominator while keeping the same budget and closure accounting.
    wiki_unit_ranking: Literal["density", "value"] = "density"
    # Gentle, bounded age weighting only for units the source-bound model marked
    # as time-sensitive. Stable records, history views and required units keep
    # their full score. A fixed request as_of makes the result reproducible.
    wiki_temporal_decay_enabled: bool = True
    wiki_temporal_half_life_days: float = Field(default=30.0, gt=0, le=3650)
    wiki_temporal_decay_floor: float = Field(default=0.85, ge=0, lt=1)
    memory_transient_ttl_seconds: int = Field(default=86400, ge=60, le=604800)
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 80
    rag_dedupe_similarity_threshold: float = 0.92
    rag_enable_semantic_dedupe: bool = False
    rag_parent_context_chars: int = 800
    rag_crawl_timeout_seconds: float = 15.0
    rag_crawl_max_pages: int = 40
    rag_crawl_max_depth: int = 1
    app_timezone: str = "Asia/Shanghai"
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_enabled: bool = True
    memory_short_window_messages: int = 12
    memory_short_ttl_seconds: int = 604800
    # Autonomous model-backed memory processing is on by default (2026-10-02,
    # user decision); set MEMORY_BACKGROUND_ENABLED=false to opt out. Records
    # and history reads remain enabled regardless.
    memory_background_enabled: bool = True
    memory_summary_enabled: bool = True
    memory_summary_timeout_seconds: float = 30.0
    workout_parser_enabled: bool = True
    input_parser_enabled: bool = True
    # ReAct 只读循环开关：默认开启，产品入口（前端）默认发送 react_readonly；纯读/问答/
    # 建议进入只读循环，含写/确认/混合整请求仍按 select_mode 回到 legacy 兜底。
    # 无模型凭证的离线环境可显式设为 false，恢复 legacy 规则路径。
    react_readonly_enabled: bool = True
    # C2-3 写入运行时开关：默认开启，显式 react_write 在关闭/非 compact/失计量时明确
    # execution_limited，零模型/领域派发，不回落 legacy。
    react_write_enabled: bool = True
    # 客户端未声明 mode 时的默认执行模式。auto：按已验证语义自动选择
    # react_write（含写且写入运行时就绪）/ react_readonly（纯读）/ legacy（其余）。
    # 既有测试套件在 tests/conftest.py 的 session fixture 中固定为 legacy 作为行为基线，
    # 由 test_chat_mode_default.py 单独验证生产默认值与自动解析规则。
    chat_default_mode: Literal["auto", "legacy", "react_readonly"] = "auto"
    chat_history_max_messages: int = 200
    chat_history_ttl_seconds: int = 2592000
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "fitness-assistant"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(env_file=PROJECT_ENV_FILE, extra="ignore")

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        # Versioned, non-secret runtime limits can override an old local .env
        # without opening/editing that credential file. Explicit process env
        # and constructor values retain the normal deployment override order.
        return (init_settings, env_settings,
                JsonConfigSettingsSource(settings_cls, json_file=PROJECT_RUNTIME_SETTINGS_FILE,
                                         json_file_encoding="utf-8"),
                dotenv_settings, file_secret_settings)

    @model_validator(mode="after")
    def normalize_official_deepseek_model_alias(self):
        # The official Flash endpoint renamed its model ID. Keep existing
        # project configuration usable without editing secrets or switching tiers.
        if (urlsplit(self.llm_base_url).hostname == "api.deepseek.com"
                and self.llm_model == "deepseek-v4-flash"):
            self.llm_model = "deepseek-flash"
        return self

    @field_validator("embedding_model", mode="before")
    @classmethod
    def normalize_embedding_model(cls, value: str | None) -> str:
        return value or "qwen3.7-text-embedding"

    @field_validator("rerank_model", mode="before")
    @classmethod
    def normalize_rerank_model(cls, value: str | None) -> str:
        return value or "qwen3.7-text-rerank"

    @field_validator("rag_answer_confidence_threshold")
    @classmethod
    def validate_rag_answer_confidence_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("rag_answer_confidence_threshold must be between 0 and 1")
        return value


def warn_unknown_env_keys(path: Path = PROJECT_ENV_FILE) -> list[str]:
    """Report key names only; never include secrets or configuration values."""
    known = {name.casefold() for name in Settings.model_fields}
    unknown = sorted(key for key in dotenv_values(path) if key.casefold() not in known) if path.exists() else []
    if unknown:
        logging.getLogger(__name__).warning("Unrecognized .env keys: %s", ", ".join(unknown))
    return unknown


settings = Settings()
