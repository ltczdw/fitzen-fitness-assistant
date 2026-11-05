from uuid import uuid4
from typing import Literal

from app.core.config import settings
from app.schemas.citation import Citation, normalize_citations
from app.schemas.clarification import TaskClarification

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfirmationTarget(BaseModel):
    """C2-3 确认选择器：三个严格字段，服务端按同 owner/conversation 的真实原请求/op/
    pending 加载；额外字段 422，格式错误 422（不猜最新 pending）。"""

    model_config = ConfigDict(extra="forbid")

    pending_request_id: str = Field(min_length=1, max_length=100)
    pending_op_id: str = Field(pattern=r"^op2_[0-9a-f]{32}$")
    pending_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("pending_request_id")
    @classmethod
    def normalize_selector(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("标识不能为空")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    user_id: str = "demo-user"
    conversation_id: str = "demo-conversation"
    # 执行模式：legacy 保持原流程；react_readonly 在开关开启且纯读/问答/建议时进入
    # 阶段B只读循环；react_write 在开关开启且 compact_v1/DB/账本/预算就绪时进入
    # C2-3 写入运行时。未知值由 Pydantic 422 拒绝。
    #
    # 未声明 mode 时取 settings.chat_default_mode：产品默认 auto，由服务端在
    # 通过语义/安全门后自动选择 react_write / react_readonly / legacy。
    # 显式 mode 始终优先，未知值由 Pydantic 422 拒绝。
    mode: Literal["auto", "legacy", "react_readonly", "react_write"] = Field(
        default_factory=lambda: settings.chat_default_mode
    )
    # C2-3 确认选择器：仅 react_write 模式合法；三字段严格校验。
    confirmation: ConfirmationTarget | None = None
    # The client can reuse this value when retrying a request.  A fresh
    # default keeps two intentional, identical messages from sharing a
    # nutrition/workout idempotency key.
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=100)

    @model_validator(mode="after")
    def confirmation_only_write(self):
        if self.confirmation is not None and self.mode != "react_write":
            raise ValueError("confirmation 仅支持 react_write 模式")
        return self

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("消息不能为空")
        return value

    @field_validator("user_id", mode="before")
    @classmethod
    def single_person(cls, value):
        return "demo-user"

    @field_validator("user_id", "conversation_id", "request_id")
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("标识不能为空")
        return value


class ChatResponse(BaseModel):
    answer: str
    intent: str
    citations: list[Citation] = Field(default_factory=list)
    clarifications: list[TaskClarification] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def structured_citations(cls, value):
        return normalize_citations(value)

    safety_level: str = "normal"
    model_used: bool = False
    memory_status: str = "ready"
    diagnostics_status: str = "ready"
    request_id: str | None = None
    message_id: str | None = None
    run_id: str | None = None
    replayed: bool = False
    metrics: dict = Field(default_factory=dict)
