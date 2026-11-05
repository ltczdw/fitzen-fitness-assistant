"""Production chat-mode default, explicit overrides and auto resolution."""
from app.core.config import Settings, settings
from app.schemas.chat import ChatRequest
from app.services.agent.react_runtime import resolve_auto_mode


class _Agent:
    """Only the write gate is consulted by auto resolution."""

    def __init__(self, gate=None):
        self.gate = gate

    def _write_gate_reason(self):
        return self.gate


WRITE_TASK = {"task_type": "workout_record", "action": "record", "side_effect": True}


def test_production_default_is_auto():
    assert Settings.model_fields["chat_default_mode"].default == "auto"
    assert Settings.model_fields["react_readonly_enabled"].default is True
    assert Settings.model_fields["react_write_enabled"].default is True


def test_unpinned_request_defaults_to_auto(monkeypatch):
    monkeypatch.setattr(settings, "chat_default_mode", "auto")
    assert ChatRequest(message="今天训练多久合适").mode == "auto"


def test_explicit_modes_always_win():
    for mode in ("legacy", "react_readonly", "react_write"):
        assert ChatRequest(message="今天训练多久合适", mode=mode).mode == mode


def test_suite_baseline_pins_legacy():
    # tests/conftest.py pins the old baseline for the existing suite; this keeps
    # the production default above from silently rerouting those tests.
    assert ChatRequest(message="今天训练多久合适").mode == "legacy"


def test_auto_pure_read_selects_readonly():
    request = ChatRequest(message="今天训练多久合适")
    assert resolve_auto_mode(
        _Agent(), request, {"tasks": [{"task_type": "training_qa"}]}
    ) == ("react_readonly", "pure_read")


def test_auto_write_selects_react_write_when_gates_ready(monkeypatch):
    from app.db import session as db

    monkeypatch.setattr(db, "DB_READY", True)
    request = ChatRequest(message="记录卧推 60kg 4组8次")
    assert resolve_auto_mode(_Agent(), request, {"tasks": [WRITE_TASK]}) == (
        "react_write", "auto_write")


def test_auto_write_falls_back_to_legacy_when_gate_closed():
    request = ChatRequest(message="记录卧推 60kg 4组8次")
    assert resolve_auto_mode(_Agent("react_write_disabled"), request, {"tasks": [WRITE_TASK]}) == (
        "legacy", "react_write_disabled")


def test_auto_write_falls_back_to_legacy_without_database():
    request = ChatRequest(message="记录卧推 60kg 4组8次")
    assert resolve_auto_mode(_Agent(), request, {"tasks": [WRITE_TASK]}) == (
        "legacy", "database_unavailable")


def test_auto_preview_uses_react_and_conversation_stays_legacy():
    preview = {"task_type": "training_plan", "action": "question",
               "delivery_type": "preview", "goal_role": "result"}
    assert resolve_auto_mode(
        _Agent(), ChatRequest(message="把周二改成30分钟"), {"tasks": [preview]}
    )[0] == "react_readonly"
    assert resolve_auto_mode(
        _Agent(), ChatRequest(message="谢谢"), {"tasks": []}
    )[0] == "legacy"
