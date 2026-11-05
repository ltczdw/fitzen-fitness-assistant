"""SIM30 R3 item 4: truthful plan-save capability boundary."""
from __future__ import annotations

import json
import re
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sim30_r2"
APP_VUE = Path(__file__).resolve().parents[2] / "frontend" / "src" / "fitzen" / "App.vue"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_replay_plan_save_dialogue_answer_uses_capability():
    from app.services.agent.dialogue_context import DialogueAnswer, execution_status_text

    fixture = _fixture("plan_save_boundary")
    item = next(item for item in fixture["items"] if item["source_call"] == 81)
    answer = DialogueAnswer.model_validate(json.loads(item["arguments"]))
    assert answer.status_kind == "save_permission"
    assert answer.capability_kind == "plan_save"
    text = execution_status_text(answer.answer, answer.status_kind, answer.capability_kind)
    assert "没有保存训练计划的工具" in text
    assert "不能" in text and "声称已经保存" in text
    assert "当前版本没有可确认的计划保存页面或入口" in text
    assert text != "不会保存。"


def test_plan_save_copy_has_no_fake_page():
    from app.services.agent.capability_catalog import CAPABILITY_REGISTRY

    entry = CAPABILITY_REGISTRY["plan_save"]
    assert entry.entry_screen is None
    assert entry.allows_chat_write is False
    for phrase in ("计划页面", "保存按钮", "点击", "请到"):
        assert phrase not in entry.copy


def test_capability_entries_point_to_real_screens():
    from app.services.agent.capability_catalog import CAPABILITY_REGISTRY

    source = APP_VUE.read_text(encoding="utf-8")
    match = re.search(r"const destinations=\[(.*?)\]", source)
    assert match
    destinations = set(re.findall(r"id:'([^']+)'", match.group(1)))
    for entry in CAPABILITY_REGISTRY.values():
        if entry.entry_screen is not None:
            assert entry.entry_screen in destinations, entry.kind
    assert CAPABILITY_REGISTRY["plan_save"].entry_screen is None


def test_save_result_without_receipt_does_not_claim_saved():
    from app.services.agent.dialogue_context import execution_status_text

    text = execution_status_text("已保存", "save_result", None, receipt_refs=[], commit_state=None)
    assert "不能确认已保存" in text
    assert "已保存。" not in text


def test_save_result_with_receipt_claims_saved():
    from app.services.agent.dialogue_context import execution_status_text

    text = execution_status_text("已保存", "save_result", None,
                                 receipt_refs=["receipt-1"], commit_state="committed")
    assert text == "已保存。"


def test_status_only_save_permission_without_capability():
    from app.services.agent.dialogue_context import execution_status_text

    assert execution_status_text("", "save_permission") == "不会保存。"


def test_plan_save_no_dml(compact_db):
    from app.db import plan_repository
    from app.services.agent.dialogue_context import DialogueAnswer, execution_status_text

    fixture = _fixture("plan_save_boundary")
    item = next(item for item in fixture["items"] if item["source_call"] == 81)
    answer = DialogueAnswer.model_validate(json.loads(item["arguments"]))
    before = plan_repository.get_stored_plan("demo-user")
    execution_status_text(answer.answer, answer.status_kind, answer.capability_kind)
    assert plan_repository.get_stored_plan("demo-user") == before
