"""Tests for profile read contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "plan_dialogue"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_profile_schema_exposes_canonical_enum():
    from app.services.agent.semantics import RequestUnderstanding

    schema = json.dumps(RequestUnderstanding.model_json_schema(), ensure_ascii=False)
    for name in ("height_cm", "body_weight_kg", "target_weight_kg", "available_days"):
        assert name in schema
    for alias in ("current_weight", "weekly_training_days", "training_days_per_week"):
        assert alias not in schema


def test_replay_profile_bad_fields_rejected(monkeypatch):
    from app.services.agent import semantics
    from app.services.agent.read_resolver import read_spec_answer
    from app.db import profile_repository

    fixture = _fixture("profile_read")
    source = fixture["source_message"]
    for item in fixture["items"]:
        raw = json.loads(item["arguments"])
        with pytest.raises(semantics.SemanticValidationError) as error:
            semantics.validate_understanding(source, raw, context={})
        assert error.value.code == "semantic_schema_invalid"

    def forbidden(*args, **kwargs):
        raise AssertionError("a malformed field contract must not read storage")

    monkeypatch.setattr(profile_repository, "get_stored_profile", forbidden)
    result = read_spec_answer("demo-user", {
        "resource": "profile", "fields": ["height", "current_weight"]})
    assert result["status"] == "unavailable"
    assert result["error"] == "read_spec_schema_invalid"
    assert "175" not in result["answer"]


def test_profile_correction_prompt_contains_canonical_values():
    from app.services.agent import semantics

    prompt = semantics.PROMPT
    for name in ("height_cm", "body_weight_kg", "target_weight_kg", "available_days"):
        assert name in prompt
    assert "不要用height/weight/current_weight" in prompt
    assert "resource=profile" in prompt


def test_profile_read_renders_175_70_68_3(compact_db):
    from app.db import profile_repository
    from app.services.agent.read_resolver import read_spec_answer

    profile_repository.save_profile("demo-user", {
        "height_cm": 175, "body_weight_kg": 70, "target_weight_kg": 68,
        "available_days": 3, "body_fat_pct": None,
    })
    result = read_spec_answer("demo-user", {
        "resource": "profile",
        "fields": ["height_cm", "body_weight_kg", "target_weight_kg", "available_days"],
    })
    assert result["authoritative"] is True and result["model_used"] is False
    assert "175cm" in result["answer"] and "70kg" in result["answer"]
    assert "68kg" in result["answer"] and "3天" in result["answer"]


def test_profile_missing_field_no_row_cross_user(compact_db):
    from app.db import profile_repository
    from app.services.agent.read_resolver import read_spec_answer

    profile_repository.save_profile("demo-user", {
        "height_cm": 175, "body_weight_kg": 70, "target_weight_kg": 68,
        "available_days": 3,
    })
    missing = read_spec_answer("demo-user", {"resource": "profile", "fields": ["body_fat_pct"]})
    assert "未填写" in missing["answer"] and "体脂率" in missing["answer"]

    no_row = read_spec_answer("nobody", {"resource": "profile", "fields": ["height_cm"]})
    assert "没有找到你的档案记录" in no_row["answer"]

    other = read_spec_answer("other-user", {"resource": "profile", "fields": ["height_cm"]})
    assert "没有找到你的档案记录" in other["answer"]
    assert "175" not in other["answer"]


def test_profile_alias_is_not_normalized(compact_db):
    from app.db import profile_repository
    from app.services.agent.read_resolver import read_spec_answer

    profile_repository.save_profile("demo-user", {"height_cm": 175, "body_weight_kg": 70})
    result = read_spec_answer("demo-user", {
        "resource": "profile", "fields": ["height", "current_weight"]})
    assert result["status"] == "unavailable"
    assert "175" not in result["answer"] and "70" not in result["answer"]


def test_profile_read_pg(monkeypatch, tmp_path):
    from tests.react_phase_c1_support import isolated
    from app.db import profile_repository
    from app.services.agent.read_resolver import read_spec_answer

    with isolated(monkeypatch, tmp_path, "postgres"):
        profile_repository.save_profile("demo-user", {
            "height_cm": 175, "body_weight_kg": 70, "target_weight_kg": 68,
            "available_days": 3,
        })
        result = read_spec_answer("demo-user", {
            "resource": "profile",
            "fields": ["height_cm", "body_weight_kg", "target_weight_kg", "available_days"],
        })
        assert "175cm" in result["answer"] and "3天" in result["answer"]
