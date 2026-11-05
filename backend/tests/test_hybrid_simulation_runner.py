"""Shared acceptance keeps a genuine evidence failure from looking successful."""
from pathlib import Path
import sys
from copy import deepcopy
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_ai_hybrid_simulation import assess, cases, BUSINESS_TABLES, table_deltas, exact_changes_allowed


def test_budget_policy_distinguishes_legacy_request_fields_from_new_task_policy():
    import run_ai_hybrid_simulation as runner
    from app.core.config import Settings, settings
    assert Settings.model_fields["model_task_max_calls"].default == 12
    assert Settings.model_fields["model_task_max_tokens"].default == 64000
    assert Settings.model_fields["model_prepare_max_calls"].default == 8
    policy = runner.budget_policy(settings, daily_enforced=False)
    assert policy["request_policy"] == "per_task_v1"
    assert policy["request_root_enforcement"] == "aggregate_only"
    assert policy["request_root_calls_enforced"] is False
    assert policy["request_root_tokens_enforced"] is False
    assert policy["legacy_request_policy"] == "shared_request"
    assert policy["request_max_calls"] == settings.model_request_max_calls
    assert policy["request_max_tokens"] == settings.model_request_max_tokens
    assert policy["task_policy"] == "uniform_v2"
    assert policy["task_max_calls"] == settings.model_task_max_calls == 12
    assert policy["task_max_tokens"] == settings.model_task_max_tokens
    assert policy["prepare_max_calls"] == settings.model_prepare_max_calls == 8
    # The old tier fields are deprecated and cannot reappear as effective policy.
    assert "task_standard_max_calls" not in policy
    assert "task_plan_max_calls" not in policy
    assert policy["daily_enforced"] is False
    assert policy["daily_max_calls"] == settings.model_daily_max_calls
    assert policy["daily_max_tokens"] == settings.model_daily_max_tokens


def test_mixed_acceptance_requires_two_complete_goals_and_actual_hybrid_evidence():
    case = next(c for c in cases() if c["id"] == "mixed_recap_plan")
    empty = {"workouts": [], "foods": [], "facts": [], "active_facts": 0, "tables": {t: [] for t in BUSINESS_TABLES}}
    response = {"answer": "卧推50，深蹲72，以下计划", "intent": "training_qa", "model_used": True,
        "metrics": {"task_outcomes": [{"task_type": "training_qa", "status": "complete"},
            {"task_type": "training_plan", "status": "complete"}], "retrieval": [
            {"corpus": "knowledge", "lexical_count": 10, "vector_count": 10, "degraded_reason": None},
            {"corpus": "knowledge", "quality_gate": {"answerable": True, "threshold": .48}}]}}
    assert all(assess(case, response, empty, empty, 7, 0).values())
    third = deepcopy(response)
    third["metrics"]["task_outcomes"].append({"task_type": "training_qa", "status": "complete"})
    assert not assess(case, third, empty, empty, 7, 0)["only_recap_and_plan_goals"]
    insufficient = deepcopy(response)
    insufficient["metrics"]["task_outcomes"][1]["status"] = "insufficient_evidence"
    assert not assess(case, insufficient, empty, empty, 7, 0)["both_goals_complete"]
    lexical = deepcopy(response)
    lexical["metrics"]["retrieval"][0]["vector_count"] = 0
    assert not assess(case, lexical, empty, empty, 7, 0)["real_hybrid_without_degradation"]
    fallback = deepcopy(response)
    fallback["metrics"]["degraded_reasons"] = ["answer_model_fallback"]
    assert not assess(case, fallback, empty, empty, 7, 0)["answer_generated_without_fallback"]


def test_new_run_covers_original_regression_and_a_different_constraint_position():
    values = cases()
    assert len({c["id"] for c in values}) == len(values)
    assert {"remember", "replace_memory", "workout", "food", "replay", "cancel", "urgent", "unknown"} <= {c["rule"] for c in values}
    alternate = next(c for c in values if c["id"] == "mixed_prefix_constraints")
    assert alternate["message"].startswith("仅提供建议，不保存") and alternate["rule"] == "mixed_read"


def test_workout_cannot_hide_unrelated_plan_or_profile_change():
    before = {t: [] for t in BUSINESS_TABLES}
    after = deepcopy(before)
    after["workout_records"] = [{"id": 1, "user_id": "demo-user", "weight_kg": 50}]
    assert exact_changes_allowed({"rule": "workout"}, table_deltas(before, after))
    after["training_plans"] = [{"id": 1, "goal": "unsolicited"}]
    assert not exact_changes_allowed({"rule": "workout"}, table_deltas(before, after))
    after["training_plans"] = []
    after["user_profiles"] = [{"user_id": "demo-user", "available_days": 5}]
    assert not exact_changes_allowed({"rule": "workout"}, table_deltas(before, after))


def test_full_fact_value_and_source_changes_are_visible_and_not_authorized_by_memory_rule():
    before = {t: [] for t in BUSINESS_TABLES}
    before["user_facts"] = [{"fact_id": "one", "value": "old", "source_quote": "same",
                              "status": "active", "valid_to": None}]
    after = deepcopy(before); after["user_facts"][0]["value"] = "silently-changed"
    assert not exact_changes_allowed({"rule": "remember"}, table_deltas(before, after))
    assert not exact_changes_allowed({"rule": "no_write"}, table_deltas(before, after))
    after = deepcopy(before); after["user_facts"][0].update(status="superseded", valid_to="2026-10-23")
    assert exact_changes_allowed({"rule": "replace_memory", "target_fact_ids": ["one"]}, table_deltas(before, after))
    after["user_facts"][0]["source_quote"] = "changed"
    assert not exact_changes_allowed({"rule": "replace_memory", "target_fact_ids": ["one"]}, table_deltas(before, after))


def test_real_sql_snapshot_captures_all_mutable_fact_fields(compact_db):
    from run_ai_hybrid_simulation import snapshot
    from app.db.memory_models import UserFact
    before = snapshot(compact_db)
    with compact_db() as s, s.begin():
        s.add(UserFact(fact_id="fixture", field="preference", value="first", source_turn_id="synthetic-turn",
            source_message_id="synthetic-user", source_quote="same quote"))
    first = snapshot(compact_db)
    with compact_db() as s, s.begin():
        s.get(UserFact, "fixture").value = "hidden change"
    second = snapshot(compact_db)
    assert set(second["tables"]) == set(BUSINESS_TABLES)
    assert first["facts"] == second["facts"]  # The old projection would miss it.
    assert table_deltas(first["tables"], second["tables"])["user_facts"]["changed"][0]["after"]["value"] == "hidden change"
    assert not exact_changes_allowed({"rule": "remember"}, table_deltas(first["tables"], second["tables"]))


def test_evidence_export_on_halt_and_cleanup_even_if_export_fails(monkeypatch):
    import run_ai_hybrid_simulation as runner
    from types import SimpleNamespace
    actions = []
    def fail_export(factory):
        actions.append("export")
        raise ValueError("synthetic export failure")
    temp = SimpleNamespace(created=True)
    def drop():
        actions.append("drop"); temp.created = False
    temp.drop = drop
    engine = SimpleNamespace(dispose=lambda: actions.append("dispose"))
    monkeypatch.setattr(runner, "export_fixture", fail_export)
    report = {"execution_complete": False, "stop_code": "physical_call_limit"}
    runner.finish_database(report, object(), engine, temp, fixture_ready=True)
    assert actions == ["export", "dispose", "drop"]
    assert report["export_error_type"] == "ValueError" and report["temporary_database_removed"] is True
