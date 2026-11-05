"""The shared check must fail on failed commands, missing tests and leaked PG DBs."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("project_checks", Path(__file__).parents[1] / "check_project.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_junit_failure_error_and_skip_are_not_passes(tmp_path):
    xml = tmp_path / "results.xml"
    xml.write_text('<testsuites><testsuite><testcase/><testcase><failure/></testcase>'
                   '<testcase><error/></testcase><testcase><skipped/></testcase></testsuite></testsuites>')
    assert checks.junit_counts(xml) == {"passed": 1, "failure": 1, "error": 1, "skipped": 1}


def test_public_checks_keep_private_bound_matrices_out_and_recent_regressions_in(monkeypatch, tmp_path):
    monkeypatch.setattr(checks, 'ROOT', tmp_path)
    seen = []
    def run(command, *, cwd, env, stdout, stderr):
        seen.append(command)
        xml = Path(command[command.index('--junitxml') + 1])
        xml.write_text('<testsuites><testsuite><testcase name="executed"/></testsuite></testsuites>')
        return 0
    monkeypatch.setattr(checks.subprocess, 'call', run)
    monkeypatch.setattr(checks.sys, 'argv', ['check_project.py', '--only', 'backend'])
    assert checks.main() == 0
    assert '--ignore=tests/test_current_formal_runtime.py' in seen[0]
    monkeypatch.setattr(checks.sys, 'argv', ['check_project.py', '--only', 'backend', '--smoke'])
    assert checks.main() == 0
    assert all('tests/test_' + name + '.py' in seen[1] for name in (
        'home_plan_evidence_conditions', 'full_request_preflight', 'saved_workout_correction'))


@pytest.mark.parametrize("events", [["create"], ["drop"], ["create", "create", "drop", "drop"], []])
def test_pg_audit_rejects_missing_or_repeated_lifecycle(tmp_path, events):
    path = tmp_path / "audit.jsonl"
    path.write_text(''.join(json.dumps({"database": "fitzen_test_" + "a" * 32,
                    "event": event}) + '\n' for event in events))
    with pytest.raises(AssertionError):
        checks.pg_lifecycle(path)


def test_pg_audit_accepts_only_exact_generated_database_pairs(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text(''.join(json.dumps({"database": "fitzen_test_" + "b" * 32,
                    "event": event}) + '\n' for event in ("create", "drop")))
    assert checks.pg_lifecycle(path) == {"created": 1, "dropped": 1, "remaining": 0}
    path.write_text(path.read_text().replace("fitzen_test_" + "b" * 32, "postgres"))
    with pytest.raises(AssertionError):
        checks.pg_lifecycle(path)


def test_check_preserves_failure_and_clears_inherited_optins(monkeypatch, tmp_path):
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(checks.sys, "argv", ["check_project.py", "--only", "backend"])
    monkeypatch.setenv("DATABASE_URL", "must-not-be-used")
    monkeypatch.setenv("FITZEN_BROWSER_E2E", "1")
    monkeypatch.setenv("FITZEN_C3_IDENTITY_REPORT", str(tmp_path / "inherited-report.jsonl"))
    def fail(command, *, cwd, env, stdout, stderr):
        assert env["DATABASE_URL"] == ""
        assert "FITZEN_BROWSER_E2E" not in env and "FITZEN_C3_IDENTITY_REPORT" not in env
        return 7
    monkeypatch.setattr(checks.subprocess, "call", fail)
    assert checks.main() == 7
    reports = list(tmp_path.glob("evidence/project-checks/*/summary.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["exit_code"] == 7


def test_check_rejects_success_without_executed_results(monkeypatch, tmp_path):
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(checks.sys, "argv", ["check_project.py", "--only", "backend"])
    monkeypatch.setattr(checks.subprocess, "call", lambda *args, **kwargs: 0)
    assert checks.main() == 1


def test_frontend_failure_survives_successful_types_and_build(monkeypatch, tmp_path):
    tests = tmp_path / "frontend/tests"
    tests.mkdir(parents=True)
    (tests / "sample.test.mjs").write_text("")
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(checks.sys, "argv", ["check_project.py", "--only", "frontend"])
    monkeypatch.setattr(checks.shutil, "which", lambda _: "node")
    codes = iter((2, 0, 0))
    monkeypatch.setattr(checks.subprocess, "call", lambda *args, **kwargs: next(codes))
    assert checks.main() == 2
    report = json.loads(next(tmp_path.glob("evidence/project-checks/*/summary.json")).read_text())
    assert [step["exit_code"] for step in report["steps"]] == [2, 0, 0]
