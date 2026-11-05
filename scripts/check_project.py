"""One local/CI check entry. No dependency installs or production configuration edits."""
import argparse
from collections import Counter
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
# These historical formal matrices require their sealed runner, identity and
# approval binding. Keep that separate contract; don't rename CI as formal15.
FORMAL = (
    "test_react_phase_c1_ledger.py", "test_react_phase_c2_1_workout.py",
    "test_react_phase_c2_2_nutrition.py", "test_react_phase_c2_3_runtime.py",
    "test_c1_c2_current_binding.py", "test_current_formal_entry.py",
    "test_current_formal_legacy_runtime.py", "test_current_formal_permissions.py",
    "test_current_formal_runtime.py",
)


def junit_counts(path):
    cases = ElementTree.parse(path).findall(".//testcase")
    return {"passed": sum(not any(c.find(tag) is not None for tag in
            ("failure", "error", "skipped")) for c in cases),
            **{tag: sum(c.find(tag) is not None for c in cases)
               for tag in ("failure", "error", "skipped")}}


def pg_lifecycle(path):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events and all(re.fullmatch(r"fitzen_test_[0-9a-f]{32}", row["database"])
                          and row["event"] in ("create", "drop") for row in events)
    created = Counter(row["database"] for row in events if row["event"] == "create")
    dropped = Counter(row["database"] for row in events if row["event"] == "drop")
    assert created == dropped and all(count == 1 for count in created.values())
    return {"created": sum(created.values()), "dropped": sum(dropped.values()), "remaining": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "backend", "frontend"), default="all")
    parser.add_argument("--smoke", action="store_true",
                        help="run the curated core smoke suite instead of the full local matrix")
    parser.add_argument("--postgres-integration", action="store_true",
                        help="run the isolated PG API/read matrix instead of regular backend checks")
    args = parser.parse_args()
    if args.smoke and args.postgres_integration:
        parser.error("--smoke and --postgres-integration are separate entries")
    if args.postgres_integration and args.only == "frontend":
        parser.error("PostgreSQL checks require backend")
    evidence = ROOT / "evidence" / "project-checks" / ("run-" + uuid4().hex)
    evidence.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # The default check cannot inherit opt-ins, external evidence destinations,
    # or a live-browser fixture from the caller. Safe launcher clears providers.
    for name in list(env):
        if name.startswith(("FITZEN_C1_", "FITZEN_C2_", "FITZEN_C3_")) or name in (
                "FITZEN_BROWSER_E2E", "FITZEN_E2E_RUN", "FITZEN_PG_DATABASE_AUDIT"):
            env.pop(name)
    if not args.postgres_integration:
        env["DATABASE_URL"] = ""
    summary = {"postgres_integration": args.postgres_integration, "steps": []}

    def run(name, command, cwd):
        with (evidence / (name + ".log")).open("x", encoding="utf-8") as log:
            try:
                code = subprocess.call(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            except OSError as exc:
                log.write(type(exc).__name__ + ": required runtime could not be started\n")
                code = 127
        summary["steps"].append({"name": name, "command": command,
                                 "exit_code": code, "log": name + ".log"})
        print(name + ": exit " + str(code), flush=True)
        return code

    result = 0
    if args.only != "frontend":
        xml = evidence / "backend.xml"
        command = [sys.executable, str(ROOT / "backend/scripts/run_tests_safely.py")]
        if args.smoke:
            command += [
                "tests/test_home_plan_evidence_conditions.py",
                "tests/test_saved_workout_correction.py",
                "tests/test_full_request_preflight.py",
                "tests/test_compound_workout_date_receipt.py",
                "tests/test_failure_response_matrix.py",
                "tests/test_plan_default_gym_optional_time.py",
                "tests/test_plan_preview_followup_limits.py",
                "tests/test_structured_json_syntax_retry.py",
                "tests/test_preview_and_nutrition_delivery.py",
                "tests/test_nutrition_and_preview_execution.py",
                "tests/test_training_timing.py",
                "tests/test_plan_preview_state.py",
                "tests/test_dialogue_continuity.py",
                "tests/test_multi_agent_graph.py",
            ]
        elif args.postgres_integration:
            env["FITZEN_PG_DATABASE_AUDIT"] = str(evidence / "pg-databases.jsonl")
            command += ["--postgres-integration", "tests/test_http_safety_boundaries.py",
                        "tests/test_manual_data_consistency.py", "tests/test_food_record_timestamp.py",
                        "tests/test_clean_startup.py",
                        "-k", "postgres"]
        else:
            command += ["tests", "../frontend/tests/test_chat_wire_contract.py",
                        "../scripts/tests/test_project_checks.py",
                        "../scripts/tests/test_release_export.py",
                        *["--ignore=tests/" + name for name in FORMAL]]
        command += ["-p", "tests.react_phase_c2_3_support", "-p", "no:cacheprovider",
                    "--tb=short", "--basetemp", str(evidence / "pytest"), "--junitxml", str(xml)]
        result = run("backend", command, ROOT / "backend") or result
        if xml.is_file():
            summary["backend_junit"] = junit_counts(xml)
            if (summary["backend_junit"]["failure"] or summary["backend_junit"]["error"]
                    or not summary["backend_junit"]["passed"]):
                result = result or 1
            if args.postgres_integration and (summary["backend_junit"]["skipped"] or
                                              not summary["backend_junit"]["passed"]):
                print("PostgreSQL check requires executed tests and zero skips", flush=True)
                result = result or 1
        else:
            result = result or 1
        if args.postgres_integration:
            try:
                summary["pg_lifecycle"] = pg_lifecycle(evidence / "pg-databases.jsonl")
            except (OSError, ValueError, KeyError, AssertionError):
                summary["pg_lifecycle"] = {"verified": False}
                print("PostgreSQL lifecycle audit missing or unpaired", flush=True)
                result = result or 1
    if args.only != "backend":
        node = shutil.which("node")
        if node is None:
            print("Node.js is required", flush=True)
            result = result or 127
        else:
            tests = sorted(str(p.relative_to(ROOT / "frontend")) for p in
                           (ROOT / "frontend/tests").glob("*.test.mjs"))
            if not tests:
                result = result or 1
            else:
                result = run("frontend-tests", [node, "--test", *tests], ROOT / "frontend") or result
            result = run("frontend-types", [node, "node_modules/vue-tsc/bin/vue-tsc.js", "-b"],
                         ROOT / "frontend") or result
            result = run("frontend-build", [node, "tests/isolated-build.mjs",
                         str(evidence), "frontend-dist"], ROOT / "frontend") or result
    summary["exit_code"] = result
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Evidence: " + str(evidence), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
