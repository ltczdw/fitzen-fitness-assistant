"""Run evidence tests; PostgreSQL child settings always point to a new disposable database.

DATABASE_URL must be explicit for --postgres-integration. Only the CREATE/DROP
helper connects to the postgres admin database. Provider values are forced empty.
No connection URL or provider value is written to the result metadata.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "tests"))
from run_tests_safely import DISABLED, BOOL_OFF  # script directory is on sys.path


def junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {k: sum(int(s.get(k, 0)) for s in suites)
              for k in ("tests", "failures", "errors", "skipped")}
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    counts["testcase_elements"] = len(list(root.iter("testcase")))
    cases = list(root.iter("testcase"))
    counts["testcase_failures"] = sum(c.find("failure") is not None for c in cases)
    counts["testcase_errors"] = sum(c.find("error") is not None for c in cases)
    counts["testcase_skipped"] = sum(c.find("skipped") is not None for c in cases)
    counts["testcase_passed"] = len(cases) - sum(counts[k] for k in
        ("testcase_failures", "testcase_errors", "testcase_skipped"))
    counts["extra_suite_tests"] = counts["tests"] - len(cases)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres-integration", action="store_true")
    parser.add_argument("--prepare-pgvector", action="store_true",
                        help="install vector only in this runner's newly created test database")
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--controlled-background", action="store_true")
    parser.add_argument("--evidence-directory", default="evidence/quality-continuation")
    args, pytest_args = parser.parse_known_args()
    assert not args.prepare_pgvector or args.postgres_integration, "pgvector preparation requires isolated PostgreSQL"
    assert args.label.replace("-", "").replace("_", "").isalnum(), "simple evidence label only"
    evidence = (ROOT / args.evidence_directory).resolve()
    assert evidence.is_relative_to(ROOT), "evidence must remain inside this test workspace"
    evidence.mkdir(parents=True, exist_ok=True)
    assert not (evidence / (args.label + ".json")).exists(), "do not overwrite completed evidence"
    os.environ.update({name: "" for name in DISABLED})
    os.environ.update(BOOL_OFF)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    temporary = None
    child_env = dict(os.environ)
    if args.controlled_background:
        child_env["FITZEN_CONTROLLED_BACKGROUND"] = "1"
        child_env["PYTHONPATH"] = str(BACKEND / "tests") + os.pathsep + child_env.get("PYTHONPATH", "")
        pytest_args = ["-p", "quality_controlled_background", *pytest_args]
    xml = evidence / (args.label + ".xml")
    command = [sys.executable, str(BACKEND / "scripts" / "run_tests_safely.py")]
    metadata = {"label": args.label, "started_utc": datetime.now(timezone.utc).isoformat(),
                "postgres_integration": args.postgres_integration,
                "providers_forced_empty": True, "background_launcher": False,
                "controlled_background_fixture": args.controlled_background,
                "pgvector_prepared": False,
                "verification_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pytest_args": pytest_args, "outer_database_dropped": None,
                "outer_database_created": False,
                "source_sha256": {}}
    metadata["business_source_sha256"] = {}
    for rel in ("backend/app/services/agent/workout_extractor.py",
                "backend/app/services/agent/multi_agent_graph.py",
                "backend/app/db/repositories.py", "backend/app/db/workout_session_repository.py",
                "backend/app/db/food_log_mutations.py", "backend/app/db/storage.py",
                "backend/app/db/session.py", "backend/app/db/workout_session_models.py",
                "backend/app/schemas/workout_session.py", "backend/app/api/workouts.py",
                "backend/app/api/workout_sessions_api.py",
                "frontend/src/WorkoutSessionManager.vue", "frontend/src/fitzen/NutritionEditModal.vue"):
        metadata["business_source_sha256"][rel] = hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    metadata["execution_directory"] = str(ROOT)
    meta_path = evidence / (args.label + ".json")
    def save_metadata():
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for p in list((BACKEND / "tests").glob("*.py")) + list((ROOT / "frontend/tests").glob("*.mjs")):
        metadata["source_sha256"][str(p.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        if args.postgres_integration:
            if not os.environ.get("DATABASE_URL", "").startswith("postgresql"):
                # Non-production test configuration travels in-process; it is never
                # printed, exported, or passed through shell command substitution.
                from app.core.config import settings
                candidate = str(settings.database_url or "")
                assert candidate.lower().startswith("postgresql"), "explicit test PostgreSQL config required"
                os.environ["DATABASE_URL"] = candidate
                metadata["database_url_source"] = "settings"
            else:
                metadata["database_url_source"] = "environment"
            assert os.environ.get("DATABASE_URL", "").startswith("postgresql"), "explicit test PostgreSQL config required"
            from quality_isolation import TempPGDatabase
            # Audit every temporary database lifecycle (outer and inner) by exact name.
            audit_path = str(evidence / (args.label + "-pg-database-audit.jsonl"))
            os.environ["FITZEN_PG_DATABASE_AUDIT"] = audit_path
            child_env["FITZEN_PG_DATABASE_AUDIT"] = audit_path
            metadata["pg_database_audit"] = audit_path
            temporary = TempPGDatabase()
            temporary.create()
            metadata["outer_database_created"] = True
            child_env["DATABASE_URL"] = temporary.url
            metadata["outer_database"] = temporary.name
            metadata["outer_database_dropped"] = False
            save_metadata()  # Retain exact ownership evidence even if this runner is interrupted.
            if args.prepare_pgvector:
                from quality_isolation import checked_engine
                from sqlalchemy import text
                preparation = checked_engine(temporary.url)
                try:
                    with preparation.begin() as connection:
                        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"))
                    metadata["pgvector_prepared"] = True
                finally:
                    preparation.dispose()
                save_metadata()
            command.append("--postgres-integration")
        else:
            # Even tests that use settings directly see in-memory SQLite, never a live default.
            child_env["DATABASE_URL"] = "sqlite://"
        command.extend([*pytest_args, "--basetemp=" + str(evidence / ("tmp-" + args.label)),
                        "--junitxml=" + str(xml), "-o", "junit_family=legacy"])
        metadata["command"] = [str(Path(c).relative_to(ROOT)) if c.startswith(str(ROOT)) else c for c in command]
        log_path = evidence / (args.label + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=BACKEND, env=child_env, stdout=log, stderr=subprocess.STDOUT)
            metadata["child_pid"] = child.pid
            save_metadata()
            try:
                code = child.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                metadata["timed_out"] = True
                if os.name == "nt":
                    # Kill only the subprocess tree launched by this runner, never an existing service.
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    child.kill()
                child.wait()
                code = 124
        metadata["exit_code"] = code
        metadata["junit"] = junit_counts(xml) if xml.exists() else None
        summary = re.findall(r"(\d+) subtests (passed|failed|skipped)", log_path.read_text(encoding="utf-8"))
        metadata["subtests_terminal"] = {status: int(count) for count, status in summary}
        print(json.dumps({"label": args.label, "exit_code": code, "junit": metadata["junit"]}))
        return code
    except Exception as exc:
        metadata["exit_code"] = 1
        metadata["runner_error_type"] = type(exc).__name__
        raise
    finally:
        try:
            if temporary is not None:
                temporary.drop()
                if metadata["outer_database_created"]:
                    metadata["outer_database_dropped"] = not temporary.created
        finally:
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save_metadata()


if __name__ == "__main__":
    raise SystemExit(main())
