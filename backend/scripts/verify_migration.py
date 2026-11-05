"""Verify that init_db() can build a real PostgreSQL schema from scratch.

Creates a throwaway database, runs the application initialization twice against
it (fresh build, then idempotent re-run after a knowledge write), checks the
tables, columns, indexes, triggers and revision semantics that the runtime
depends on, then drops the throwaway database again. The application database
in DATABASE_URL is never modified: only `CREATE DATABASE`/`DROP DATABASE` of a
`fitzen_migration_check_*` name is issued against the server.

Usage (from backend/):
    .venv/Scripts/python.exe scripts/verify_migration.py
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECTED_TABLES = {
    "answer_feedback",
    "chat_runs",
    "failure_cases",
    "knowledge_chunks",
    "knowledge_documents",
    "knowledge_index_versions",
    "memory_outbox",
    "memory_turns",
    "nutrition_logs",
    "review_audits",
    "training_plans",
    "user_facts",
    "user_profiles",
    "wiki_aliases",
    "wiki_evidence",
    "wiki_facts",
    "wiki_links",
    "wiki_pages",
    "wiki_revisions",
    "wiki_sections",
    "workout_records",
    "workout_session_exercises",
    "workout_session_sets",
    "workout_sessions",
}
EXPECTED_KNOWLEDGE_COLUMNS = {
    "document_id",
    "parent_id",
    "chunk_index",
    "chunk_count",
    "content_hash",
    "metadata_json",
    "semantic_embedding",
    "embedding_version",
}
EXPECTED_TRIGGER_TABLES = {"knowledge_chunks", "knowledge_documents"}
REVISION_TABLE = "knowledge_index_versions"


def child_main(out_path: str) -> int:
    """Runs inside the throwaway database; collects one JSON report."""
    from sqlalchemy import create_engine, text
    from app.db import session as db
    from app.db.session import init_db

    first = init_db()
    engine = db.engine
    report = {"first_init": first}
    with engine.begin() as connection:
        if not first:
            raise SystemExit("init_db failed on a fresh database; see logged exception")
        connection.execute(text("INSERT INTO knowledge_documents (document_id, title, content, source, content_hash) VALUES ('verify-migration', 'verify', 'verify', 'verify', 'verify-migration-hash')"))
    with engine.connect() as connection:
        report["revision_after_write"] = connection.scalar(text(f"SELECT revision FROM {REVISION_TABLE} WHERE corpus='knowledge'"))
    report["second_init"] = init_db()
    with engine.connect() as connection:
        report["revision_after_second_init"] = connection.scalar(text(f"SELECT revision FROM {REVISION_TABLE} WHERE corpus='knowledge'"))
        report["tables"] = sorted(r[0] for r in connection.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'")))
        report["knowledge_columns"] = sorted(r[0] for r in connection.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='knowledge_chunks'")))
        # information_schema.triggers never reports TRUNCATE, so read the real definitions.
        report["trigger_definitions"] = sorted(
            r[0] for r in connection.execute(text("SELECT pg_get_triggerdef(oid) FROM pg_trigger WHERE NOT tgisinternal"))
        )
        report["functions"] = sorted(r[0] for r in connection.execute(text("SELECT proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='public'")))
        report["indexes"] = sorted(r[0] for r in connection.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname='public'")))
        report["extensions"] = sorted(r[0] for r in connection.execute(text("SELECT extname FROM pg_extension")))
        report["chunk_count"] = connection.scalar(text("SELECT count(*) FROM knowledge_chunks"))
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child_main(args.out)

    from sqlalchemy import create_engine, text
    from app.core.config import settings

    if not settings.database_url.startswith("postgresql"):
        raise SystemExit("This verification requires a PostgreSQL DATABASE_URL.")
    admin = create_engine(settings.database_url, isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 5})
    schema = "fitzen_migration_check_" + uuid4().hex[:10]
    out_path = Path(os.environ.get("TEMP", ".")) / (schema + ".json")
    scratch_url = settings.database_url.rsplit("/", 1)[0] + "/" + schema
    failures: list[str] = []
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{schema}"'))
        print(f"created throwaway database {schema}")
        env = dict(os.environ, DATABASE_URL=scratch_url)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", "--out", str(out_path)],
            env=env, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            print(completed.stdout, completed.stderr)
            raise SystemExit("child run failed against the throwaway database")
        report = json.loads(out_path.read_text(encoding="utf-8"))

        def check(label: str, condition: bool, detail: object = ""):
            print(("PASS " if condition else "FAIL ") + label + ("" if condition else f"  <- {detail}"))
            if not condition:
                failures.append(label)

        check("fresh init_db succeeds", report["first_init"] is True)
        missing = EXPECTED_TABLES.difference(report["tables"])
        check("all application tables exist", not missing, missing)
        check("knowledge migration columns exist", EXPECTED_KNOWLEDGE_COLUMNS.issubset(report["knowledge_columns"]), sorted(EXPECTED_KNOWLEDGE_COLUMNS.difference(report["knowledge_columns"])))
        check("pgvector extension installed", "vector" in report["extensions"], report["extensions"])
        check("hnsw indexes installed", any("hnsw" in name for name in report["indexes"]), [i for i in report["indexes"] if "ix_" in i])
        check("knowledge revision row exists", isinstance(report["revision_after_write"], int), report["revision_after_write"])
        check("write bumps revision", report["revision_after_write"] > 0, report["revision_after_write"])
        check("re-running init_db succeeds", report["second_init"] is True)
        check("re-running init_db preserves revision", report["revision_after_second_init"] == report["revision_after_write"], (report["revision_after_write"], report["revision_after_second_init"]))
        triggers = report["trigger_definitions"]
        check(
            "revision triggers cover insert/update/delete/truncate per statement",
            len(triggers) == len(EXPECTED_TRIGGER_TABLES)
            and all(f"ON public.{table}" in definition for table in EXPECTED_TRIGGER_TABLES for definition in [next(t for t in triggers if f"ON public.{table}" in t)])
            and all("INSERT" in t and "UPDATE" in t and "DELETE" in t and "TRUNCATE" in t and "FOR EACH STATEMENT" in t for t in triggers),
            triggers,
        )
        check("revision trigger function exists", "bump_knowledge_index_revision" in report["functions"], report["functions"])
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{schema}" WITH (FORCE)'))
        admin.dispose()
        out_path.unlink(missing_ok=True)
        print(f"dropped throwaway database {schema}")
    if failures:
        print(f"{len(failures)} check(s) failed: " + ", ".join(failures))
        return 1
    print("migration verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
