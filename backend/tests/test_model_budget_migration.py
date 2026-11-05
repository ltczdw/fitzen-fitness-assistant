"""Exercise the explicit migration CLI against a disposable SQLite DB only."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


def test_migration_preview_is_read_only_and_apply_only_adds_three_tables(tmp_path):
    database = tmp_path / 'additive-only.db'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE existing_sentinel (value TEXT)')
        connection.execute("INSERT INTO existing_sentinel VALUES ('synthetic-kept')")

    env = dict(os.environ)
    env.update({name: '' for name in ('LLM_API_KEY', 'RETRIEVAL_EMBEDDING_API_KEY',
                                    'RERANK_API_KEY', 'LANGSMITH_API_KEY', 'MCP_URL')})
    env.update(DATABASE_URL='sqlite:///' + database.as_posix(), MEMORY_BACKGROUND_ENABLED='false',
               MEMORY_SUMMARY_ENABLED='false', LANGSMITH_TRACING='false', REDIS_ENABLED='false')
    backend = Path(__file__).resolve().parents[1]

    def run(*args):
        return subprocess.run([sys.executable, 'scripts/migrate_model_budgets.py', *args],
                              cwd=backend, env=env, capture_output=True, text=True,
                              encoding='utf-8', timeout=15)

    def snapshot():
        with sqlite3.connect(database) as connection:
            tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            values = connection.execute('SELECT value FROM existing_sentinel').fetchall()
            return tables, values

    preview = run()
    assert preview.returncode == 1
    assert "'applied': False" in preview.stdout
    assert snapshot() == ({'existing_sentinel'}, [('synthetic-kept',)])
    expected = {'existing_sentinel', 'model_quotas', 'model_call_ledger', 'model_result_cache'}
    for _ in range(2):
        applied = run('--apply')
        assert applied.returncode == 0, applied.stderr
        assert "'ready': True" in applied.stdout
        assert snapshot() == (expected, [('synthetic-kept',)])
    assert run().returncode == 0
