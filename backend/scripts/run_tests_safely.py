r"""Run the backend test suite with every provider credential forced empty.

Windows PowerShell 5.1 removes an environment variable when it is assigned an
empty string (``$env:LLM_API_KEY=''``), which lets ``backend/.env`` supply the
real keys again. This launcher disables application provider configuration in
the child process and verifies effective settings before running pytest.
It is not a network sandbox: tests must still mock all external clients.

Usage (from backend/):
    .venv\Scripts\python.exe scripts\run_tests_safely.py [--check-config] [pytest args]
    .venv\Scripts\python.exe scripts\run_tests_safely.py --postgres-integration [pytest args]

Inherited FITZEN_DB_TESTS / FITZEN_QUALITY_DB values are always removed so a
parent shell cannot silently enable or disable the PostgreSQL suites; only
--postgres-integration opts them in, and it verifies the effective child
configuration before pytest starts.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISABLED = (
    "LLM_API_KEY", "RETRIEVAL_EMBEDDING_API_KEY", "RERANK_API_KEY",
    "LANGSMITH_API_KEY", "MCP_URL",
)
BOOL_OFF = {"REDIS_ENABLED": "false", "LANGSMITH_TRACING": "false",
            "MEMORY_BACKGROUND_ENABLED": "false", "MEMORY_SUMMARY_ENABLED": "false"}

CONFIG_PROBE = r"""from app.core.config import settings
import json, os
result={
 'llm':bool(settings.llm_api_key),
 'embedding':bool(settings.retrieval_embedding_api_key),
 'rerank':bool(settings.rerank_api_key),
 'langsmith':bool(settings.langsmith_api_key or settings.langsmith_tracing),
 'mcp':bool(settings.mcp_url),
 'database_configured':bool(settings.database_url),
 'database_postgresql':str(settings.database_url).lower().startswith('postgresql'),
 'postgres_integration':os.environ.get('FITZEN_DB_TESTS')=='1',
 'quality_postgresql':os.environ.get('FITZEN_QUALITY_DB')=='postgres',
}
print(json.dumps(result,sort_keys=True))
raise SystemExit(2 if any(result[name] for name in ('llm','embedding','rerank','langsmith','mcp')) else 0)
"""


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres-integration",action="store_true",
                        help="enable isolated FITZEN_DB_TESTS and FITZEN_QUALITY_DB PostgreSQL tests")
    parser.add_argument("--check-config",action="store_true",
                        help="verify effective child-process provider credentials are disabled, then exit")
    options,pytest_args=parser.parse_known_args()
    env = dict(os.environ)
    env.update({name: "" for name in DISABLED})
    env.update(BOOL_OFF)
    if options.postgres_integration:
        env["FITZEN_DB_TESTS"]="1"
        env["FITZEN_QUALITY_DB"]="postgres"
    else:
        env.pop("FITZEN_DB_TESTS",None)
        env.pop("FITZEN_QUALITY_DB",None)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # Inspect effective Pydantic settings in an actual child process. Print
    # booleans only; never echo secret values. Fail closed if any provider is on.
    probe=subprocess.run([sys.executable,"-c",CONFIG_PROBE],cwd=Path(__file__).resolve().parents[1],
                         env=env,capture_output=True,text=True,encoding="utf-8")
    if probe.stdout.strip():
        print("effective_provider_configuration="+probe.stdout.strip())
    if probe.returncode != 0:
        print("安全配置检查失败；pytest 未启动。")
        return 2
    effective=json.loads(probe.stdout)
    if options.postgres_integration and not (
            effective["database_configured"] and effective["database_postgresql"]
            and effective["postgres_integration"] and effective["quality_postgresql"]):
        print("PostgreSQL opt-in 配置未生效或 DATABASE_URL 不是 PostgreSQL；pytest 未启动。")
        return 2
    if options.check_config:
        print("provider credentials disabled; postgres_integration="+
              str(options.postgres_integration).lower())
        return 0
    command = [sys.executable, "-m", "pytest", "-q", "--disable-warnings", *pytest_args]
    return subprocess.call(command, cwd=Path(__file__).resolve().parents[1], env=env)

if __name__ == "__main__":
    raise SystemExit(main())
