"""Run the installed interpreter's real loopback API with a fresh isolated DB."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest
from sqlalchemy import text

from quality_isolation import isolated_database


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
def test_real_cli_startup_health_assets_and_manual_record(monkeypatch, tmp_path, dialect):
    with isolated_database(monkeypatch, tmp_path, dialect) as engine:
        env = dict(os.environ)
        env.update({name: "" for name in ("LLM_API_KEY", "RETRIEVAL_EMBEDDING_API_KEY",
                   "RERANK_API_KEY", "LANGSMITH_API_KEY", "MCP_URL")})
        env.update({"DATABASE_URL": engine.url.render_as_string(hide_password=False),
                    "REDIS_ENABLED": "false", "LANGSMITH_TRACING": "false",
                    "MEMORY_BACKGROUND_ENABLED": "false", "MEMORY_SUMMARY_ENABLED": "false",
                    "REACT_READONLY_ENABLED": "false", "REACT_WRITE_ENABLED": "false"})
        env.pop("PYTHONPATH", None)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        command = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                   "--port", str(port), "--log-level", "warning"]
        with (tmp_path / "api.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=Path(__file__).parents[1], env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False) as client:
                    deadline = time.monotonic() + 20
                    while time.monotonic() < deadline:
                        assert process.poll() is None, "isolated API exited before readiness"
                        try:
                            ready = client.get("/api/health")
                            if ready.status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(.1)
                    else:
                        pytest.fail("isolated API did not become ready")
                    assert ready.json()["ready"] is True
                    assert ready.json()["components"]["llm"]["configured"] is False
                    assert ready.json()["components"]["redis"]["status"] == "disabled"
                    asset = client.get("/api/quality")
                    assert asset.status_code == 200 and "text/html" in asset.headers["content-type"]
                    urgent = client.post("/api/chat", json={"message": "现在胸痛，怎么缓解胸痛？"})
                    # No model is configured in this real subprocess. Dangerous
                    # wording must not be promoted to urgent locally; the shared
                    # safety judgment is unavailable and the request fails closed
                    # with zero business/model-ledger writes.
                    assert urgent.status_code == 200
                    assert urgent.json()["safety_level"] == "unknown"
                    assert urgent.json()["intent"] == "execution_limited"
                    assert "安全语义判断" in urgent.json()["answer"]
                    record = {"user_id": "startup-synthetic", "exercise": "卧推", "sets": 4,
                              "reps": 8, "weight_kg": 60, "idempotency_key": "startup-synthetic"}
                    saved = client.post("/api/workouts", json=record)
                    assert saved.status_code == 200
                    history = client.get("/api/workouts", params={"user_id": "startup-synthetic"})
                    assert history.status_code == 200
                    assert len(history.json()) == 1 and history.json()[0]["id"] == saved.json()["id"]
                    with engine.connect() as connection:
                        assert connection.execute(text("SELECT COUNT(*) FROM model_call_ledger")).scalar_one() == 0
            finally:
                # Windows venv python.exe is a launcher; terminating only that
                # parent can leave its interpreter child listening. Stop only
                # this test's known PID tree while the parent is still alive.
                if sys.platform == "win32" and process.poll() is None:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdout=log, stderr=subprocess.STDOUT, check=False,
                            creationflags=subprocess.CREATE_NO_WINDOW)
                elif process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
        assert process.poll() is not None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) != 0:
                    break
            time.sleep(.1)
        else:
            pytest.fail("test API still listening")
