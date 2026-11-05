import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import event, text

from app.core.config import settings
from app.db import session as db
from app.services.agent.capabilities import chat_record_capability, probe_chat_record_storage


@pytest.mark.parametrize('reason', ['react_write_disabled', 'non_compact_memory_backend', 'database_unavailable', 'llm_not_configured'])
def test_capability_reports_specific_closed_gate_without_storage_write(compact_db, monkeypatch, reason):
    monkeypatch.setattr(settings, 'react_write_enabled', reason != 'react_write_disabled')
    if reason == 'non_compact_memory_backend': monkeypatch.setattr(settings, 'memory_read_backend', 'redis')
    if reason == 'database_unavailable': monkeypatch.setattr(db, 'DB_READY', False)
    result = chat_record_capability(llm_configured=reason != 'llm_not_configured')
    assert result == {'available': False, 'reason': reason}


def test_readiness_probe_is_readonly_and_never_creates_quota_rows(compact_db, monkeypatch):
    monkeypatch.setattr(settings, 'react_write_enabled', True)
    statements = []
    engine = db.engine
    def observe(conn, cursor, statement, *args): statements.append(statement.strip().upper())
    event.listen(engine, 'before_cursor_execute', observe)
    try:
        assert chat_record_capability(llm_configured=True) == {'available': True, 'reason': 'available'}
        assert not any(s.startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP')) for s in statements)
    finally:
        event.remove(engine, 'before_cursor_execute', observe)
    with compact_db() as session:
        assert session.execute(text('SELECT COUNT(*) FROM model_quotas')).scalar() == 0


@pytest.mark.parametrize('table,reason', [('agent_execution_events', 'execution_ledger_unavailable'), ('model_result_cache', 'model_budget_storage_unavailable')])
def test_missing_dependency_is_unavailable_and_not_repaired(compact_db, table, reason):
    with db.engine.begin() as connection: connection.execute(text('DROP TABLE ' + table))
    assert probe_chat_record_storage() == (False, reason)
    from sqlalchemy import inspect
    assert not inspect(db.engine).has_table(table)


@pytest.mark.parametrize('enabled', [True, False])
def test_real_health_publishes_capability_without_model_or_record_calls(compact_db, monkeypatch, enabled):
    from app.main import app
    from app.api import chat, health
    monkeypatch.setattr(settings, 'react_write_enabled', enabled)
    monkeypatch.setattr(chat.agent.graph.llm, 'status', lambda: {'configured': True, 'circuit': {'state': 'closed'}})
    model = AsyncMock(side_effect=AssertionError('Health must not call inference'))
    monkeypatch.setattr(chat.agent.graph.llm, 'complete', model)
    monkeypatch.setattr(health.memory_service, 'initialize', AsyncMock(return_value=False))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get('/api/health')
            assert response.status_code == 200
            assert response.json()['capabilities']['chat_record'] == {
                'available': enabled, 'reason': 'available' if enabled else 'react_write_disabled'}
    asyncio.run(run())
    model.assert_not_called()
