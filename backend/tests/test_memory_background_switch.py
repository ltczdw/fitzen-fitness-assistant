import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.core.config import settings, Settings
from app.services.memory import compact
from app.services.memory.compact import CompactMemory
from app.services.memory.deepseek_summary import DeepSeekSummaryClient
from app.services.memory.redis_memory import RedisConversationMemory
from evals.v2.safety import deny_network


def test_default_background_processing_is_enabled_and_opt_out_preserved():
    # 2026-10-02 用户决定：后台记忆处理默认开启。字段默认值断言不受测试
    # 启动器强制环境变量影响；显式 opt-out（环境变量或显式参数）仍然生效。
    assert Settings.model_fields['memory_background_enabled'].default is True
    assert Settings(_env_file=None, memory_background_enabled=False).memory_background_enabled is False


def test_disabled_compact_start_does_not_read_db_backfill_or_create_worker(monkeypatch):
    monkeypatch.setattr(settings,'memory_background_enabled',False)
    monkeypatch.setattr(settings,'memory_summary_enabled',True)
    monkeypatch.setattr(compact.db,'DB_READY',True)
    monkeypatch.setattr(settings,'memory_read_backend','compact_v1')
    epoch=Mock(side_effect=AssertionError('paused worker read database'))
    monkeypatch.setattr(compact,'epoch_value',epoch)
    memory=CompactMemory(SimpleNamespace())
    with deny_network():asyncio.run(memory.start())
    assert memory.task is None
    assert memory.status()['background_enabled'] is False
    assert memory.status()['background_worker_running'] is False
    epoch.assert_not_called()


def test_disabled_consume_never_claims_or_changes_queue_attempts(monkeypatch):
    monkeypatch.setattr(settings,'memory_background_enabled',False)
    claim=Mock(side_effect=AssertionError('paused worker claimed queued task'))
    monkeypatch.setattr(compact,'claim_event',claim)
    with deny_network():assert asyncio.run(compact.consume_one()) is False
    claim.assert_not_called()


def test_summary_disabled_also_blocks_worker_when_background_opted_in(monkeypatch):
    monkeypatch.setattr(settings,'memory_background_enabled',True)
    monkeypatch.setattr(settings,'memory_summary_enabled',False)
    claim=Mock(side_effect=AssertionError('disabled summary still claimed work'))
    monkeypatch.setattr(compact,'claim_event',claim)
    with deny_network():assert asyncio.run(compact.consume_one()) is False
    claim.assert_not_called()


def test_disabled_legacy_schedule_and_direct_consume_do_not_connect(monkeypatch):
    monkeypatch.setattr(settings,'memory_background_enabled',False)
    memory=RedisConversationMemory(enabled=True)
    initialize=AsyncMock(side_effect=AssertionError('paused summary connected Redis'))
    monkeypatch.setattr(memory,'initialize',initialize)
    async def check():
        memory._schedule_summary('test-user','test-conversation')
        assert memory._summary_tasks=={}
        assert await memory._summarize_pending('test-user','test-conversation') is False
        await memory.close()
    with deny_network():asyncio.run(check())
    initialize.assert_not_awaited()


def test_disabled_summary_probe_never_checks_or_calls_provider(monkeypatch):
    monkeypatch.setattr(settings,'memory_background_enabled',False)
    llm=SimpleNamespace(configured=Mock(side_effect=AssertionError('provider checked')),complete=AsyncMock())
    client=DeepSeekSummaryClient(llm)
    with deny_network():assert asyncio.run(client.probe(force=True)) is False
    assert client.status()['enabled'] is False
    llm.configured.assert_not_called()
    llm.complete.assert_not_awaited()


def test_explicit_opt_in_keeps_worker_consumption_available(monkeypatch):
    monkeypatch.setattr(settings,'memory_background_enabled',True)
    monkeypatch.setattr(settings,'memory_summary_enabled',True)
    claim=Mock(return_value=None)
    monkeypatch.setattr(compact,'claim_event',claim)
    with deny_network():assert asyncio.run(compact.consume_one()) is False
    claim.assert_called_once()
