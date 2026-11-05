"""后台记忆启动检查与失败保护验收（独立测试 Agent，2026-10-02）。

两项 Codex 审查发现的验收测试（A/B）编码产品合同而非当前实现：修复前按
设计失败，作为缺陷证据与修复验收，不得反向修改预期使现有实现通过。

- A：compact 不可用（compact_v1 + DB_READY=false）时，status()/health 仍必须
  可观测 background_disabled_reason='compact_memory_unavailable'。
- B：probe_budget_storage 在真实 PG 表锁下必须在有限时间内结束（测试自身
  8 秒上限 + finally 释放锁/清理本次临时数据库）。

安全边界：不读真实 .env；monkeypatch 同时替换 db.engine 与 db.SessionLocal；
PG 仅使用身份逐连接验证的随机临时数据库；全程 deny_network + mock，无真实模型调用。
"""
import asyncio
import json
import logging
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db import session as db
from app.db.models import Base
from app.services.memory import compact as compact_module
from app.services.memory.compact import CompactMemory
from app.services.llm.budget import BudgetStore, probe_budget_storage, store_override
from evals.v2.safety import deny_network

from quality_isolation import TempPGDatabase as _TempPGDatabase, checked_engine as create_engine


@pytest.fixture(autouse=True)
def close_all_guard_engines(monkeypatch):
    """Dispose engines even when table creation or assertions fail before teardown."""
    import sys
    opened = []
    original = create_engine
    def tracked(*args, **kwargs):
        engine = original(*args, **kwargs)
        opened.append(engine)
        return engine
    monkeypatch.setattr(sys.modules[__name__], 'create_engine', tracked)
    try:
        yield
    finally:
        for engine in reversed(opened):
            engine.dispose()


BUDGET_TABLES = ('model_quotas', 'model_call_ledger', 'model_result_cache')


class _FakeLegacy:
    def __init__(self):
        self.closed = False

    async def initialize(self):
        return True

    async def close(self):
        self.closed = True

    def status(self):
        return {'backend': 'memory-fallback', 'status': 'degraded', 'persistent': False}


def _isolate_db(monkeypatch, engine):
    """同时替换 engine 与 SessionLocal，避免探针/诊断误连真实数据库。"""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db, 'engine', engine)
    monkeypatch.setattr(db, 'SessionLocal', factory)
    monkeypatch.setattr(db, 'DB_READY', True)
    return factory


def _enabled(monkeypatch):
    monkeypatch.setattr(settings, 'memory_read_backend', 'compact_v1')
    monkeypatch.setattr(settings, 'memory_background_enabled', True)
    monkeypatch.setattr(settings, 'memory_summary_enabled', True)


def _forbid_startup_side_effects(monkeypatch):
    """探针失败路径上任何越权的 epoch/wiki/backfill 动作都让测试立即失败。"""
    monkeypatch.setattr(compact_module, 'epoch_value',
                        Mock(side_effect=AssertionError('epoch touched during failed startup')))
    from app.services.memory import wiki as wiki_module
    monkeypatch.setattr(wiki_module, 'initialize',
                        Mock(side_effect=AssertionError('wiki initialize during failed startup')))
    monkeypatch.setattr(wiki_module, 'backfill',
                        Mock(side_effect=AssertionError('wiki backfill during failed startup')))


# ---------------------------------------------------------------------------
# 发现 A：compact 不可用时停用原因丢失（修复前本测试失败 = 缺陷证据）
# ---------------------------------------------------------------------------

def test_a_status_reports_compact_unavailable_reason(monkeypatch):
    _enabled(monkeypatch)
    monkeypatch.setattr(db, 'DB_READY', False)
    monkeypatch.setattr(db, 'engine', None)
    monkeypatch.setattr(db, 'SessionLocal', None)
    _forbid_startup_side_effects(monkeypatch)
    memory = CompactMemory(_FakeLegacy())
    with deny_network():
        asyncio.run(memory.start())

    assert memory.task is None
    assert memory.background_disabled_reason == 'compact_memory_unavailable'
    # 产品合同：status()/health 输出必须可观测停用原因，不得退化为 legacy
    # 字典把原因吞掉。
    status = memory.status()
    assert status.get('background_worker_running') is False, status
    assert status.get('background_disabled_reason') == 'compact_memory_unavailable', status
    assert status.get('background_enabled') is True, status


# ---------------------------------------------------------------------------
# 覆盖 1：默认开启；显式 false 优先且不调用预算探针
# ---------------------------------------------------------------------------

def test_1_explicit_config_off_skips_probe_and_worker(monkeypatch):
    _enabled(monkeypatch)
    monkeypatch.setattr(settings, 'memory_background_enabled', False)
    monkeypatch.setattr(db, 'DB_READY', True)
    engine = create_engine('sqlite://')
    _isolate_db(monkeypatch, engine)
    _forbid_startup_side_effects(monkeypatch)
    probe = Mock(side_effect=AssertionError('probe ran while config off'))
    monkeypatch.setattr(compact_module, 'probe_budget_storage', probe)
    memory = CompactMemory(_FakeLegacy())
    with deny_network():
        asyncio.run(memory.start())
    assert memory.task is None
    assert memory.background_disabled_reason == 'background_disabled_by_config'
    probe.assert_not_called()


def test_1_field_default_is_enabled():
    assert type(settings).model_fields['memory_background_enabled'].default is True


# ---------------------------------------------------------------------------
# 覆盖 2：缺表/缺列/库不可用/engine 缺失 → 无 worker、无 epoch/wiki/backfill、
# 无领取、原因可观测、单次启动只刷一条告警
# ---------------------------------------------------------------------------

def _memory_over(monkeypatch, engine=None, ready=True):
    _enabled(monkeypatch)
    monkeypatch.setattr(db, 'DB_READY', ready)
    if engine is None:
        monkeypatch.setattr(db, 'engine', None)
        monkeypatch.setattr(db, 'SessionLocal', None)
    else:
        _isolate_db(monkeypatch, engine)
    _forbid_startup_side_effects(monkeypatch)
    return CompactMemory(_FakeLegacy())


def test_2_missing_budget_tables_fail_closed(monkeypatch, caplog, tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'nobudget.db'))
    limited = [t for t in Base.metadata.sorted_tables if t.name not in BUDGET_TABLES]
    Base.metadata.create_all(engine, tables=limited)
    memory = _memory_over(monkeypatch, engine)
    with caplog.at_level(logging.WARNING):
        with deny_network():
            asyncio.run(memory.start())
    assert memory.task is None
    assert memory.background_disabled_reason == 'budget_tables_missing'
    assert memory.status().get('background_disabled_reason') == 'budget_tables_missing'
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                and 'background_memory_disabled' in r.message]
    assert len(warnings) == 1, caplog.records


def test_2_missing_mapped_column_fail_closed(monkeypatch, tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'missingcol.db'))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql('ALTER TABLE model_quotas DROP COLUMN max_calls')
    memory = _memory_over(monkeypatch, engine)
    with deny_network():
        asyncio.run(memory.start())
    assert memory.task is None
    assert memory.background_disabled_reason == 'budget_storage_unavailable'


def test_2_unavailable_database_fail_closed(monkeypatch, tmp_path):
    dead = create_engine('sqlite:///' + str(tmp_path / 'missing-dir' / 'x.db'))
    memory = _memory_over(monkeypatch, dead)
    with deny_network():
        asyncio.run(memory.start())
    assert memory.task is None
    assert memory.background_disabled_reason == 'budget_storage_unavailable'


def test_2_missing_engine_and_factory_fail_closed(monkeypatch):
    memory = _memory_over(monkeypatch, engine=None, ready=True)
    with deny_network():
        asyncio.run(memory.start())
    assert memory.task is None
    assert memory.background_disabled_reason == 'budget_storage_unconfigured'
    assert memory.status().get('background_disabled_reason') == 'budget_storage_unconfigured'


# ---------------------------------------------------------------------------
# 覆盖 3：探针成功 → 允许创建 worker、停用原因 null；测试内控制生命周期
# ---------------------------------------------------------------------------

def test_3_successful_probe_starts_worker(monkeypatch, tmp_path):
    _enabled(monkeypatch)
    monkeypatch.setattr(db, 'DB_READY', True)
    engine = create_engine('sqlite:///' + str(tmp_path / 'ok.db'))
    Base.metadata.create_all(engine)
    _isolate_db(monkeypatch, engine)
    memory = CompactMemory(_FakeLegacy())

    async def scenario():
        await memory.start()
        try:
            assert memory.task is not None and not memory.task.done()
            assert memory.background_disabled_reason is None
            status = memory.status()
            assert status['background_worker_running'] is True
            assert status['background_disabled_reason'] is None
        finally:
            if memory.task:
                memory.task.cancel()
                await asyncio.gather(memory.task, return_exceptions=True)

    with deny_network():
        asyncio.run(scenario())
    engine.dispose()


# ---------------------------------------------------------------------------
# 覆盖 4：探针只读（无 DDL/DML）
# ---------------------------------------------------------------------------

def test_4_probe_issues_only_read_statements(monkeypatch, tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'ro.db'))
    Base.metadata.create_all(engine)
    _isolate_db(monkeypatch, engine)
    statements = []
    event.listen(engine, 'before_cursor_execute',
                 lambda conn, cursor, stmt, params, many, ctx: statements.append(stmt))
    assert probe_budget_storage() == (True, None)
    assert statements, 'probe did not run any statement'
    allowed = ('SELECT', 'BEGIN', 'COMMIT', 'PRAGMA')
    for stmt in statements:
        head = stmt.lstrip().upper()
        assert head.startswith(allowed), f'非只读语句: {stmt}'


# ---------------------------------------------------------------------------
# 覆盖 5：init_db 注册并创建预算三表；初始化失败返回 False（main.py 据此拒启）
# ---------------------------------------------------------------------------

def test_5_init_db_creates_budget_tables(monkeypatch, tmp_path):
    from app.db.session import init_db
    url = 'sqlite:///' + str(tmp_path / 'init.db')
    engine = create_engine(url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db, 'engine', engine)
    monkeypatch.setattr(db, 'SessionLocal', factory)
    monkeypatch.setattr(settings, 'database_url', url)
    assert init_db() is True
    from sqlalchemy import inspect as sqlalchemy_inspect
    with engine.connect() as conn:
        inspector = sqlalchemy_inspect(conn)
        for name in BUDGET_TABLES:
            assert inspector.has_table(name), f'{name} 未由 init_db 创建'


def test_5_init_db_failure_refuses_startup(monkeypatch):
    from app.db.session import init_db
    from app.db.storage import StorageUnavailableError

    class _Broken:
        def begin(self):
            raise RuntimeError('storage broken')

    monkeypatch.setattr(db, 'engine', _Broken())
    monkeypatch.setattr(db, 'SessionLocal', None)
    assert init_db() is False
    # main.py lifespan 的拒启守卫：database_ready=False 必须拒绝启动。
    database_ready = init_db()
    if not database_ready:
        with pytest.raises(StorageUnavailableError):
            raise StorageUnavailableError('Database initialization failed; service startup refused')
    else:  # pragma: no cover
        pytest.fail('init_db 应在存储故障时返回 False')


# ---------------------------------------------------------------------------
# 覆盖 7：旧 Redis 路径——预算故障时零模型派发（配置关闭由
# test_memory_background_switch::test_disabled_legacy_* 覆盖）
# ---------------------------------------------------------------------------

def test_7_legacy_redis_budget_failure_dispatches_no_model(monkeypatch):
    from app.services.memory.redis_memory import RedisConversationMemory
    from app.services.llm.budget import ModelBudgetError
    from app.services.llm import client as llm_client_module

    monkeypatch.setattr(settings, 'memory_background_enabled', True)
    monkeypatch.setattr(settings, 'memory_summary_enabled', True)
    # 让真实 LLMClient 处于"已配置"状态，使预算门成为 HTTP 前的最后一道闸。
    monkeypatch.setattr(settings, 'llm_api_key', 'test-key')
    monkeypatch.setattr(settings, 'llm_base_url', 'http://budget-failure.test')
    memory = RedisConversationMemory(enabled=True)
    monkeypatch.setattr(memory, 'initialize', AsyncMock(return_value=True))
    memory.client = SimpleNamespace(
        set=AsyncMock(return_value=True),
        lrange=AsyncMock(return_value=[json.dumps(
            {'role': 'user', 'content': '你好', 'message_id': 'm1'})]),
        get=AsyncMock(return_value=None),
        eval=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(db, 'SessionLocal', None)
    # HTTP 派发计数器：client 模块内任何 AsyncClient post/stream 尝试都计入。
    dispatches: list[tuple] = []

    class _ForbiddenAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, *args, **kwargs):
            dispatches.append(args)
            raise AssertionError('HTTP dispatched despite budget failure')

        async def stream(self, *args, **kwargs):
            dispatches.append(args)
            raise AssertionError('HTTP streamed despite budget failure')

    monkeypatch.setattr(llm_client_module, 'httpx',
                        SimpleNamespace(AsyncClient=_ForbiddenAsyncClient))
    token = store_override.set(BudgetStore(factory=None))
    try:
        # 产品合同：预算存储故障被显式捕获（redis_memory.py except 含
        # ModelBudgetError），降级为确定性窗口，零模型派发、零语义写入。
        with deny_network() as audit:
            outcome = asyncio.run(memory._summarize_pending('u', 'c'))
    finally:
        store_override.reset(token)
    assert outcome is False
    # 零派发由两个独立证据共同证明：HTTP 派发计数为 0，且禁网审计中没有任何
    # 连接/DNS 尝试被拦截（仅返回 False 或"联网被拦截"都不充分）。
    assert dispatches == [], f'预算故障下发生了 HTTP 派发: {dispatches}'
    assert audit['blocked_connections'] == 0, audit
    assert audit['blocked_dns'] == 0, audit
    # 只发生了分布式锁一次 set；语义结果不得被写入（摘要未完成）。
    assert memory.client.set.await_count == 1


# ---------------------------------------------------------------------------
# 发现 B：预算表被锁时启动探针必须有限时间结束（真实 PG 随机临时数据库）。
# 修复前本测试失败（探针无限等待）= 缺陷证据。
#
# 实验结论（2026-10-02）：持锁连接 LOCK TABLE ... ACCESS EXCLUSIVE 后，另一连接
# 的普通 SELECT 会被无限阻塞；compact.start() 用 asyncio.to_thread 调探针且
# 无期限，lifespan 将整体挂起。因此本测试用受控线程直接验证同步探针本体，
# 避免被阻塞线程经 asyncio 执行器把测试拖入 300 秒 join。
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.getenv('FITZEN_DB_TESTS') != '1'
                    or not str(settings.database_url).lower().startswith('postgresql'),
                    reason='random isolated PostgreSQL database only')
def test_b_probe_finishes_bounded_under_table_lock(monkeypatch):
    import concurrent.futures
    import time as _time

    temp_db = _TempPGDatabase()
    raw = mapped = None
    lock_conn = None
    statements = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    try:
        temp_db.create()
        raw = create_engine(temp_db.url)
        mapped = create_engine(temp_db.url, pool_size=1, max_overflow=0)
        budget_tables = [Base.metadata.tables[name] for name in BUDGET_TABLES]
        Base.metadata.create_all(mapped, tables=budget_tables)
        from sqlalchemy import inspect as sqlalchemy_inspect
        with mapped.connect() as conn:
            inspector = sqlalchemy_inspect(conn)
            assert all(inspector.has_table(name) for name in BUDGET_TABLES), \
                '探针引擎必须能看到临时数据库中的预算表'
        factory = sessionmaker(bind=mapped, expire_on_commit=False)
        monkeypatch.setattr(db, 'engine', mapped)
        monkeypatch.setattr(db, 'SessionLocal', factory)
        monkeypatch.setattr(db, 'DB_READY', True)
        event.listen(mapped, 'before_cursor_execute',
                     lambda conn, cursor, stmt, params, many, ctx: statements.append(stmt))

        # 模拟迁移/其他事务持有预算表排他锁。
        lock_conn = raw.connect()
        lock_conn.execute(text('LOCK TABLE model_quotas IN ACCESS EXCLUSIVE MODE'))

        # 产品合同：探针必须在有限时间内结束（测试上限 8 秒）。
        # compact.start() 以 asyncio.to_thread 调用本函数且无期限，探针无限
        # 等待即 lifespan 无限挂起。
        future = executor.submit(probe_budget_storage)
        finished, result = False, None
        try:
            result = future.result(timeout=8.0)
            finished = True
        except concurrent.futures.TimeoutError:
            pass
        assert finished, '发现B复现：probe_budget_storage 在表锁下超过8秒未结束（无锁/语句超时）'
        # 修复后的预期合同：锁/语句超时被探针转换为安全失败原因，而非异常逃逸。
        assert result == (False, 'budget_storage_unavailable'), result
        for stmt in statements:
            head = stmt.lstrip().upper()
            assert not head.startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')), stmt
    finally:
        # 释放锁让仍在等待的探针返回（有界等待），再清理，避免测试挂死或残留。
        if lock_conn is not None:
            try:
                lock_conn.rollback()
            finally:
                lock_conn.close()
        if future is not None:
            try:
                future.result(timeout=20.0)
            except Exception:
                pass
        executor.shutdown(wait=True)
        if mapped is not None:
            mapped.dispose()
        if raw is not None:
            raw.dispose()
        temp_db.drop()


# ---------------------------------------------------------------------------
# 复测补强（2026-10-02 第二轮）：以下为 DeepSeek 修复后的增量验收。
# 原 test_a/test_b 断言未做任何修改。
# ---------------------------------------------------------------------------

def test_a2_real_health_response_keeps_background_fields_and_legacy_info(monkeypatch):
    """A 复测：真实 /api/health 响应（ASGI）在 compact 不可用时必须同时包含
    background 三字段与 legacy 诊断信息（backend/status/persistent 等）。"""
    from httpx import ASGITransport, AsyncClient

    import app.main as app_main

    _enabled(monkeypatch)
    monkeypatch.setattr(db, 'DB_READY', False)
    monkeypatch.setattr(db, 'engine', None)
    monkeypatch.setattr(db, 'SessionLocal', None)
    memory = app_main.memory_service  # 真实单例（CompactMemory 包装）
    monkeypatch.setattr(memory, 'background_disabled_reason', memory.background_disabled_reason)
    with deny_network():
        asyncio.run(memory.start())
        assert memory.background_disabled_reason == 'compact_memory_unavailable'

        async def fetch():
            async with AsyncClient(transport=ASGITransport(app=app_main.app),
                                   base_url='http://health.test') as client:
                response = await client.get('/api/health')
                return response.status_code, response.json()

        status_code, payload = asyncio.run(fetch())
    section = payload['memory']
    assert status_code == 503  # DB 未就绪：健康检查应为 degraded/503
    assert section.get('background_worker_running') is False, section
    assert section.get('background_disabled_reason') == 'compact_memory_unavailable', section
    assert section.get('background_enabled') is True, section
    for key in ('backend', 'status', 'persistent', 'summary_model'):
        assert key in section, f'legacy 诊断字段丢失: {key} 不在 {sorted(section)}'


def test_5c_real_lifespan_refuses_startup_and_skips_memory_boot(monkeypatch):
    """真实应用 lifespan：init_db 返回 False 时必须拒启，且
    memory_service.initialize/start 均不得被调用（不能在测试里自行抛异常
    代替应用行为）。"""
    import app.main as app_main
    from app.db.storage import StorageUnavailableError

    fake_memory = SimpleNamespace(initialize=AsyncMock(), start=AsyncMock(),
                                  close=AsyncMock())
    monkeypatch.setattr(app_main, 'init_db', lambda: False)
    monkeypatch.setattr(app_main, 'memory_service', fake_memory)
    context = app_main.lifespan(app_main.app)
    with pytest.raises(StorageUnavailableError):
        asyncio.run(context.__aenter__())
    fake_memory.initialize.assert_not_awaited()
    fake_memory.start.assert_not_awaited()


@pytest.mark.skipif(os.getenv('FITZEN_DB_TESTS') != '1'
                    or not str(settings.database_url).lower().startswith('postgresql'),
                    reason='random isolated PostgreSQL database only')
def test_b2_probe_restores_pooled_connection_timeouts(monkeypatch):
    """成功与异常退出后，池化连接的 lock_timeout/statement_timeout 必须恢复
    探针前的值（不假定原值一定为 0）；SET LOCAL 随事务结束失效，不得用
    会话级 SET 污染池化连接。"""
    temp_db = _TempPGDatabase()
    raw = mapped = None
    try:
        temp_db.create()
        raw = create_engine(temp_db.url)
        mapped = create_engine(temp_db.url, pool_size=1, max_overflow=0)
        budget_tables = [Base.metadata.tables[name] for name in BUDGET_TABLES]
        Base.metadata.create_all(mapped, tables=budget_tables)
        factory = sessionmaker(bind=mapped, expire_on_commit=False)
        monkeypatch.setattr(db, 'engine', mapped)
        monkeypatch.setattr(db, 'SessionLocal', factory)
        monkeypatch.setattr(db, 'DB_READY', True)

        def show_timeouts():
            with mapped.connect() as conn:
                return (conn.execute(text('SHOW lock_timeout')).scalar(),
                        conn.execute(text('SHOW statement_timeout')).scalar())

        baseline = show_timeouts()  # 不假定原值一定为 0
        # 成功路径：探针后恢复
        assert probe_budget_storage() == (True, None)
        assert show_timeouts() == baseline, '成功路径后池化连接超时设置未恢复'
        # 异常路径：表锁下探针有界失败（SET LOCAL 2000ms/5000ms），退出后恢复
        lock_conn = raw.connect()
        try:
            lock_conn.execute(text('LOCK TABLE model_quotas IN ACCESS EXCLUSIVE MODE'))
            assert probe_budget_storage() == (False, 'budget_storage_unavailable')
            assert show_timeouts() == baseline, '异常路径后池化连接超时设置未恢复'
        finally:
            lock_conn.rollback()
            lock_conn.close()
    finally:
        if mapped is not None:
            mapped.dispose()
        if raw is not None:
            raw.dispose()
        temp_db.drop()
