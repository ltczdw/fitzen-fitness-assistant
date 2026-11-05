"""Tests for react supervisor budget."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.model_budget_models import ModelCallLedger, ModelQuota, ModelResultCache
from app.services.llm.budget import BudgetStore, model_scope
from app.services.llm.client import LLMClient
from app.services.agent.react_graph import (
    MAX_SUPERVISOR_PHYSICAL_ATTEMPTS,
    ReactDependencies,
)

SRC = "你好"


@pytest.fixture
def budget_store(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "budgets.db"))
    for model in (ModelQuota, ModelCallLedger, ModelResultCache):
        model.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield BudgetStore(factory)
    engine.dispose()


def _gateway_llm(monkeypatch, handler, base_url):
    monkeypatch.setattr(settings, "llm_api_key", "mock-not-real")
    monkeypatch.setattr(settings, "llm_base_url", base_url)
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0,0,0")
    original = httpx.AsyncClient
    seen = []

    def counting_handler(request):
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(counting_handler), **kwargs))
    return seen


def _react_deps(llm_client):
    # tool_executor/llm 均为占位：supervisor_step 只调用 llm.complete_message。
    return ReactDependencies(llm=llm_client, tool_executor=None, user_id="u1",
                             request_id="req-budget")


def _assistant_state():
    return {"messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": SRC}]}


def _always_503(request):
    return httpx.Response(503)


def test_supervisor_retries_count_into_react_supervisor_stage(
        budget_store, monkeypatch):
    """对照：解除自动stage/熔断两个更低上限后，物理尝试=4且全部计入同一 stage。"""
    monkeypatch.setattr(settings, "model_stage_max_calls", MAX_SUPERVISOR_PHYSICAL_ATTEMPTS)
    monkeypatch.setattr(settings, "llm_circuit_failure_threshold", 50)
    seen = _gateway_llm(monkeypatch, _always_503, "https://react-budget-a.invalid/v1")

    async def run():
        client = LLMClient()
        client.admission.token = AsyncMock()
        with model_scope("react-budget-a", store=budget_store, max_calls=12):
            deps = _react_deps(client)
            first = await deps_llm_round(deps)
            second = await deps_llm_round(deps)
        return first, second

    async def deps_llm_round(deps):
        from app.services.agent.react_graph import supervisor_step
        try:
            await supervisor_step(_assistant_state(), deps)
            return "ok"
        except Exception as exc:
            return type(exc).__name__

    first, second = asyncio.run(run())
    transport_attempts = len(seen)
    assert transport_attempts == MAX_SUPERVISOR_PHYSICAL_ATTEMPTS, (
        f"合同7对照: 网关重试应计入稳定 react_supervisor stage, 物理尝试 "
        f"{transport_attempts} != {MAX_SUPERVISOR_PHYSICAL_ATTEMPTS}")
    assert first != "ok", "合同7对照: 持续503必须失败而非伪成功"
    # 第二轮 supervisor 调用零传输：stage 额度已被第一轮重试耗尽（fail-closed）。
    assert second in {"ModelBudgetError", "LLMGatewayError"}, (
        f"合同7对照: stage耗尽后必须立即拒绝: {second!r}")
    with budget_store.factory() as session:
        ledgers = session.scalars(select(ModelCallLedger)).all()
        chat_attempts = [row for row in ledgers if row.capability == "chat"]
        assert len(chat_attempts) == MAX_SUPERVISOR_PHYSICAL_ATTEMPTS, (
            f"合同7对照: 账本物理尝试数 {len(chat_attempts)}")
        stage_quotas = [row for row in session.scalars(select(ModelQuota)).all()
                        if row.kind == "stage"]
        assert any(quota.calls == MAX_SUPERVISOR_PHYSICAL_ATTEMPTS for quota in stage_quotas), (
            f"合同7对照: stage额度未按重试累计: "
            f"{[(q.quota_id, q.calls) for q in stage_quotas]}")


def test_default_settings_capped_by_stricter_stage_with_ledger(
        budget_store, monkeypatch):
    """T1修订（复验合同§2）：原「恰好4次」断言错误，合同是最多4且受更严限制。

    默认配置下实际物理尝试 = min(自动per-payload stage=model_stage_max_calls=2,
    熔断阈值=3, react_supervisor stage=4) = 2。本fixture须验证：确切的更严上限、
    真实账本计数与失败状态；不得放宽为0<=n<=4。不在测试外提高预算/熔断配置。
    """
    seen = _gateway_llm(monkeypatch, _always_503, "https://react-budget-b.invalid/v1")

    async def run():
        client = LLMClient()
        client.admission.token = AsyncMock()
        with model_scope("react-budget-b", store=budget_store, max_calls=12):
            deps = _react_deps(client)
            from app.services.agent.react_graph import supervisor_step
            try:
                await supervisor_step(_assistant_state(), deps)
                return "ok"
            except Exception as exc:
                return type(exc).__name__

    outcome = asyncio.run(run())
    stricter_cap = min(settings.model_stage_max_calls, settings.llm_circuit_failure_threshold,
                       MAX_SUPERVISOR_PHYSICAL_ATTEMPTS)
    assert len(seen) == stricter_cap, (
        f"合同7(T1): 默认配置物理尝试 {len(seen)} != 更严上限 {stricter_cap} "
        f"(自动stage={settings.model_stage_max_calls}, 熔断={settings.llm_circuit_failure_threshold}, "
        f"react_supervisor={MAX_SUPERVISOR_PHYSICAL_ATTEMPTS}), outcome={outcome!r}")
    assert outcome != "ok", "合同7(T1): 持续503必须以失败收场，不得伪成功"
    with budget_store.factory() as session:
        chat_rows = [row for row in session.scalars(select(ModelCallLedger)).all()
                     if row.capability == "chat"]
        assert len(chat_rows) == stricter_cap, (
            f"合同7(T1): 账本物理尝试计数 {len(chat_rows)} != {stricter_cap}")
        stage_quotas = [row for row in session.scalars(select(ModelQuota)).all()
                        if row.kind == "stage"]
        assert any(quota.calls == stricter_cap for quota in stage_quotas), (
            f"合同7(T1): stage额度未按实际尝试累计: "
            f"{[(q.quota_id, q.calls) for q in stage_quotas]}")


def test_daily_cap_tightens_within_supervisor_stage(budget_store, monkeypatch):
    monkeypatch.setattr(settings, "model_daily_max_calls", 1)
    seen = _gateway_llm(monkeypatch, _always_503, "https://react-budget-c.invalid/v1")

    async def run():
        client = LLMClient()
        client.admission.token = AsyncMock()
        with model_scope("react-budget-c", store=budget_store, max_calls=12):
            deps = _react_deps(client)
            from app.services.agent.react_graph import supervisor_step
            try:
                await supervisor_step(_assistant_state(), deps)
                return "ok"
            except Exception as exc:
                return type(exc).__name__

    asyncio.run(run())
    assert len(seen) == 1, (
        f"合同7: request/day 更严限制应在主管 stage 内生效, 实际传输尝试 {len(seen)}")


def test_missing_budget_scope_fails_closed_not_reported_verified(monkeypatch):
    """conftest autouse 夹具会给每个测试套一层隔离 scope；为复现「缺预算scope」，
    在协程上下文内显式清空 active_scope，验证 fail-closed 而非伪报预算已验证。"""
    seen = _gateway_llm(monkeypatch, _always_503, "https://react-budget-d.invalid/v1")
    from app.services.llm.budget import active_scope
    from app.services.agent.react_graph import _main_agent_node

    async def run():
        client = LLMClient()
        client.admission.token = AsyncMock()
        token = active_scope.set(None)
        try:
            # 故意无 model_scope：预算基础设施缺失时必须 fail-closed。
            return await _main_agent_node(_assistant_state(), _react_deps(client))
        finally:
            active_scope.reset(token)

    result = asyncio.run(run())
    assert len(seen) == 0, f"合同7: 缺scope不得发起任何物理尝试: {len(seen)}"
    assert result["terminated"] is True
    assert result["termination_reason"] == "model_unavailable"
    assert result["protocol_error"] == "model_unavailable:LLMGatewayError"
