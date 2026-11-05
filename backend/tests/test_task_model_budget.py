"""Independent semantic-task model budgets (uniform 12 per business task).

The durable store and real gateway are used throughout; provider traffic is
always a local MockTransport and SQLite is isolated under ``tmp_path``.
"""
from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from unittest.mock import AsyncMock

from app.core import clock
from app.core.config import settings
from app.db.model_budget_models import ModelCallLedger, ModelQuota, ModelResultCache
from app.services.llm.budget import (
    BudgetStore,
    ModelBudgetError,
    ModelBudgetUnavailable,
    ModelTaskBudgetError,
    key,
    model_prepare,
    model_scope,
    model_task,
    paid_attempt,
)


def make_store(tmp_path, name="task-budget.db"):
    engine = create_engine("sqlite:///" + str(tmp_path / name))
    for model in (ModelQuota, ModelCallLedger, ModelResultCache):
        model.__table__.create(engine)
    return BudgetStore(sessionmaker(bind=engine, expire_on_commit=False)), engine


@pytest.fixture
def store(tmp_path):
    budget, engine = make_store(tmp_path)
    try:
        yield budget
    finally:
        engine.dispose()


@contextmanager
def request_scope(store, *, request_id="req-task", user="user-a", **options):
    options.setdefault("max_calls", 12)
    options.setdefault("max_tokens", 64000)
    with model_scope(
        "request:" + request_id,
        kind="request",
        store=store,
        owner_id=key(user + ":conversation"),
        policy="per_task",
        **options,
    ) as scope:
        yield scope


def spend(goal, *, calls=1, task_type="training_qa", tokens=None):
    with model_task(goal, task_type=task_type):
        for index in range(calls):
            with paid_attempt("chat", "mock-model", {"goal": goal, "index": index}, 8) as attempt:
                if tokens is not None:
                    attempt.observe({"usage": {"total_tokens": tokens}})


def quota_rows(store):
    with store.factory() as session:
        return list(session.scalars(select(ModelQuota)))


def ledger_rows(store):
    with store.factory() as session:
        return list(session.scalars(select(ModelCallLedger).order_by(ModelCallLedger.started_at)))


def test_three_tasks_exceed_old_request_twelve_and_each_stays_within_twelve(store):
    with request_scope(store) as scope:
        for goal in ("goal-a", "goal-b", "goal-c"):
            spend(goal, calls=5, tokens=5000)
        # goal-a has 5 calls; seven more reach the uniform 12 and the next fails.
        with pytest.raises(ModelTaskBudgetError, match="task"):
            spend("goal-a", calls=8)
        # A task stop does not consume or block a different task's allowance.
        spend("goal-b", calls=3)
        spend("goal-c", calls=3)
        assert scope.policy == "per_task"

    with store.factory() as session:
        root = session.get(ModelQuota, scope.quota_id)
        tasks = {row.quota_id: row for row in session.scalars(select(ModelQuota))
                 if row.quota_id.startswith("task:")}
    assert root.calls == 28
    assert root.calls > settings.model_request_max_calls
    assert root.tokens > settings.model_request_max_tokens
    assert len(tasks) == 3
    assert sorted(row.calls for row in tasks.values()) == [8, 8, 12]
    assert all(row.max_calls <= settings.model_task_max_calls for row in tasks.values())


def test_uniform_twelve_for_plan_and_standard_with_prepare_shared_once(store):
    with request_scope(store) as scope:
        spend("plan-goal", calls=12, task_type="training_plan")
        with pytest.raises(ModelTaskBudgetError, match="task"):
            spend("plan-goal", calls=1, task_type="training_plan")
        spend("standard-goal", calls=12)
        with pytest.raises(ModelTaskBudgetError, match="task"):
            spend("standard-goal", calls=1)
        with model_prepare():
            for index in range(settings.model_prepare_max_calls):
                with paid_attempt("chat", "mock", {"prepare": index}, 8):
                    pass
            with pytest.raises(ModelBudgetError, match="prepare"):
                with paid_attempt("chat", "mock", {"prepare": "overflow"}, 8):
                    pass
        # Prepare has its own fixed identity and is not charged to user tasks.
        spend("another-task", calls=1)

    with store.factory() as session:
        prepare = [row for row in session.scalars(select(ModelQuota)) if row.kind == "prepare"]
        plan = [row for row in session.scalars(select(ModelQuota))
                if row.kind == "task_plan"]
        standard = [row for row in session.scalars(select(ModelQuota))
                    if row.kind == "task_standard"]
        root = session.get(ModelQuota, scope.quota_id)
    assert len(prepare) == 1 and prepare[0].calls == settings.model_prepare_max_calls
    assert len(plan) == 1 and plan[0].max_calls == settings.model_task_max_calls == 12
    assert len(standard) == 2
    assert all(row.max_calls == settings.model_task_max_calls for row in plan + standard)
    assert root.calls == 12 + 12 + settings.model_prepare_max_calls + 1


def test_deprecated_tier_fields_cannot_change_a_new_task_allowance(store, monkeypatch):
    # Old .env values remain loadable for compatibility but are never consulted
    # when a new business task is created.
    monkeypatch.setattr(settings, "model_task_standard_max_calls", 1, raising=False)
    monkeypatch.setattr(settings, "model_task_plan_max_calls", 2, raising=False)
    with request_scope(store, request_id="deprecated-tiers"):
        with model_task("new-standard", task_type="training_qa") as standard:
            assert standard.max_calls == settings.model_task_max_calls == 12
        with model_task("new-plan", task_type="training_plan") as plan:
            assert plan.max_calls == settings.model_task_max_calls == 12


def test_task_identity_uses_frozen_span_not_renamed_type_or_store_rebuild(tmp_path, store):
    from app.services.agent.task_budget import task_goal

    source = "alpha span and beta span"
    first = {"id": "model-free-id-1", "task_type": "training_qa", "text": "alpha span",
             "payload": {"source_start": 0, "source_end": 10}}
    renamed = {"id": "model-free-id-2", "task_type": "training_plan", "text": "renamed text",
               "payload": {"source_start": 0, "source_end": 10}}
    other = {"id": "model-free-id-3", "task_type": "training_qa", "text": "beta span",
             "payload": {"source_start": 15, "source_end": 24}}
    assert task_goal(first, source) == task_goal(renamed, source)
    assert task_goal(first, source) != task_goal(other, source)

    with request_scope(store, request_id="frozen-identity"):
        spend(task_goal(first, source), calls=12, task_type="training_qa")
    # Rebuilding the store object over the same durable factory cannot refresh
    # or upgrade the already frozen task allowance.
    rebuilt = BudgetStore(store.factory)
    with request_scope(rebuilt, request_id="frozen-identity"):
        with pytest.raises(ModelTaskBudgetError, match="task"):
            spend(task_goal(renamed, source), calls=1, task_type="training_plan")
        spend(task_goal(other, source), calls=1)
    # A different user with the same request id and span is isolated.
    with request_scope(rebuilt, request_id="frozen-identity", user="user-b"):
        spend(task_goal(first, source), calls=12, task_type="training_qa")


def test_root_aggregate_is_not_enforced_but_daily_deadline_cancel_and_storage_are(tmp_path, store, monkeypatch):
    with request_scope(store) as scope:
        for index in range(13):
            spend("aggregate-" + str(index), calls=1, tokens=6000)
        assert scope.remaining() > 0
        scope.started = time.monotonic() - scope.ttl_seconds - 1
        with pytest.raises(ModelBudgetError, match="deadline"):
            spend("deadline-goal", calls=1)
    with request_scope(store, request_id="cancel-request") as scope:
        scope.cancelled = True
        with pytest.raises(ModelBudgetError, match="cancelled"):
            spend("cancel-goal", calls=1)
    daily_store, engine = make_store(tmp_path, "daily-strict.db")
    try:
        monkeypatch.setattr(settings, "model_daily_max_calls", 1)
        with request_scope(daily_store, request_id="daily-request"):
            with pytest.raises(ModelBudgetError, match="daily"):
                spend("first-daily", calls=2)
    finally:
        engine.dispose()
    broken = BudgetStore(lambda: (_ for _ in ()).throw(RuntimeError("storage down")))
    with model_scope("request:broken", kind="request", store=broken,
                     owner_id=key("user-a:conversation"), policy="per_task"):
        with pytest.raises(ModelBudgetUnavailable):
            spend("broken-goal", calls=1)


def test_old_shared_request_with_prior_consumption_gets_no_new_task_quota(store):
    moment = clock.utc_now()
    root_id = "scope:" + key("request:" + "request:old-request")
    with store.factory() as session, session.begin():
        session.add(ModelQuota(quota_id=root_id, kind="request",
                               calls=11, tokens=100, max_calls=12, max_tokens=64000,
                               created_at=moment, expires_at=moment + timedelta(seconds=600)))
    with request_scope(store, request_id="old-request") as scope:
        # Recovery resolves the frozen old strategy before cache-key use.
        scope.store.resolve_scope_policy(scope)
        assert scope.policy == "shared"
        with model_task("old-task", task_type="training_qa") as task:
            assert task is None  # no new per-task allowance for an old request
            with paid_attempt("chat", "mock", {"old": 12}, 8):
                pass
            with pytest.raises(ModelBudgetError, match="exhausted:request"):
                with paid_attempt("chat", "mock", {"old": 13}, 8):
                    pass
    with store.factory() as session:
        assert session.get(ModelQuota, root_id).calls == 12
        assert session.get(ModelQuota, "policy:" + key(root_id)) is None
        assert not [row for row in session.scalars(select(ModelQuota))
                    if row.quota_id.startswith("task:")]


def test_frozen_zero_consumption_request_keeps_shared_limits_and_no_task_quota(store):
    from app.services.agent.react_write_runtime import _budget_bootstrap

    with request_scope(store, request_id="frozen-zero") as scope:
        assert asyncio.run(_budget_bootstrap(scope, frozen_limits=(3, 1000))) is True
        assert scope.policy == "shared"
        with model_task("should-not-exist", task_type="training_plan") as task:
            assert task is None
            for index in range(3):
                with paid_attempt("chat", "mock", {"frozen": index}, 8):
                    pass
            with pytest.raises(ModelBudgetError, match="exhausted:request"):
                with paid_attempt("chat", "mock", {"frozen": 4}, 8):
                    pass
    with store.factory() as session:
        assert session.get(ModelQuota, scope.quota_id).max_calls <= 3
        assert session.get(ModelQuota, scope.quota_id).max_tokens <= 1000
        assert not [row for row in session.scalars(select(ModelQuota))
                    if row.quota_id.startswith(("task:", "prepare:"))]


def test_task_token_stage_and_daily_failures_roll_back_cross_layer_counts(store, monkeypatch):
    with request_scope(store) as scope:
        with model_task("atomic-goal", task_type="training_qa") as task:
            moment = clock.utc_now()
            with store.factory() as session, session.begin():
                session.add(ModelQuota(quota_id=task.quota_id, kind="task_standard", calls=0,
                                       tokens=0, max_calls=8, max_tokens=1,
                                       created_at=moment, expires_at=moment + timedelta(seconds=600)))
            def snapshot():
                return {(row.quota_id, row.kind, row.calls, row.tokens)
                        for row in quota_rows(store)}
            before = snapshot()
            with pytest.raises(ModelTaskBudgetError, match="token"):
                with paid_attempt("chat", "mock", {"too-large": "x" * 5000}, 8):
                    pass
            assert snapshot() == before
        # A shared stage ceiling still stops before transport and rolls back
        # the task/root/daily rows created by the failed attempt.
        with model_scope("request:stage-fail", kind="request", store=store,
                         owner_id=key("user-a:conversation"), policy="per_task"):
            from app.services.llm.budget import model_stage
            before = snapshot()
            with model_stage("bounded-stage", max_calls=1):
                spend("stage-goal", calls=1)
                before = snapshot()
                with pytest.raises(ModelBudgetError):
                    spend("stage-goal", calls=1)
            assert snapshot() == before


def test_async_to_thread_shares_the_same_task_context(store):
    async def run():
        with request_scope(store, request_id="thread-task"):
            with model_task("thread-goal", task_type="training_qa"):
                for index in range(settings.model_task_max_calls - 1):
                    with paid_attempt("chat", "mock", {"index": index}, 0):
                        pass
                def threaded():
                    with paid_attempt("chat", "mock", {"thread": True}, 0):
                        pass
                    with pytest.raises(ModelTaskBudgetError):
                        with paid_attempt("chat", "mock", {"thread": "overflow"}, 0):
                            pass
                await asyncio.to_thread(threaded)

    asyncio.run(run())


def test_same_task_concurrency_never_exceeds_limit(store):
    from contextvars import copy_context

    def call(index):
        try:
            with model_task("parallel-goal", task_type="training_qa"):
                with paid_attempt("chat", "mock", {"index": index}, 0):
                    pass
            return True
        except ModelTaskBudgetError:
            return False

    with request_scope(store):
        contexts = [copy_context() for _ in range(16)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda pair: pair[0].run(call, pair[1]),
                                    zip(contexts, range(16))))
    assert sum(results) == settings.model_task_max_calls == 12


def test_task_cache_is_target_scoped_and_hits_make_no_second_physical_call(store, monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "mock-not-real")
    monkeypatch.setattr(settings, "llm_base_url", "https://task-cache.invalid/v1")
    monkeypatch.setattr(settings, "llm_model", "mock")
    from app.services.llm.client import LLMClient

    seen = []
    original = httpx.AsyncClient

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                         "usage": {"prompt_tokens": 4, "completion_tokens": 2}})

    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))

    async def run():
        client = LLMClient()
        client.admission.token = AsyncMock()
        messages = [{"role": "user", "content": "synthetic same payload"}]
        with request_scope(store, request_id="cache-request"):
            with model_task("cache-a", task_type="training_qa"):
                assert await client.complete(messages) == "ok"
                assert await client.complete(messages) == "ok"
            with model_task("cache-b", task_type="training_qa"):
                assert await client.complete(messages) == "ok"

    asyncio.run(run())
    assert len(seen) == 2  # same target cached; a different target is isolated
    rows = ledger_rows(store)
    assert len(rows) == 2
    assert all(any(qid.startswith("task:") for qid in row.quota_ids) for row in rows)


def test_react_limited_task_keeps_other_answers_and_is_never_complete(store):
    from app.services.agent.react_graph import (
        ReactDependencies, execute_react_batch, finalize_node,
    )
    from app.services.agent.react_models import DomainObservation, SourceScope, ToolStatus

    source = "first question; second question"
    candidates = [
        {"id": "goal-a", "task_type": "training_qa", "text": "first question",
         "source_start": 0, "source_end": 14},
        {"id": "goal-b", "task_type": "training_qa", "text": "second question",
         "source_start": 16, "source_end": 31},
    ]
    def call(call_id, start, end):
        return {"id": call_id, "type": "function", "function": {
            "name": "answer_training",
            "arguments": json.dumps({"query": source[start:end],
                                     "source_ref": {"start": start, "end": end,
                                                    "exact_text": source[start:end]}}) }}

    async def execute(name, args, context):
        if args["source_ref"]["start"] == 0:
            for index in range(settings.model_task_max_calls + 1):
                try:
                    with paid_attempt("chat", "mock", {"a": index}, 0):
                        pass
                except ModelTaskBudgetError:
                    break
            # A worker that swallows the budget error still cannot become complete.
            return DomainObservation(status=ToolStatus.COMPLETED, summary="A answer")
        with paid_attempt("chat", "mock", {"b": 1}, 0):
            pass
        return DomainObservation(status=ToolStatus.COMPLETED, summary="B answer")

    state = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": source},
            {"role": "assistant", "content": "", "tool_calls": [
                call("call-a", 0, 14), call("call-b", 16, 31)]},
        ],
        "intent_candidates": candidates,
        "source_text": source,
        "source_scope": SourceScope(text=source, intent_candidates=candidates),
        "user_id": "user-a", "conversation_id": "conversation",
        "request_id": "react-limit", "observations": [], "covered_goals": [],
        "committed_tasks": [], "goal_results": {}, "pending_goals": [],
        "tool_steps": 0, "tool_steps_by_task": {}, "limited_tasks": [],
    }
    deps = ReactDependencies(llm=object(), tool_executor=execute,
                             user_id="user-a", request_id="react-limit")
    with request_scope(store, request_id="react-limit"):
        result = asyncio.run(execute_react_batch(state, deps))
        final = finalize_node({**state, **result}, deps)
    assert "goal-b" in result["covered_goals"]
    assert "goal-b" in result["goal_results"]
    assert "goal-a" in result["limited_tasks"]
    assert "goal-a" not in result["goal_results"]
    assert final["status"] == "limited"
    assert "B answer" in final["final_answer"]
    assert "A answer" not in final["final_answer"]


def test_task_limited_never_erases_real_commit_receipts():
    from types import SimpleNamespace
    from app.services.agent.react_graph import _bounded_task_observation
    from app.services.agent.react_models import DomainObservation, ToolStatus

    task = SimpleNamespace(limited=True, exhausted_code="model_task_budget_exhausted:task")
    committed = DomainObservation(status=ToolStatus.COMPLETED, commit_state="committed",
                                  receipt_refs=["rcpt-real"])
    assert _bounded_task_observation(committed, task) is committed
    read = DomainObservation(status=ToolStatus.COMPLETED, summary="late read answer")
    bounded = _bounded_task_observation(read, task)
    assert bounded.status == ToolStatus.UNAVAILABLE
    assert bounded.summary != "late read answer"


def test_legacy_worker_charges_its_own_task_and_other_worker_continues(store):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    source = "first legacy goal; second legacy goal"

    class Worker:
        async def ainvoke(self, state):
            if state["task_id"] == "task-1":
                for index in range(settings.model_task_max_calls + 1):
                    try:
                        with paid_attempt("chat", "mock", {"task": "one", "i": index}, 0):
                            pass
                    except ModelTaskBudgetError:
                        break
                # Swallowed by a worker fallback, but must not become complete.
                return {"result": {"worker": "training_qa", "status": "complete",
                                   "answer": "task one answer", "citations": [], "model_used": True}}
            with paid_attempt("chat", "mock", {"task": "two"}, 0):
                pass
            return {"result": {"worker": "training_qa", "status": "complete",
                               "answer": "task two answer", "citations": [], "model_used": True}}

    graph = FitnessMultiAgentGraph()
    graph.workers["training_qa"] = Worker()

    def state(task_id, start, end):
        text = source[start:end]
        return {
            "user_id": "user-a", "conversation_id": "conversation",
            "raw_message": source, "request_id": "legacy-task",
            "current_task": {"id": task_id, "task_type": "training_qa",
                             "worker": "training_qa", "action": "question", "text": text,
                             "payload": {"source_start": start, "source_end": end},
                             "depends_on": []},
            "profile": {}, "training_history": [], "nutrition_summary": {},
            "worker_results": [], "completed_tasks": [], "model_used": False,
            "memory_backend": "compact_v1",
        }

    with request_scope(store, request_id="legacy-task"):
        first = asyncio.run(graph._run_worker("training_qa", state("task-1", 0, 17)))
        second = asyncio.run(graph._run_worker("training_qa", state("task-2", 19, 36)))
    assert first["worker_results"][0]["status"] == "execution_limited"
    assert first["completed_tasks"] == []
    assert second["worker_results"][0]["status"] == "complete"
    assert second["completed_tasks"] == ["task-2"]
    with store.factory() as session:
        tasks = [row for row in session.scalars(select(ModelQuota))
                 if row.quota_id.startswith("task:")]
    assert sorted(row.calls for row in tasks) == [1, settings.model_task_max_calls]


def test_write_bootstrap_marks_new_policy_and_never_upgrades_old_request(store):
    from app.services.agent.react_write_runtime import _budget_bootstrap
    from app.services.llm.budget import ModelScope, key as budget_key

    owner = budget_key("user-a:conversation")
    new_scope = ModelScope("request:write-new", "request", store, 12, 64000, 600,
                           owner_id=owner, policy="per_task")
    assert asyncio.run(_budget_bootstrap(new_scope)) is True
    assert new_scope.policy == "per_task"
    marker_id = "policy:" + budget_key(new_scope.quota_id)
    with store.factory() as session:
        marker = session.get(ModelQuota, marker_id)
        assert marker is not None and marker.kind == "budget_policy"

    moment = clock.utc_now()
    old_id = "scope:" + budget_key("request:" + "request:write-old")
    with store.factory() as session, session.begin():
        session.add(ModelQuota(quota_id=old_id, kind="request", calls=3, tokens=100,
                               max_calls=12, max_tokens=64000,
                               created_at=moment, expires_at=moment + timedelta(seconds=600)))
    old_scope = ModelScope("request:write-old", "request", store, 12, 64000, 600,
                           owner_id=owner, policy="per_task")
    assert asyncio.run(_budget_bootstrap(old_scope)) is True
    assert old_scope.policy == "shared"
    explicit_shared = ModelScope("request:write-explicit", "request", store, 1, 1000, 600,
                                  owner_id=owner, policy="shared")
    assert asyncio.run(_budget_bootstrap(explicit_shared)) is True
    assert explicit_shared.policy == "shared"
    with store.factory() as session:
        assert session.get(ModelQuota, "policy:" + budget_key(old_scope.quota_id)) is None
        assert session.get(ModelQuota, "policy:" + budget_key(explicit_shared.quota_id)) is None
        assert not [row for row in session.scalars(select(ModelQuota))
                    if row.quota_id.startswith("task:")]


def test_real_http_legacy_and_readonly_charge_dispatched_tasks(compact_db, monkeypatch):
    """No replacement of _respond_impl: real default/readonly scheduling."""
    import app.services.rag.online as rag_online
    from types import SimpleNamespace
    from app.main import app
    from app.services.agent.agent import agent
    from app.services.agent import execution
    from app.services.llm.budget import store_override
    from app.services.llm.client import LLMClient

    source = "如何安排每周三次力量训练并保证恢复？"
    monkeypatch.setattr(settings, "llm_api_key", "mock-not-real")
    monkeypatch.setattr(settings, "llm_base_url", "https://real-entry-budget.invalid/v1")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0,0,0")
    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "planner_shadow_enabled", False)
    monkeypatch.setattr(agent, "_run_planner_shadow", AsyncMock())
    monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value={
        "memory_backend": "compact_v1", "short_memory": [], "retrieved_history": [],
        "nutrition_summary": {}, "information_sources": [], "long_memory": {},
    }))
    async def call_mcp(name, arguments, **kwargs):
        return [] if name == "get_training_history" else {}
    monkeypatch.setattr(agent.graph, "_call_mcp", call_mcp)
    monkeypatch.setattr(agent.graph.mcp, "complete_with_tools", AsyncMock(return_value=None))

    hits = [
        SimpleNamespace(title="恢复原则", source="knowledge://budget",
                        chunk_id="chunk-1", content="每周三次力量训练应安排恢复日和动作质量。",
                        score=1.0, metadata={"rerank_status": "model", "rerank_score": 1.0}),
        SimpleNamespace(title="训练频率", source="knowledge://budget",
                        chunk_id="chunk-2", content="每周三次力量训练可覆盖主要肌群，并保证恢复。",
                        score=0.99, metadata={"rerank_status": "model", "rerank_score": 1.0}),
        SimpleNamespace(title="恢复", source="knowledge://budget",
                        chunk_id="chunk-3", content="力量训练之间安排恢复，避免连续高强度。",
                        score=0.98, metadata={"rerank_status": "model", "rerank_score": 1.0}),
    ]
    async def search(*args, **kwargs):
        return hits
    monkeypatch.setattr(rag_online, "search", search, raising=False)

    understanding = {
        "tasks": [{"task_type": "training_qa", "action": "question", "text": source,
                   "is_negated": False, "is_hypothetical": False, "is_third_party": False}],
        "reply_mode": "grounded", "reply_purpose": "general", "relation": "new",
        "current_is_critical": False, "critical_fact_fields": [], "critical_history_ids": [],
        "dialogue_ids": [], "references": [], "memory_requests": [], "needs_clarification": False,
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "normal"},
    }
    supervisor_calls = []
    def provider(request):
        body = json.loads(request.content.decode("utf-8"))
        tools = {item.get("function", {}).get("name") for item in body.get("tools") or []}
        if "return_request_understanding" in tools:
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "understanding", "type": "function",
                "function": {"name": "return_request_understanding",
                             "arguments": json.dumps(understanding, ensure_ascii=False)}}]}
        elif tools & {"answer_training", "advise_nutrition", "plan_training"}:
            supervisor_calls.append(1)
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "read-answer", "type": "function",
                "function": {"name": "answer_training",
                             "arguments": json.dumps({"query": source},
                                                     ensure_ascii=False)}}]}
        else:
            message = {"role": "assistant", "content": "真实入口训练回答"}
        return httpx.Response(200, json={"choices": [{"message": message}],
                                         "usage": {"prompt_tokens": 12, "completion_tokens": 5}})

    original = httpx.AsyncClient
    store = BudgetStore(compact_db)
    token = store_override.set(store)
    try:
        async def run():
            transport = httpx.ASGITransport(app=app)
            async with original(transport=transport, base_url="http://real-entry") as api:
                monkeypatch.setattr(
                    httpx, "AsyncClient",
                    lambda **kwargs: original(transport=httpx.MockTransport(provider), **kwargs))
                for mode in ("legacy", "react_readonly"):
                    response = await api.post("/api/chat", json={
                        "message": source, "mode": mode, "request_id": "real-" + mode,
                        "conversation_id": "real-" + mode})
                    assert response.status_code == 200, response.text
                    assert response.json()["answer"]
        asyncio.run(run())
    finally:
        store_override.reset(token)

    with compact_db() as session:
        task_rows = [row for row in session.scalars(select(ModelQuota))
                     if row.kind == "task_standard"]
        prepare_rows = [row for row in session.scalars(select(ModelQuota))
                        if row.kind == "prepare"]
        markers = [row for row in session.scalars(select(ModelQuota))
                   if row.kind == "budget_policy"]
        ledgers = list(session.scalars(select(ModelCallLedger)))
    assert len(task_rows) == 2 and all(row.calls >= 1 for row in task_rows), (
        [(row.quota_id, row.kind, row.calls) for row in task_rows],
        [(row.quota_id, row.kind, row.calls) for row in prepare_rows],
        [(row.scope_id, row.kind, row.capability, row.status) for row in ledgers])
    assert prepare_rows and markers
    assert len(ledgers) >= 4
    task_ledgers = [row for row in ledgers
                    if any(qid.startswith("task:") for qid in row.quota_ids)]
    assert len(task_ledgers) >= 2 and all(row.capability == "chat" for row in task_ledgers)
    assert supervisor_calls  # readonly supervisor was a real graph call, not a fake response


def test_http_json_sse_and_modes_charge_tasks_through_real_gateway(compact_db, monkeypatch):
    from app.main import app
    from app.schemas.chat import ChatResponse
    from app.services.agent.agent import agent
    from app.services.llm.client import LLMClient

    monkeypatch.setattr(settings, "llm_api_key", "mock-not-real")
    monkeypatch.setattr(settings, "llm_base_url", "https://http-task-budget.invalid/v1")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0,0,0")
    monkeypatch.setattr(agent, "_run_planner_shadow", AsyncMock())

    seen = []
    original = httpx.AsyncClient

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}],
                                         "usage": {"prompt_tokens": 4, "completion_tokens": 2}})

    async def fake_respond(request, parsed_input=None):
        client = LLMClient()
        client.admission.token = AsyncMock()
        for goal in ("http-a", "http-b"):
            with model_task(goal, task_type="training_qa"):
                for index in range(7):
                    await client.complete([
                        {"role": "user", "content": f"{request.request_id}:{goal}:{index}"}])
        return ChatResponse(answer="ok", intent="training_qa", model_used=True)

    monkeypatch.setattr(agent, "_respond_impl", fake_respond)
    store = BudgetStore(compact_db)
    from app.services.llm.budget import store_override
    token = store_override.set(store)
    try:
        async def run():
            transport = httpx.ASGITransport(app=app)
            async with original(transport=transport, base_url="http://task-budget") as api:
                # The ASGI client is already bound. Patch only provider calls
                # made later by fake_respond inside the request.
                monkeypatch.setattr(
                    httpx, "AsyncClient",
                    lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
                for mode in ("legacy", "react_readonly", "react_write"):
                    response = await api.post("/api/chat", json={
                        "message": "synthetic " + mode, "mode": mode,
                        "request_id": "http-" + mode, "conversation_id": "http-" + mode})
                    assert response.status_code == 200 and response.json()["answer"] == "ok"
                async with api.stream("POST", "/api/chat/stream", json={
                        "message": "synthetic sse", "mode": "legacy",
                        "request_id": "http-sse", "conversation_id": "http-sse"}) as response:
                    body = "".join([chunk async for chunk in response.aiter_text()])
                    assert response.status_code == 200
                    assert '"status"' in body or '"answer"' in body or "event: meta" in body
        asyncio.run(run())
    finally:
        store_override.reset(token)

    with compact_db() as session:
        root_rows = [row for row in session.scalars(select(ModelQuota)) if row.kind == "request"]
        task_rows = [row for row in session.scalars(select(ModelQuota)) if row.quota_id.startswith("task:")]
        ledgers = list(session.scalars(select(ModelCallLedger)))
    assert len(root_rows) == 4
    assert all(row.calls == 14 for row in root_rows)
    assert len(task_rows) == 8
    assert all(row.calls == 7 for row in task_rows)
    assert len(ledgers) == 56
    assert all(any(qid.startswith("task:") for qid in row.quota_ids)
               and any(qid.startswith("daily:") for qid in row.quota_ids)
               and any(qid.startswith("stage:") for qid in row.quota_ids)
               for row in ledgers)
