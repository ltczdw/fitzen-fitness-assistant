import asyncio
from contextvars import ContextVar
from threading import Event, get_ident

import pytest

from app.services.rag.blocking import BlockingRetrievalPool
from app.services.rag.retrieval import embedding_cache, retrieval_traces


async def wait_started(event):
    async def poll():
        while not event.is_set():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(poll(), timeout=2)


def test_slow_retrieval_does_not_block_loop_and_preserves_request_context():
    pool = BlockingRetrievalPool(1)
    started, release = Event(), Event()
    identity = ContextVar("test_identity")
    loop_thread = get_ident()

    def search():
        assert get_ident() != loop_thread
        assert identity.get() == "request-one"
        embedding_cache.get()["query"] = [1.0]
        retrieval_traces.get().append({"corpus": "knowledge"})
        started.set()
        assert release.wait(3)
        return ["answer"]

    async def run():
        identity.set("request-one")
        cache, traces = {}, []
        tokens = (embedding_cache.set(cache), retrieval_traces.set(traces))
        try:
            task = asyncio.create_task(pool.run(search))
            await wait_started(started)
            # This lightweight request executes while search is still blocked.
            async def health():
                return "ok"
            assert await asyncio.wait_for(health(), 0.5) == "ok"
            assert not task.done()
            release.set()
            assert await task == ["answer"]
            assert cache == {"query": [1.0]}
            assert traces == [{"corpus": "knowledge"}]
        finally:
            release.set()
            embedding_cache.reset(tokens[0])
            retrieval_traces.reset(tokens[1])

    try:
        asyncio.run(run())
    finally:
        release.set()
        pool.close()


def test_cancelled_request_keeps_worker_slot_until_real_work_finishes():
    pool = BlockingRetrievalPool(1)
    started, release, second_started = Event(), Event(), Event()

    def slow():
        started.set()
        assert release.wait(3)

    async def run():
        first = asyncio.create_task(pool.run(slow))
        await wait_started(started)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(pool.run(second_started.set))
        await asyncio.sleep(0.03)
        assert not second_started.is_set()
        # The second job waits before submit, not in the executor's unbounded queue.
        assert pool._executor._work_queue.qsize() == 0
        release.set()
        await asyncio.wait_for(second, 2)
        assert second_started.is_set()

    try:
        asyncio.run(run())
    finally:
        release.set()
        pool.close()


def test_cancel_while_waiting_never_submits_job_and_errors_release_capacity():
    pool = BlockingRetrievalPool(1)
    started, release, unwanted = Event(), Event(), Event()

    def slow():
        started.set()
        assert release.wait(3)

    def fail():
        raise ValueError("expected")

    async def run():
        first = asyncio.create_task(pool.run(slow))
        await wait_started(started)
        waiting = asyncio.create_task(pool.run(unwanted.set))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        release.set()
        await first
        with pytest.raises(ValueError, match="expected"):
            await pool.run(fail)
        assert await pool.run(lambda: 42) == 42
        assert not unwanted.is_set()

    try:
        asyncio.run(run())
    finally:
        release.set()
        pool.close()


@pytest.mark.parametrize("worker,task_type", [
    ("training_qa", "training_qa"),
    ("plan_adjust", "training_plan"),
    ("nutrition", "nutrition_advice"),
])
def test_real_graph_workers_offload_search(monkeypatch, worker, task_type):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    loop_thread = get_ident()
    calls = []

    def search(*args, **kwargs):
        calls.append(get_ident())
        return []  # Empty evidence follows deterministic refusal, no model call.

    monkeypatch.setattr(graph.rag, "search", search)
    async def run():
        result = await graph.workers[worker].ainvoke({
            "user_id": "test", "raw_message": "解释热身",
            "task_text": "解释热身", "task_type": task_type,
            "task_action": "question",
            "intent_result": {"name": "training_qa", "confidence": 1.0,
                              "action": "question", "safety_level": "normal"},
        })
        assert result["result"]["status"] == "insufficient_evidence"
    asyncio.run(run())
    assert len(calls) == 1 and calls[0] != loop_thread
