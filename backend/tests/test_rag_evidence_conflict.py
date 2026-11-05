"""Source-bound RAG evidence conflict handling (all providers mocked)."""
from __future__ import annotations

import asyncio
import copy
from unittest.mock import AsyncMock

import pytest

from app.services.llm.budget import ModelTaskBudgetError, model_scope
from app.services.rag.evidence_conflict import (
    EvidenceClaimModel,
    EvidenceConditionModel,
    EvidenceConflictModel,
    EvidenceCitationModel,
    analyze_evidence_conflicts,
    answer_system_rules,
    apply_evidence_notice,
    controlled_evidence_answer,
    evidence_notice,
    filter_hits_for_answer,
    finalize_evidence_answer,
)
from app.services.rag.retrieval import RetrievedChunk
from app.schemas.citation import citations_from_hits


QUERY = "训练频率怎么安排"


def _chunk(title, content, chunk_id, source=None, metadata=None):
    return RetrievedChunk(
        title=title,
        content=content,
        score=0.95,
        source=source or ("doc-" + chunk_id),
        metadata={"rerank_status": "model", "rerank_score": 0.95, **(metadata or {})},
        chunk_id=chunk_id,
    )


def _hits():
    return [
        _chunk("来源甲", "同条件下建议每周训练三天，逐步增加。", "chunk-a",
               metadata={"url": "https://example.org/a", "document_date": "2024-01-01"}),
        _chunk("来源乙", "同条件下建议每周训练五天，注意恢复。", "chunk-b",
               metadata={"url": "https://example.org/b"}),
    ]


def _same_source_hits():
    return [
        _chunk("同一文档上篇", "同条件下建议每周训练三天，逐步增加。", "same-a",
               source="knowledge://same-doc"),
        _chunk("同一文档下篇", "同条件下建议每周训练五天，注意恢复。", "same-b",
               source="knowledge://same-doc"),
    ]


def _conflict(*, claim_text="模型归纳的结论标签", irrelevant=None, chunk_ids=("chunk-a", "chunk-b"),
              quotes=("每周训练三天", "每周训练五天")):
    return EvidenceConflictModel(
        claims=[EvidenceClaimModel(
            claim=claim_text,
            relation="contradiction",
            evidence=[
                EvidenceCitationModel(chunk_id=chunk_ids[0], quote=quotes[0]),
                EvidenceCitationModel(chunk_id=chunk_ids[1], quote=quotes[1]),
            ],
        )],
        irrelevant_chunk_ids=list(irrelevant or []),
    )


def _consistent(irrelevant=None):
    return EvidenceConflictModel(
        claims=[EvidenceClaimModel(
            claim="模型归纳的一致标签",
            relation="consistent",
            evidence=[
                EvidenceCitationModel(chunk_id="chunk-a", quote="逐步增加"),
                EvidenceCitationModel(chunk_id="chunk-b", quote="注意恢复"),
            ],
        )],
        irrelevant_chunk_ids=list(irrelevant or []),
    )


class FakeLLM:
    def __init__(self, structured=None, structured_error=None, answer="可执行回答"):
        self.structured = structured
        self.structured_error = structured_error
        self.answer = answer
        self.structured_calls = []
        self.answer_calls = []

    async def complete_structured(self, messages, schema, *, name=None, temperature=0.0,
                                  validator=None, **kwargs):
        self.structured_calls.append({
            "messages": copy.deepcopy(messages),
            "name": name,
            "schema": schema,
        })
        if self.structured_error is not None:
            raise self.structured_error
        return self.structured

    async def complete_message(self, messages, **kwargs):
        self.answer_calls.append(copy.deepcopy(messages))
        return {"role": "assistant", "content": self.answer}

    async def complete(self, messages, **kwargs):
        self.answer_calls.append(copy.deepcopy(messages))
        return self.answer


def _run(coro_factory):
    async def run():
        with model_scope("rag-conflict-test", max_calls=20, max_tokens=50_000):
            return await coro_factory()
    return asyncio.run(run())


def _all_prompt_text(fake):
    return "\n".join(str(message.get("content") or "")
                     for call in fake.answer_calls for message in call)


def test_same_source_contradiction_is_analysed_and_controlled():
    fake = FakeLLM(
        structured=_conflict(chunk_ids=("same-a", "same-b")),
        answer="只采来源甲：每周三天是唯一正确答案")
    analysis = _run(lambda: analyze_evidence_conflicts(fake, QUERY, _same_source_hits()))
    assert analysis.status == "analyzed"
    assert analysis.has_unresolved_conflict
    answer = finalize_evidence_answer("模型正文", analysis)
    assert "《同一文档上篇》原文：“每周训练三天”" in answer
    assert "《同一文档下篇》原文：“每周训练五天”" in answer
    assert "无法判定唯一结论" in answer and "结论不确定" not in answer
    assert "只采来源甲" not in answer
    assert "哪一方正确" in answer
    assert len(fake.structured_calls) == 1
    assert "same-a" in fake.structured_calls[0]["messages"][1]["content"]
    assert "same-b" in fake.structured_calls[0]["messages"][1]["content"]


def test_different_source_conflict_uses_verified_quotes_and_public_metadata():
    analysis = _run(lambda: analyze_evidence_conflicts(
        FakeLLM(structured=_conflict()), QUERY, _hits()))
    answer = finalize_evidence_answer("模型正文", analysis)
    assert "《来源甲》原文：“每周训练三天”" in answer
    assert "《来源乙》原文：“每周训练五天”" in answer
    citations = citations_from_hits(_hits())
    assert {item["chunk_id"] for item in citations} == {"chunk-a", "chunk-b"}
    assert {item["url"] for item in citations} == {
        "https://example.org/a", "https://example.org/b"}


def test_exact_duplicate_content_is_not_called_and_not_voted():
    duplicate = _chunk("来源乙副本", "同条件下建议每周训练三天，逐步增加。", "chunk-a-copy",
                       source="doc-copy")
    fake = FakeLLM(structured=_conflict())
    analysis = _run(lambda: analyze_evidence_conflicts(fake, QUERY, [_hits()[0], duplicate]))
    assert analysis.status == "not_applicable"
    assert analysis.reason == "fewer_than_two_distinct_chunks"
    assert fake.structured_calls == []
    assert finalize_evidence_answer("原有回答", analysis) == "原有回答"


def test_condition_difference_never_exposes_model_rewrite():
    fabricated = "模型虚构的条件解释：初学者必须每周三次"
    result = EvidenceConflictModel(claims=[EvidenceClaimModel(
        claim="模型虚构的结论",
        relation="condition_difference",
        conditions=[EvidenceConditionModel(
            text=fabricated, chunk_id="chunk-a", quote="初学者", kind="applicability"),
            EvidenceConditionModel(text=fabricated, chunk_id="chunk-b", quote="进阶者", kind="applicability")],
        evidence=[
            EvidenceCitationModel(chunk_id="chunk-a", quote="每周训练三天"),
            EvidenceCitationModel(chunk_id="chunk-b", quote="每周训练五天"),
        ],
    )])
    analysis = _run(lambda: analyze_evidence_conflicts(
        FakeLLM(structured=result), QUERY, [
            _chunk('来源甲','初学者每周训练三天。','chunk-a'),
            _chunk('来源乙','进阶者每周训练五天。','chunk-b')]))
    answer = finalize_evidence_answer("模型正文", analysis)
    notice = evidence_notice(analysis)
    rules = answer_system_rules(analysis)
    assert "《来源甲》条件原文：“初学者”" in answer
    assert fabricated not in answer and fabricated not in notice and fabricated not in rules
    assert "模型虚构的结论" not in answer and "模型虚构的结论" not in rules
    assert "不能合并为同一条件下的结论" in answer


def test_claim_text_is_never_presented_as_verified_fact():
    fabricated = "模型虚构的结论：必须选来源甲"
    analysis = _run(lambda: analyze_evidence_conflicts(
        FakeLLM(structured=_conflict(claim_text=fabricated)), QUERY, _hits()))
    answer = finalize_evidence_answer("模型正文", analysis)
    notice = evidence_notice(analysis)
    rules = answer_system_rules(analysis)
    assert fabricated not in answer and fabricated not in notice and fabricated not in rules
    assert "每周训练三天" in answer and "每周训练五天" in answer
    assert "chunk-a" in rules and "chunk-b" in rules


def test_time_or_version_difference_uses_only_real_metadata():
    result = EvidenceConflictModel(claims=[EvidenceClaimModel(
        claim="模型虚构的时间解释",
        relation="time_or_version_difference",
        evidence=[
            EvidenceCitationModel(chunk_id="chunk-a", quote="每周训练三天"),
            EvidenceCitationModel(chunk_id="chunk-b", quote="每周训练五天"),
        ],
    )])
    fake = FakeLLM(structured=result)
    analysis = _run(lambda: analyze_evidence_conflicts(fake, QUERY, _hits()))
    answer = finalize_evidence_answer("模型正文", analysis)
    assert "日期 2024-01-01" in answer
    assert "来源乙》未注明可核实的日期或版本" in answer
    assert "版本 20" not in answer and "v2" not in answer
    assert "模型虚构的时间解释" not in answer
    prompt = fake.structured_calls[0]["messages"][1]["content"]
    assert "2024-01-01" in prompt
    assert "retrieved_at" not in prompt


def test_filtered_noise_is_removed_from_answer_prompt_and_citations(monkeypatch):
    from app.services.agent import react_readonly_workers as workers
    from app.services.agent.agent import agent
    from app.services.rag import online

    noise = _chunk("无关噪声", "与训练频率无关的拉伸动作说明。", "noise")
    hits = [*_hits(), noise]
    fake = FakeLLM(structured=_consistent(irrelevant=["noise"]), answer="正常回答")
    monkeypatch.setattr(agent.graph, "llm", fake)
    monkeypatch.setattr(agent.graph, "rag", object())

    async def fake_search(*args, **kwargs):
        return hits

    monkeypatch.setattr(online, "search", fake_search)

    async def call():
        with model_scope("react-filter", max_calls=20, max_tokens=50_000):
            return await workers.answer_training(
                {"query": QUERY}, {"user_id": "demo-user", "request_id": "react-filter"})

    result = asyncio.run(call()).to_dict()
    assert result["status"] == "completed"
    assert result["summary"] == "正常回答"
    assert {item["chunk_id"] for item in result["result"]["citations"]} == {
        "chunk-a", "chunk-b"}
    prompt = _all_prompt_text(fake)
    assert "与训练频率无关的拉伸动作说明" not in prompt
    assert "noise" not in prompt


def test_filtered_noise_is_removed_from_legacy_fallback_and_citations(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    noise = _chunk("无关噪声", "与训练频率无关的拉伸动作说明。", "noise")
    hits = [noise, *_hits()]
    graph = FitnessMultiAgentGraph()
    fake = FakeLLM(structured=_consistent(irrelevant=["noise"]), answer="")
    monkeypatch.setattr(graph, "llm", fake)
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=hits))
    complete_with_tools = AsyncMock(return_value=None)
    monkeypatch.setattr(graph.mcp, "complete_with_tools", complete_with_tools)

    async def call():
        with model_scope("legacy-filter", max_calls=20, max_tokens=50_000):
            return await graph.workers["training_qa"].ainvoke({
                "user_id": "demo-user", "raw_message": QUERY,
                "task_text": QUERY, "task_type": "training_qa",
                "task_action": "question", "intent_result": {},
            })

    result = asyncio.run(call())["result"]
    assert "与训练频率无关的拉伸动作说明" not in result["answer"]
    assert {item["chunk_id"] for item in result["citations"]} == {
        "chunk-a", "chunk-b"}
    prompt = complete_with_tools.await_args.args[1]
    assert all("与训练频率无关" not in str(message.get("content")) for message in prompt)


def test_filtered_noise_is_removed_from_execution_prompt_and_citations(monkeypatch):
    from app.services.agent.execution import ChatExecutionMixin
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.schemas.chat import ChatRequest

    graph = FitnessMultiAgentGraph()
    request = ChatRequest(user_id="demo-user", conversation_id="filter",
                          request_id="exec-filter", message="蛋白质怎么补充")
    noise = _chunk("无关噪声", "与蛋白质无关的拉伸动作说明。", "noise")
    hits = [noise, *_hits()]
    fake = FakeLLM(structured=_consistent(irrelevant=["noise"]), answer="营养回答")
    graph.llm = fake

    async def fake_context(*args, **kwargs):
        return {}

    async def fake_search(*args, **kwargs):
        return hits

    graph._load_context_node = fake_context
    monkeypatch.setattr("app.services.rag.online.search", fake_search)

    class Harness(ChatExecutionMixin):
        def __init__(self, graph):
            self.graph = graph

        @staticmethod
        def _last_worker_result(result):
            return {}

    harness = Harness(graph)

    async def call():
        with model_scope("exec-filter", max_calls=20, max_tokens=50_000):
            return await harness._complete_nutrition_advice(request, {}, {})

    answer, used, citations = asyncio.run(call())
    assert used is True and answer == "营养回答"
    assert {item["chunk_id"] for item in citations} == {"chunk-a", "chunk-b"}
    prompt = _all_prompt_text(fake)
    assert "与蛋白质无关的拉伸动作说明" not in prompt
    assert "noise" not in prompt


def test_filtered_duplicate_noise_content_is_removed_without_voting():
    noise_one = _chunk("噪声一", "完全相同的无关拉伸说明。", "noise-1", source="doc-n1")
    noise_two = _chunk("噪声二", "完全相同的无关拉伸说明。", "noise-2", source="doc-n2")
    hits = [*_hits(), noise_one, noise_two]
    analysis = _run(lambda: analyze_evidence_conflicts(
        FakeLLM(structured=_consistent(irrelevant=["noise-1"])), QUERY, hits))
    filtered = filter_hits_for_answer(hits, analysis)
    assert filtered.applied is True
    assert {hit.chunk_id for hit in filtered.hits} == {"chunk-a", "chunk-b"}
    assert filtered.excluded_chunk_ids == ("noise-1",)


def _seven_hits_with_irrelevant_head():
    hits = [
        *[_chunk(f"无关片段{i}", f"与问题无关的内容{i}。", f"a{i}", source=f"s{i}")
          for i in range(1, 7)],
        _chunk("尾部", "同条件下建议每周训练三天，逐步增加。", "tail", source="tail-doc"),
    ]
    return hits


def _all_head_irrelevant():
    return EvidenceConflictModel(
        claims=[], irrelevant_chunk_ids=[f"a{i}" for i in range(1, 7)])


def test_react_all_irrelevant_without_tail_returns_insufficient(monkeypatch):
    from app.services.agent import react_readonly_workers as workers
    from app.services.agent.agent import agent
    from app.services.rag import online

    hits = [
        _chunk("一", "无关内容一。", "a1", source="s1"),
        _chunk("二", "无关内容二。", "a2", source="s2"),
        _chunk("三", "无关内容三。", "a3", source="s3"),
    ]
    fake = FakeLLM(
        structured=EvidenceConflictModel(
            claims=[], irrelevant_chunk_ids=["a1", "a2", "a3"]),
        answer="不应出现")
    monkeypatch.setattr(agent.graph, "llm", fake)
    monkeypatch.setattr(agent.graph, "rag", object())

    async def fake_search(*args, **kwargs):
        return hits

    monkeypatch.setattr(online, "search", fake_search)

    async def call():
        with model_scope("react-all-irrelevant", max_calls=20, max_tokens=50_000):
            return await workers.answer_training(
                {"query": QUERY}, {"user_id": "demo-user", "request_id": "react-all-irrelevant"})

    result = asyncio.run(call()).to_dict()
    assert result["status"] == "rejected"
    assert result["error"] == "insufficient_evidence"
    assert "不应出现" not in (result.get("summary") or "")
    assert fake.answer_calls == []


def test_react_all_irrelevant_with_tail_uses_only_tail(monkeypatch):
    from app.services.agent import react_readonly_workers as workers
    from app.services.agent.agent import agent
    from app.services.rag import online

    hits = _seven_hits_with_irrelevant_head()
    fake = FakeLLM(structured=_all_head_irrelevant(), answer="尾部回答")
    monkeypatch.setattr(agent.graph, "llm", fake)
    monkeypatch.setattr(agent.graph, "rag", object())

    async def fake_search(*args, **kwargs):
        return hits

    monkeypatch.setattr(online, "search", fake_search)

    async def call():
        with model_scope("react-tail", max_calls=20, max_tokens=50_000):
            return await workers.answer_training(
                {"query": QUERY}, {"user_id": "demo-user", "request_id": "react-tail"})

    result = asyncio.run(call()).to_dict()
    assert result["status"] == "completed"
    assert result["summary"] == "尾部回答"
    assert {item["chunk_id"] for item in result["result"]["citations"]} == {"tail"}
    prompt = _all_prompt_text(fake)
    assert "同条件下建议每周训练三天" in prompt
    assert all(f"无关内容{i}" not in prompt for i in range(1, 7))


def test_react_all_irrelevant_tail_failing_quality_returns_insufficient(monkeypatch):
    from app.services.agent import react_readonly_workers as workers
    from app.services.agent.agent import agent
    from app.services.rag import online

    hits = _seven_hits_with_irrelevant_head()
    hits[-1] = _chunk(
        "弱尾部", "同条件下建议每周训练三天，逐步增加。", "tail", source="tail-doc",
        metadata={"rerank_score": 0.01})
    fake = FakeLLM(structured=_all_head_irrelevant(), answer="不应出现")
    monkeypatch.setattr(agent.graph, "llm", fake)
    monkeypatch.setattr(agent.graph, "rag", object())

    async def fake_search(*args, **kwargs):
        return hits

    monkeypatch.setattr(online, "search", fake_search)

    async def call():
        with model_scope("react-weak-tail", max_calls=20, max_tokens=50_000):
            return await workers.answer_training(
                {"query": QUERY}, {"user_id": "demo-user", "request_id": "react-weak-tail"})

    result = asyncio.run(call()).to_dict()
    assert result["status"] == "rejected"
    assert result["error"] == "insufficient_evidence"
    assert "不应出现" not in (result.get("summary") or "")
    assert fake.answer_calls == []


def test_legacy_all_irrelevant_without_tail_returns_insufficient(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    hits = [
        _chunk("一", "无关内容一。", "a1", source="s1"),
        _chunk("二", "无关内容二。", "a2", source="s2"),
        _chunk("三", "无关内容三。", "a3", source="s3"),
    ]
    graph = FitnessMultiAgentGraph()
    fake = FakeLLM(
        structured=EvidenceConflictModel(
            claims=[], irrelevant_chunk_ids=["a1", "a2", "a3"]),
        answer="不应出现")
    monkeypatch.setattr(graph, "llm", fake)
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=hits))
    complete_with_tools = AsyncMock(return_value={"content": "不应出现"})
    monkeypatch.setattr(graph.mcp, "complete_with_tools", complete_with_tools)

    async def call():
        with model_scope("legacy-all-irrelevant", max_calls=20, max_tokens=50_000):
            return await graph.workers["training_qa"].ainvoke({
                "user_id": "demo-user", "raw_message": QUERY,
                "task_text": QUERY, "task_type": "training_qa",
                "task_action": "question", "intent_result": {},
            })

    result = asyncio.run(call())["result"]
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert "不应出现" not in result["answer"]
    assert fake.answer_calls == []
    assert complete_with_tools.await_count == 0


def test_legacy_plan_all_irrelevant_without_tail_returns_insufficient(monkeypatch):
    from app.services.agent import task_information
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    hits = [
        _chunk("一", "无关内容一。", "a1", source="s1"),
        _chunk("二", "无关内容二。", "a2", source="s2"),
        _chunk("三", "无关内容三。", "a3", source="s3"),
    ]
    graph = FitnessMultiAgentGraph()
    fake = FakeLLM(
        structured=EvidenceConflictModel(
            claims=[], irrelevant_chunk_ids=["a1", "a2", "a3"]),
        answer="不应出现")
    monkeypatch.setattr(graph, "llm", fake)
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=hits))
    complete_with_tools = AsyncMock(return_value={"content": "不应出现"})
    monkeypatch.setattr(graph.mcp, "complete_with_tools", complete_with_tools)

    async def fake_information(*args, **kwargs):
        return {}, None

    monkeypatch.setattr(task_information, "check_plan_information", fake_information)

    async def call():
        with model_scope("legacy-plan-irrelevant", max_calls=20, max_tokens=50_000):
            return await graph.workers["plan_adjust"].ainvoke({
                "user_id": "demo-user", "raw_message": QUERY,
                "task_text": QUERY, "task_type": "training_plan",
                "task_action": "question", "intent_result": {},
            })

    result = asyncio.run(call())["result"]
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert "不应出现" not in result["answer"]
    assert fake.answer_calls == []
    assert complete_with_tools.await_count == 0


def test_legacy_all_irrelevant_with_tail_uses_only_tail(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    hits = _seven_hits_with_irrelevant_head()
    graph = FitnessMultiAgentGraph()
    fake = FakeLLM(structured=_all_head_irrelevant(), answer="不应出现")
    monkeypatch.setattr(graph, "llm", fake)
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=hits))
    complete_with_tools = AsyncMock(return_value={"content": "尾部回答"})
    monkeypatch.setattr(graph.mcp, "complete_with_tools", complete_with_tools)

    async def call():
        with model_scope("legacy-tail", max_calls=20, max_tokens=50_000):
            return await graph.workers["training_qa"].ainvoke({
                "user_id": "demo-user", "raw_message": QUERY,
                "task_text": QUERY, "task_type": "training_qa",
                "task_action": "question", "intent_result": {},
            })

    result = asyncio.run(call())["result"]
    assert result["status"] == "complete"
    assert {item["chunk_id"] for item in result["citations"]} == {"tail"}
    prompt = complete_with_tools.await_args.args[1]
    text = "\n".join(str(message.get("content") or "") for message in prompt)
    assert "同条件下建议每周训练三天" in text
    assert all(f"无关内容{i}" not in text for i in range(1, 7))


def test_execution_all_irrelevant_without_tail_returns_insufficient(monkeypatch):
    from app.services.agent.execution import ChatExecutionMixin
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.schemas.chat import ChatRequest

    graph = FitnessMultiAgentGraph()
    request = ChatRequest(user_id="demo-user", conversation_id="all-irrelevant",
                          request_id="exec-all-irrelevant", message="蛋白质怎么补充")
    hits = [
        _chunk("一", "无关内容一。", "a1", source="s1"),
        _chunk("二", "无关内容二。", "a2", source="s2"),
        _chunk("三", "无关内容三。", "a3", source="s3"),
    ]
    fake = FakeLLM(
        structured=EvidenceConflictModel(
            claims=[], irrelevant_chunk_ids=["a1", "a2", "a3"]),
        answer="不应出现")
    graph.llm = fake

    async def fake_context(*args, **kwargs):
        return {}

    async def fake_search(*args, **kwargs):
        return hits

    graph._load_context_node = fake_context
    monkeypatch.setattr("app.services.rag.online.search", fake_search)

    class Harness(ChatExecutionMixin):
        def __init__(self, graph):
            self.graph = graph

        @staticmethod
        def _last_worker_result(result):
            return {}

    harness = Harness(graph)

    async def call():
        with model_scope("exec-all-irrelevant", max_calls=20, max_tokens=50_000):
            return await harness._complete_nutrition_advice(request, {}, {})

    answer, used, citations = asyncio.run(call())
    assert used is False and citations == []
    assert "不应出现" not in answer
    assert "不足以支持" in answer
    assert fake.answer_calls == []


def test_execution_all_irrelevant_with_tail_uses_only_tail(monkeypatch):
    from app.services.agent.execution import ChatExecutionMixin
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.schemas.chat import ChatRequest

    graph = FitnessMultiAgentGraph()
    request = ChatRequest(user_id="demo-user", conversation_id="exec-tail",
                          request_id="exec-tail", message="蛋白质怎么补充")
    hits = _seven_hits_with_irrelevant_head()
    fake = FakeLLM(structured=_all_head_irrelevant(), answer="尾部回答")
    graph.llm = fake

    async def fake_context(*args, **kwargs):
        return {}

    async def fake_search(*args, **kwargs):
        return hits

    graph._load_context_node = fake_context
    monkeypatch.setattr("app.services.rag.online.search", fake_search)

    class Harness(ChatExecutionMixin):
        def __init__(self, graph):
            self.graph = graph

        @staticmethod
        def _last_worker_result(result):
            return {}

    harness = Harness(graph)

    async def call():
        with model_scope("exec-tail", max_calls=20, max_tokens=50_000):
            return await harness._complete_nutrition_advice(request, {}, {})

    answer, used, citations = asyncio.run(call())
    assert used is True and answer == "尾部回答"
    assert {item["chunk_id"] for item in citations} == {"tail"}
    prompt = _all_prompt_text(fake)
    assert "同条件下建议每周训练三天" in prompt
    assert all(f"无关内容{i}" not in prompt for i in range(1, 7))


def test_all_irrelevant_without_tail_filters_to_empty():
    hits = [
        _chunk("一", "无关内容一。", "a1", source="s1"),
        _chunk("二", "无关内容二。", "a2", source="s2"),
        _chunk("三", "无关内容三。", "a3", source="s3"),
    ]
    analysis = _run(lambda: analyze_evidence_conflicts(
        FakeLLM(structured=EvidenceConflictModel(
            claims=[], irrelevant_chunk_ids=["a1", "a2", "a3"])),
        QUERY, hits))
    filtered = filter_hits_for_answer(hits, analysis)
    assert filtered.applied is True
    assert filtered.hits == ()
    assert filtered.reason == "applied_no_remaining"
    assert set(filtered.excluded_chunk_ids) == {"a1", "a2", "a3"}


def test_all_irrelevant_with_unanalysed_tail_keeps_only_tail():
    hits = _seven_hits_with_irrelevant_head()
    analysis = _run(lambda: analyze_evidence_conflicts(
        FakeLLM(structured=_all_head_irrelevant()), QUERY, hits))
    assert analysis.status == "analyzed"
    assert analysis.irrelevant_chunk_ids == tuple(f"a{i}" for i in range(1, 7))
    filtered = filter_hits_for_answer(hits, analysis)
    assert filtered.applied is True
    assert {hit.chunk_id for hit in filtered.hits} == {"tail"}
    assert filtered.reason == "applied_with_unanalysed_tail"


def test_analysis_failure_keeps_all_hits_and_never_claims_exclusion(monkeypatch):
    from app.services.agent import react_readonly_workers as workers
    from app.services.agent.agent import agent
    from app.services.rag import online

    noise = _chunk("无关噪声", "与训练频率无关的拉伸动作说明。", "noise")
    hits = [*_hits(), noise]
    fake = FakeLLM(structured_error=RuntimeError("synthetic failure"), answer="普通回答")
    monkeypatch.setattr(agent.graph, "llm", fake)
    monkeypatch.setattr(agent.graph, "rag", object())

    async def fake_search(*args, **kwargs):
        return hits

    monkeypatch.setattr(online, "search", fake_search)

    async def call():
        with model_scope("react-degraded", max_calls=20, max_tokens=50_000):
            return await workers.answer_training(
                {"query": QUERY}, {"user_id": "demo-user", "request_id": "react-degraded"})

    result = asyncio.run(call()).to_dict()
    assert result["status"] == "completed"
    assert result["summary"] == "普通回答"
    assert "已排除" not in result["summary"]
    assert {item["chunk_id"] for item in result["result"]["citations"]} == {
        "chunk-a", "chunk-b", "noise"}
    prompt = _all_prompt_text(fake)
    assert "与训练频率无关的拉伸动作说明" in prompt
    assert "本次证据辨析不可用" in prompt


def test_fabricated_ids_and_wrong_quotes_cannot_enter_results():
    result = EvidenceConflictModel(claims=[
        EvidenceClaimModel(
            claim="虚构冲突",
            relation="contradiction",
            evidence=[
                EvidenceCitationModel(chunk_id="chunk-a", quote="每周训练三天"),
                EvidenceCitationModel(chunk_id="fake-chunk", quote="伪造原文"),
                EvidenceCitationModel(chunk_id="chunk-b", quote="每周训练四天"),
            ],
        ),
        EvidenceClaimModel(
            claim="可核实的一致主张",
            relation="consistent",
            evidence=[EvidenceCitationModel(chunk_id="chunk-b", quote="注意恢复")],
        ),
    ])
    fake = FakeLLM(structured=result)
    analysis = _run(lambda: analyze_evidence_conflicts(fake, QUERY, _hits()))
    assert analysis.status == "partial"
    assert analysis.invalid_references == 3
    assert not analysis.has_unresolved_conflict
    assert analysis.claims[0].relation == "consistent"
    assert all(item.chunk_id != "fake-chunk" for item in analysis.claims[0].evidence)
    notice = evidence_notice(analysis)
    assert "未用于证据判断" in notice and "未据此排除检索片段" in notice
    assert "fake-chunk" not in notice and "伪造原文" not in notice
    assert "fake-chunk" not in answer_system_rules(analysis)
    filtered = filter_hits_for_answer(_hits(), analysis)
    assert filtered.applied is False
    assert len(filtered.hits) == 2


def test_all_invalid_references_degrade_without_claiming_consistency():
    result = EvidenceConflictModel(claims=[EvidenceClaimModel(
        claim="无法核实",
        relation="consistent",
        evidence=[EvidenceCitationModel(chunk_id="fake-chunk", quote="伪造原文")],
    )])
    analysis = _run(lambda: analyze_evidence_conflicts(FakeLLM(structured=result), QUERY, _hits()))
    assert analysis.status == "degraded"
    assert analysis.reason == "no_valid_claim_evidence"
    assert evidence_notice(analysis) == ""
    rules = answer_system_rules(analysis)
    assert "本次证据辨析不可用" in rules and "不得声称不同来源已经一致" in rules
    assert "fake-chunk" not in rules
    assert finalize_evidence_answer("模型正文", analysis) == "模型正文"


def test_model_exception_degrades_instead_of_claiming_consensus():
    analysis = _run(lambda: analyze_evidence_conflicts(
        FakeLLM(structured_error=RuntimeError("synthetic provider failure")), QUERY, _hits()))
    assert analysis.status == "degraded"
    assert analysis.reason == "model_or_protocol_failure"
    assert evidence_notice(analysis) == ""
    rules = answer_system_rules(analysis)
    assert "本次证据辨析不可用" in rules
    assert "不得声称不同来源已经一致" in rules
    filtered = filter_hits_for_answer(_hits(), analysis)
    assert filtered.applied is False
    assert filtered.hits == tuple(_hits())
    assert filtered.reason == "analysis_not_integrity_ok"


def test_budget_stop_propagates_and_is_not_swallowed():
    fake = FakeLLM(structured_error=ModelTaskBudgetError("model_task_budget_exhausted"))
    with pytest.raises(ModelTaskBudgetError):
        _run(lambda: analyze_evidence_conflicts(fake, QUERY, _hits()))


def test_no_scope_skips_the_paid_call():
    from app.services.llm.budget import active_scope

    fake = FakeLLM(structured=_conflict())
    token = active_scope.set(None)
    try:
        analysis = asyncio.run(analyze_evidence_conflicts(fake, QUERY, _hits()))
    finally:
        active_scope.reset(token)
    assert analysis.status == "not_applicable"
    assert analysis.reason == "model_budget_scope_absent"
    assert fake.structured_calls == []


def test_schema_has_no_self_reported_confidence_or_probability():
    schema = EvidenceConflictModel.model_json_schema()
    text = str(schema).lower()
    assert "confidence" not in text
    assert "probability" not in text


def test_normal_consistent_answer_is_unchanged():
    analysis = _run(lambda: analyze_evidence_conflicts(
        FakeLLM(structured=_consistent()), QUERY, _hits()))
    assert analysis.status == "analyzed"
    assert apply_evidence_notice("原有回答", analysis) == "原有回答"
    assert answer_system_rules(analysis) == ""
    assert finalize_evidence_answer("原有回答", analysis) == "原有回答"


@pytest.mark.parametrize("worker_name", ["answer_training", "advise_nutrition"])
def test_react_conflict_entries_use_controlled_answer_not_model_text(monkeypatch, worker_name):
    from app.services.agent import react_readonly_workers as workers
    from app.services.agent.agent import agent
    from app.services.rag import online

    fake = FakeLLM(
        structured=_conflict(chunk_ids=("same-a", "same-b")),
        answer="只采来源甲：唯一正确答案")
    monkeypatch.setattr(agent.graph, "llm", fake)
    monkeypatch.setattr(agent.graph, "rag", object())

    async def fake_search(*args, **kwargs):
        return _same_source_hits()

    monkeypatch.setattr(online, "search", fake_search)

    async def call():
        with model_scope("react-conflict-" + worker_name, max_calls=20, max_tokens=50_000):
            return await getattr(workers, worker_name)(
                {"query": QUERY}, {"user_id": "demo-user", "request_id": "react-conflict"})

    result = asyncio.run(call()).to_dict()
    assert result["status"] == "completed"
    assert "无法判定唯一结论" in result["summary"]
    assert "只采来源甲" not in result["summary"]
    assert {item["chunk_id"] for item in result["result"]["citations"]} == {
        "same-a", "same-b"}
    assert fake.answer_calls == []


def test_react_condition_entry_does_not_let_model_merge_sides(monkeypatch):
    from app.services.agent import react_readonly_workers as workers
    from app.services.agent.agent import agent
    from app.services.rag import online

    result = EvidenceConflictModel(claims=[EvidenceClaimModel(
        claim="模型虚构的条件结论",
        relation="condition_difference",
        conditions=[EvidenceConditionModel(
            text="模型虚构条件", chunk_id="chunk-a", quote="初学者", kind="applicability"),
            EvidenceConditionModel(text="模型虚构条件",chunk_id="chunk-b",quote="进阶者",kind="applicability")],
        evidence=[
            EvidenceCitationModel(chunk_id="chunk-a", quote="每周训练三天"),
            EvidenceCitationModel(chunk_id="chunk-b", quote="每周训练五天"),
        ],
    )])
    fake = FakeLLM(structured=result, answer="合并为唯一结论：每周四天")
    monkeypatch.setattr(agent.graph, "llm", fake)
    monkeypatch.setattr(agent.graph, "rag", object())

    async def fake_search(*args, **kwargs):
        return [_chunk('来源甲','初学者每周训练三天。','chunk-a'),
                _chunk('来源乙','进阶者每周训练五天。','chunk-b')]

    monkeypatch.setattr(online, "search", fake_search)

    async def call():
        with model_scope("react-condition", max_calls=20, max_tokens=50_000):
            return await workers.answer_training(
                {"query": QUERY}, {"user_id": "demo-user", "request_id": "react-condition"})

    answer = asyncio.run(call()).to_dict()["summary"]
    assert "《来源甲》条件原文：“初学者”" in answer
    assert "模型虚构条件" not in answer
    assert "合并为唯一结论" not in answer
    assert fake.answer_calls == []


def test_react_plan_entry_uses_controlled_answer_for_conflict(monkeypatch):
    from app.services.agent import react_readonly_workers as workers
    from app.services.agent import task_information
    from app.services.agent.agent import agent
    from app.services.rag import online

    fake = FakeLLM(
        structured=_conflict(chunk_ids=("same-a", "same-b")),
        answer="模型单方计划")
    monkeypatch.setattr(agent.graph, "llm", fake)
    monkeypatch.setattr(agent.graph, "rag", object())

    async def fake_search(*args, **kwargs):
        return _same_source_hits()

    async def fake_information(*args, **kwargs):
        return {}, None

    monkeypatch.setattr(online, "search", fake_search)
    monkeypatch.setattr(task_information, "check_plan_information", fake_information)

    async def call():
        with model_scope("react-plan-conflict", max_calls=20, max_tokens=50_000):
            return await workers.plan_training(
                {"query": QUERY},
                {"user_id": "demo-user", "request_id": "react-plan-conflict",
                 "source_text": QUERY, "semantic_scope": {}})

    result = asyncio.run(call()).to_dict()
    assert result["status"] == "completed"
    assert "无法判定唯一结论" in result["summary"]
    assert "模型单方计划" not in result["summary"]
    assert {item["chunk_id"] for item in result["result"]["citations"]} == {
        "same-a", "same-b"}
    assert fake.answer_calls == []


def test_legacy_qa_nutrition_and_plan_entries_use_controlled_answer(monkeypatch):
    from app.services.agent import task_information
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    for worker, task_type, message in (
            ("training_qa", "training_qa", QUERY),
            ("nutrition", "nutrition_advice", "蛋白质怎么补充"),
            ("plan_adjust", "training_plan", QUERY)):
        graph = FitnessMultiAgentGraph()
        fake = FakeLLM(
            structured=_conflict(chunk_ids=("same-a", "same-b")),
            answer="模型单方回答")
        monkeypatch.setattr(graph, "llm", fake)
        monkeypatch.setattr(graph, "_readonly_retrieval",
                            AsyncMock(return_value=_same_source_hits()))
        complete_with_tools = AsyncMock(return_value={"content": "模型单方回答"})
        monkeypatch.setattr(graph.mcp, "complete_with_tools", complete_with_tools)

        async def fake_information(*args, **kwargs):
            return {}, None

        monkeypatch.setattr(task_information, "check_plan_information", fake_information)

        async def call():
            with model_scope("legacy-" + worker, max_calls=20, max_tokens=50_000):
                return await graph.workers[worker].ainvoke({
                    "user_id": "demo-user", "raw_message": message,
                    "task_text": message, "task_type": task_type,
                    "task_action": "question", "intent_result": {},
                })

        result = asyncio.run(call())["result"]
        assert "无法判定唯一结论" in result["answer"]
        assert "模型单方回答" not in result["answer"]
        assert {item["chunk_id"] for item in result["citations"]} == {
            "same-a", "same-b"}
        assert fake.answer_calls == []
        assert complete_with_tools.await_count == 0


def test_execution_conflict_uses_controlled_answer(monkeypatch):
    from app.services.agent.execution import ChatExecutionMixin
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.schemas.chat import ChatRequest

    graph = FitnessMultiAgentGraph()
    request = ChatRequest(user_id="demo-user", conversation_id="conflict",
                          request_id="exec-conflict", message="蛋白质怎么补充")
    fake = FakeLLM(
        structured=_conflict(chunk_ids=("same-a", "same-b")),
        answer="模型单方营养结论")
    graph.llm = fake

    async def fake_context(*args, **kwargs):
        return {}

    async def fake_search(*args, **kwargs):
        return _same_source_hits()

    graph._load_context_node = fake_context
    monkeypatch.setattr("app.services.rag.online.search", fake_search)

    class Harness(ChatExecutionMixin):
        def __init__(self, graph):
            self.graph = graph

        @staticmethod
        def _last_worker_result(result):
            return {}

    harness = Harness(graph)

    async def call():
        with model_scope("exec-conflict", max_calls=20, max_tokens=50_000):
            return await harness._complete_nutrition_advice(request, {}, {})

    answer, used, citations = asyncio.run(call())
    assert used is False
    assert "无法判定唯一结论" in answer
    assert "模型单方营养结论" not in answer
    assert {item["chunk_id"] for item in citations} == {"same-a", "same-b"}
    assert fake.answer_calls == []


def test_execution_budget_stop_still_returns_limited(monkeypatch):
    from app.services.agent.execution import ChatExecutionMixin
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.schemas.chat import ChatRequest

    graph = FitnessMultiAgentGraph()
    request = ChatRequest(user_id="demo-user", conversation_id="limited",
                          request_id="exec-limited", message="蛋白质怎么补充")
    graph.llm = FakeLLM(structured_error=ModelTaskBudgetError("model_task_budget_exhausted"))

    async def fake_context(*args, **kwargs):
        return {}

    async def fake_search(*args, **kwargs):
        return _same_source_hits()

    graph._load_context_node = fake_context
    monkeypatch.setattr("app.services.rag.online.search", fake_search)

    class Harness(ChatExecutionMixin):
        def __init__(self, graph):
            self.graph = graph

        @staticmethod
        def _last_worker_result(result):
            return {}

    harness = Harness(graph)

    async def call():
        with model_scope("exec-limited", max_calls=20, max_tokens=50_000):
            return await harness._complete_nutrition_advice(request, {}, {})

    answer, used, citations = asyncio.run(call())
    assert used is False and citations == []
    assert "已达到模型调用上限" in answer


@pytest.mark.parametrize("copies", [0, 1, 2])
def test_application_notice_footer_is_idempotent_and_does_not_rewrite_body(copies):
    from app.services.rag.evidence_conflict import EvidenceAnalysis, EvidenceClaim
    analysis = EvidenceAnalysis(status="analyzed", claims=(
        EvidenceClaim(claim="", relation="insufficient", conditions=(), evidence=()),))
    notice = evidence_notice(analysis)
    assert notice
    body = "具体食物与条件。正文引用：\n> " + notice.strip() + "（引用结束）"
    answer = body + notice * copies
    finalized = apply_evidence_notice(answer, analysis)
    assert finalized == body + notice
    assert apply_evidence_notice(finalized, analysis) == finalized
    # A different application's notice must not be deleted based on its prefix.
    other = "\n\n证据说明：不同的核验结果。"
    assert apply_evidence_notice(body + other, analysis) == body + other + notice
