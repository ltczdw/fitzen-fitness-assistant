"""Coverage measurements use the same retrieval and evidence gate as chat."""
import asyncio
import json

from app.services.agent.intent import IntentResult
from app.services.llm.jobs import model_job
from app.services.rag.pipeline import HybridRAGPipeline
from app.services.rag.quality import assess_rag_evidence
from app.services.rag.retrieval import retrieval_traces, embedding_cache


async def prefill_query_embeddings(queries):
    from app.core.config import settings
    from app.services.rag.embedding_client import embedding_client
    vectors = await embedding_client.aembed(queries)
    return {(query, settings.retrieval_embedding_model, settings.retrieval_embedding_index_version): vector
            for query, vector in zip(queries, vectors, strict=True)}


def remaining_daily_calls():
    """Read only aggregate quota metadata, never personal records or config secrets."""
    from app.services.llm.budget import BudgetStore
    return BudgetStore().daily_status()['remaining_calls']


def representative_entries(entries, count, offset=0):
    if not 1 <= count <= 24 or offset < 0:
        raise ValueError("verify_count_must_be_1_to_24")
    groups = {}
    for entry in entries:
        groups.setdefault(entry["topic"], []).append(entry)
    selected = []
    for index in range(max((len(group) for group in groups.values()), default=0)):
        selected.extend(group[index] for group in groups.values() if index < len(group))
    return selected[offset:offset + count]


def expected_sources(entry, question_index):
    return set(entry.get('expected_sources', {}).get(str(question_index), [entry['id']]))


def verify_coverage(catalog, documents, args):
    available = {document.metadata["coverage_id"] for document in documents}
    entries = [entry for entry in catalog if entry["id"] in available]
    pipeline = HybridRAGPipeline()
    lexical = []
    for entry in entries:
        for question_index, query in enumerate(entry["questions"]):
            hits = pipeline.lexical(query, 5)
            lexical.append({"id": entry["id"], "topic": entry["topic"], "question": query,
                            "expected_in_top5": any(hit.metadata.get("coverage_id") in expected_sources(entry, question_index) for hit in hits)})
    requested_ids = getattr(args, 'verify_ids', None)
    if requested_ids and not set(requested_ids) <= {entry['id'] for entry in entries}:
        raise ValueError('unknown_or_unavailable_verify_ids')
    selected = [entry for entry in entries if not requested_ids or entry['id'] in requested_ids]
    chosen = representative_entries(selected, args.verify_count, getattr(args, 'verify_offset', 0))
    question_index = getattr(args, 'verify_question_index', 0)
    if not chosen or any(len(entry['questions']) <= question_index for entry in chosen):
        raise ValueError('verification_selection_empty_or_missing_question')
    # The same provider/query vectors can prefill the existing request cache.
    # No retrieval scores, candidates or quality thresholds are substituted.
    batched = getattr(args, 'batch_query_embeddings', False)
    estimated = len(chosen) + ((len(chosen) + 19) // 20 if batched else len(chosen))
    reserve = getattr(args, 'reserve_model_calls', 20)
    allowed = remaining_daily_calls() >= estimated + reserve
    identity = {"sources": [(document.source, document.metadata["content_sha256"]) for document in documents],
                "queries": [entry["questions"][question_index] for entry in chosen]}
    cache_token = None
    stop_reason = None
    async def evaluate():
        nonlocal cache_token, stop_reason
        results = []
        if not allowed:
            return results
        with model_job("official-coverage-retrieval", args, identity):
            if batched:
                from app.services.rag.embedding_client import EmbeddingUnavailable
                try:
                    cache_token = embedding_cache.set(await prefill_query_embeddings(identity['queries']))
                except EmbeddingUnavailable as error:
                    stop_reason = 'batch_embedding:' + str(error)
                    return results
            for entry in chosen:
                task_type = entry.get('question_task_types', {}).get(str(question_index)) or entry.get('task_type') or ("safety_triage" if entry["topic"] in {"injury", "safety"} else (
                    "nutrition_advice" if entry["topic"] in {"nutrition", "weight", "supplements"} else "training_qa"))
                intent = {"safety_triage": "symptom_triage", "nutrition_advice": "nutrition"}.get(task_type, "training_qa")
                query = entry["questions"][question_index]
                traces = []
                token = retrieval_traces.set(traces)
                try:
                    hits = await pipeline.asearch(query, top_k=5, skip_rewrite=True,
                        intent_result=IntentResult(intent, 1.0, "normal", "question"))
                finally:
                    retrieval_traces.reset(token)
                degraded = ';'.join(trace['degraded_reason'] for trace in traces if trace.get('degraded_reason'))
                assessment = assess_rag_evidence(query, hits, task_type=task_type)
                row = {"id": entry["id"], "topic": entry["topic"], "question": query, "task_type": task_type,
                       "assessment": assessment.as_dict(), "hit_count": len(hits),
                       "expected_source_ids": sorted(expected_sources(entry, question_index)),
                       "expected_in_top5": any(hit.metadata.get("coverage_id") in expected_sources(entry, question_index) for hit in hits),
                       "degraded_reason": degraded or None,
                       "sources": [{"id": hit.metadata.get("coverage_id"), "title": hit.title,
                                    "url": hit.metadata.get("url"), "score": hit.metadata.get("rerank_score")} for hit in hits]}
                results.append(row)
                print(json.dumps({"verification": entry["id"], "answerable": assessment.answerable,
                                  "confidence": assessment.confidence, "expected_in_top5": row["expected_in_top5"]}), flush=True)
                # Provider/budget failures are infrastructure failures, not
                # evidence of a corpus gap. Preserve the case and stop the batch.
                if degraded:
                    break
        return results
    async def run():
        try:
            return await evaluate()
        finally:
            if cache_token is not None:
                embedding_cache.reset(cache_token)
            from app.services.rag.async_http import close
            await close()
    live = asyncio.run(run())
    return {"lexical": lexical, "lexical_passed": sum(row["expected_in_top5"] for row in lexical),
            "lexical_total": len(lexical), "live": live, "live_answerable": sum(row["assessment"]["answerable"] for row in live),
            "live_passed": sum(row["assessment"]["answerable"] and row["expected_in_top5"] and not row['degraded_reason'] for row in live),
            "live_total": len(live), "live_planned": len(chosen),
            "batch_query_embeddings": batched, "estimated_model_calls": estimated,
            "stopped_reason": ('insufficient_daily_budget_with_chat_reserve' if not allowed else
                stop_reason or next((row['degraded_reason'] for row in live if row['degraded_reason']), None)),
            "scope": "retrieval and existing evidence gate, not clinical diagnosis or full chat generation"}
