"""Read-only reference queries supplied by shared model understanding."""
from app.services.agent.semantics import (
    parent_reference_for, understanding_covers, reference_for, understand,
)


def resolve_reference(query, history):
    """Only use a validated request-local model result; ambiguity keeps raw text."""
    item = reference_for(query)
    if item is None or item.ambiguous:
        return query
    return item.query


async def resolve_reference_model(query, history, llm, *, parent_source=None):
    item = reference_for(query, source=parent_source if parent_source is not None else query)
    if item is not None:
        return query if item.ambiguous else item.query
    parent_item = parent_reference_for(query, parent_source)
    if parent_item is not None:
        # The model supplied an explicit rewrite for the trusted parent request;
        # an exact read-only child may consume it without a second understanding.
        return parent_item.query
    if parent_source is not None and understanding_covers(query, source=parent_source):
        # The current request already owns a validated decision for this exact
        # task text. It has no applicable rewrite, so keep the original query
        # instead of paying for a second RequestUnderstanding.
        return query
    await understand(llm, query, context={"short_memory": history})
    item = reference_for(query, source=query)
    return item.query if item is not None and not item.ambiguous else query
