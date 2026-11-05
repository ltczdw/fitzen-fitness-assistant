"""Budgeted complete exchanges for routing; older sources stay addressable."""
from __future__ import annotations

from sqlalchemy import select

from app.core.config import settings
from app.db.memory_models import MemoryTurn
from app.services.context.token_counter import TokenCounter


def load_window(cid: str, epoch: int) -> dict:
    from app.services.memory import compact
    limit=int(settings.memory_short_window_turns)
    if not limit:
        return {}
    with compact.session() as session:
        rows=list(session.execute(select(MemoryTurn.turn_id,MemoryTurn.messages,
            MemoryTurn.content_hash,MemoryTurn.completed_at,
            MemoryTurn.response['intent'].as_string().label('response_intent'),
            MemoryTurn.response['metrics']['execution_mode'].as_string().label('execution_mode'),
            MemoryTurn.response['metrics']['react_status'].as_string().label('react_status')).where(
            MemoryTurn.conversation_id==cid,MemoryTurn.memory_epoch==epoch,
            MemoryTurn.status=='completed',MemoryTurn.hidden_at.is_(None),
            MemoryTurn.deleted_at.is_(None)).order_by(
            MemoryTurn.completed_at.desc(),MemoryTurn.turn_id.desc()).limit(limit)))
    counter=TokenCounter()
    accepted=[]
    omitted=[]
    messages=[]
    for row in rows:
        turn=[{**m,'turn_id':row.turn_id,'app_meta':{**m.get('app_meta',{}),
            'recent_window':True,'recent_completed_at':row.completed_at.isoformat() if row.completed_at else '',
            'recent_content_hash':row.content_hash}}
            for m in row.messages if m.get('role') in {'user','assistant'}]
        # Whole exchanges are the deterministic complete-unit fallback. No
        # model call or sentence/negation splitter is introduced on load.
        if counter.messages([*turn,*messages])>settings.memory_recent_max_tokens:
            omitted.append(row.turn_id)
        else:
            messages=[*turn,*messages]
            accepted.append(row.turn_id)
    # Failure replies remain visible as conversation history, but cannot replace
    # the latest usable answer identity. Reuse a visible candidate in the loaded
    # window; if all ten failed, query the prior result with filtering before LIMIT.
    candidate=next((row for row in rows if row.response_intent != 'execution_limited'
                    and row.execution_mode != 'limited' and row.react_status in {None, 'complete'}),None)
    latest=compact.latest_result_dialogue(cid,epoch,
        latest_messages=candidate.messages if candidate is not None else (None if rows else []),
        cached_messages=[{**m, 'turn_id': row.turn_id,
                          'app_meta': {**m.get('app_meta', {}), 'recent_content_hash': row.content_hash}}
                         for row in reversed(rows) for m in row.messages
                         if m.get('role') in {'user', 'assistant'}])
    return {'short_memory':messages,'recent_window_messages':messages,
            'recent_result_dialogue':latest,
            'recent_window':{'turn_limit':limit,'token_limit':settings.memory_recent_max_tokens,
                'candidate_turns':len(rows),'loaded_turn_ids':list(reversed(accepted)),
                'omitted_turn_ids':omitted,'tokens':counter.messages(messages) if messages else 0}}


def merge_messages(memory: dict, selected: list[dict]) -> list[dict]:
    """Selected Wiki originals replace duplicates, retaining their unit tags."""
    ids={m.get('message_id') for m in selected}
    recent=[m for m in memory.get('recent_window_messages') or [] if m.get('message_id') not in ids]
    return sorted([*recent,*selected],key=lambda m:(
        str(m.get('app_meta',{}).get('recent_completed_at') or m.get('app_meta',{}).get('wiki_completed_at') or ''),
        str(m.get('turn_id') or ''),0 if m.get('role')=='user' else 1))


def fit_routing_history(memory, history, render, counter, input_budget):
    """Fit optional whole recent turns against the actual gateway envelope.

    Selected evidence is never evicted. Removing a default-window turn leaves
    its source address/omission report, so a required later read must restore it.
    """
    optional={str(m.get('message_id')):m for m in memory.get('short_memory') or []
              if m.get('app_meta',{}).get('recent_window') and not m.get('app_meta',{}).get('wiki_page_id')}
    if not optional:
        return history
    required=set(memory.get('recent_required_ids') or [])
    groups={}
    for item in history:
        mid=item['message_id']
        if mid in optional:
            groups.setdefault(optional[mid]['turn_id'],[]).append(mid)
    fixed={mid for mids in groups.values() if required.intersection(mids) for mid in mids}
    keep={item['message_id'] for item in history if item['message_id'] not in optional or item['message_id'] in fixed}
    for mids in reversed(list(groups.values())):
        trial=keep|set(mids)
        chosen=[item for item in history if item['message_id'] in trial]
        if counter.messages(render(chosen))<=input_budget:
            keep=trial
    dropped={mid for mid in optional if mid not in keep}
    if dropped:
        memory['short_memory']=[m for m in memory['short_memory'] if str(m.get('message_id')) not in dropped]
        memory['recent_window_messages']=[m for m in memory.get('recent_window_messages') or []
                                          if str(m.get('message_id')) not in dropped]
        report=memory['recent_window']
        dropped_turns={optional[mid]['turn_id'] for mid in dropped}
        report['omitted_turn_ids']=list(dict.fromkeys([*report['omitted_turn_ids'],*dropped_turns]))
        report['loaded_turn_ids']=[t for t in report['loaded_turn_ids'] if t not in dropped_turns]
        report['tokens']=counter.messages(memory['recent_window_messages']) if memory['recent_window_messages'] else 0
    return [item for item in history if item['message_id'] in keep]
