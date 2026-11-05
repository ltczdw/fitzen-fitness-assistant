import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.db.memory_models import MemoryTurn
from app.services.agent import dialogue_context,semantics
from app.services.context.budget import ContextBudgetManager,ContextBudgetExceeded
from app.services.memory import compact,wiki_context as wc
from app.services.memory.recent_window import load_window
from tests.test_wiki_addressed_context import memory_db,add_turn,load,pid


@pytest.fixture(autouse=True)
def hybrid(monkeypatch):
    monkeypatch.setattr(settings,'memory_short_window_turns',10)
    monkeypatch.setattr(settings,'memory_recent_max_tokens',12000)


def enable(monkeypatch):
    # The imported addressed-only fixture intentionally sets zero after the
    # autouse setup. These tests explicitly exercise the product hybrid mode.
    monkeypatch.setattr(settings,'memory_short_window_turns',10)


def test_persistent_defaults_and_explicit_addressed_only_override(monkeypatch):
    from app.core.config import Settings
    monkeypatch.delenv('MEMORY_SHORT_WINDOW_TURNS',raising=False)
    monkeypatch.delenv('MEMORY_RECENT_MAX_TOKENS',raising=False)
    current=Settings(_env_file=None)
    assert current.memory_short_window_turns==10 and current.memory_recent_max_tokens==12000
    assert Settings(_env_file=None,memory_short_window_turns=0).memory_short_window_turns==0


@pytest.mark.parametrize('unrelated',[0,100])
def test_sql_ten_visible_pairs_with_one_window_query(memory_db,monkeypatch,unrelated):
    enable(monkeypatch)
    ids=[add_turn('目标'+str(i),'完整回答'+str(i)) for i in range(15)]
    for i in range(unrelated):add_turn('无关会话'+str(i),'不能进入本会话',cid='other')
    memory_db.queries.clear()
    memory=load_window('wiki-chat',0)
    assert len(memory_db.queries)==1
    assert memory['recent_window']['loaded_turn_ids']==ids[-10:]
    assert len(memory['short_memory'])==20
    status=compact.CompactMemory(SimpleNamespace()).status()
    assert status['window_turns']==10 and status['recent_max_tokens']==12000
    assert '不能进入本会话' not in json.dumps(memory,ensure_ascii=False)
    assert 'LIMIT' in memory_db.queries[0].upper()
    with memory_db.factory.begin() as s:
        s.get(MemoryTurn,ids[-1]).hidden_at=compact.now()
        s.get(MemoryTurn,ids[-2]).deleted_at=compact.now()
        s.get(MemoryTurn,ids[-3]).memory_epoch=1
    visible=load_window('wiki-chat',0)
    assert not set(ids[-3:]).intersection(visible['recent_window']['loaded_turn_ids'])
    assert len(visible['recent_window']['loaded_turn_ids'])==10


def test_full_recent_roles_conditions_and_no_extra_history_read(memory_db,monkeypatch):
    enable(monkeypatch)
    ids=[add_turn('条件不能省略'+str(i),'回答否定条件'+str(i)) for i in range(12)]
    memory=load()
    messages=dialogue_context.semantic_messages(semantics.PROMPT,'继续',memory)
    text=json.dumps(messages,ensure_ascii=False)
    assert all('条件不能省略'+str(i) in text and '回答否定条件'+str(i) in text for i in range(2,12))
    assert '回答否定条件0"' not in text
    assert not semantics._history_required({'tasks':[{'task_type':'training_qa','action':'question',
        'history_requirement':'required','context_ids':[ids[-2]+':user']}]},memory)
    assert semantics._history_required({'tasks':[{'task_type':'training_qa','action':'question',
        'history_requirement':'required','context_ids':[ids[0]+':user']}]},memory)


def test_oversized_turn_is_wholly_omitted_and_remains_addressable(memory_db,monkeypatch):
    enable(monkeypatch)
    monkeypatch.setattr(settings,'memory_recent_max_tokens',1000)
    small=add_turn('器械是哑铃','完整短回答，不能省略条件。')
    huge=add_turn('另一个目标','长回答条件否定不可截断'*1000)
    memory=load()
    assert huge in memory['recent_window']['omitted_turn_ids']
    assert [m['turn_id'] for m in memory['short_memory']]==[small,small]
    assert pid(memory,huge)
    rendered=json.dumps(dialogue_context.semantic_messages(semantics.PROMPT,'继续',memory),ensure_ascii=False)
    assert '长回答条件否定不可截断' not in rendered
    assert 'omitted_turn_ids' in rendered


def test_actual_gateway_envelope_prunes_only_optional_whole_turns(memory_db,monkeypatch):
    enable(monkeypatch)
    old=add_turn('旧条件','旧回答'*400)
    latest=add_turn('新条件','新回答'*400)
    memory=load()
    base=dialogue_context.semantic_messages(semantics.PROMPT,'继续',{})
    from app.services.llm import client
    counter=client.context_budget.counter
    limit=counter.messages(base)+1400
    manager=ContextBudgetManager(window=limit+1,output=1,margin=0,counter=counter)
    monkeypatch.setattr(client,'context_budget',manager)
    packed=dialogue_context.semantic_messages(semantics.PROMPT,'继续',memory)
    assert counter.messages(packed)<=manager.input_budget
    for tid in (old,latest):
        selected=[m for m in memory['short_memory'] if m['turn_id']==tid]
        assert len(selected) in (0,2)
    # Once selected as required, that complete pair cannot be silently pruned.
    memory=load()
    memory['recent_required_ids']=[latest+':user',latest+':assistant']
    tiny=ContextBudgetManager(window=counter.messages(base)+500,output=1,margin=0,counter=counter)
    monkeypatch.setattr(client,'context_budget',tiny)
    required=dialogue_context.semantic_messages(semantics.PROMPT,'继续',memory)
    assert any(m.get('content','').endswith('新回答'*400) for m in required)
    with pytest.raises(ContextBudgetExceeded):tiny.validate(required)


def test_recent_wiki_read_reuses_original_and_older_read_merges_without_duplicates(memory_db,monkeypatch):
    enable(monkeypatch)
    ids=[add_turn('目标'+str(i),'保留完整条件'+str(i)) for i in range(12)]
    memory=load()
    memory.update(wiki_catalog_entries={e['page_id']:e for e in memory['wiki_catalog']['entries']},
                  wiki_loaded_pages={},wiki_read_tokens=0)
    recent=pid(memory,ids[-1])
    memory_db.queries.clear()
    pages,cost=wc.read_pages(memory,[recent],0)
    assert cost==0 and len(pages)==1
    assert not any('SELECT wiki_sections.page_id, wiki_sections.source_message_id' in q for q in memory_db.queries)
    wc._merge(memory,pages,cost)
    older=pid(memory,ids[0])
    pages,cost=wc.read_pages(memory,[older],14000)
    assert cost>0
    wc._merge(memory,pages,cost)
    memory['wiki_prepared']=True
    mids=[m['message_id'] for m in memory['short_memory']]
    assert len(mids)==len(set(mids))==22
    assert memory['wiki_read_tokens']==cost
    selected=dialogue_context.scoped_worker_memory(memory,{'context_ids':[ids[-1]+':user']},'training_qa')
    assert [m['message_id'] for m in selected['short_memory']]==[ids[-1]+':user',ids[-1]+':assistant']
    assert not any(m.get('content')=='保留完整条件0' for m in selected['recent_window_messages'])


def test_recent_source_is_revalidated_even_after_wiki_prepare(memory_db,monkeypatch):
    enable(monkeypatch)
    old=add_turn('旧目标','旧回答')
    recent=add_turn('当前条件不可丢','当前回答')
    memory=load()
    memory.update(wiki_catalog_entries={e['page_id']:e for e in memory['wiki_catalog']['entries']},wiki_loaded_pages={},wiki_read_tokens=0)
    pages,cost=wc.read_pages(memory,[pid(memory,old)],14000)
    wc._merge(memory,pages,cost)
    memory['wiki_prepared']=True
    # Keep only the old Wiki page selected; the recent raw source still needs
    # its independent current-row visibility check.
    request=__import__('app.schemas.chat',fromlist=['ChatRequest']).ChatRequest(message='继续',conversation_id='wiki-chat')
    tid,_=compact.begin_turn(request)
    token=compact.current_turn.set(tid)
    try:
        with memory_db.factory.begin() as s:s.get(MemoryTurn,recent).hidden_at=compact.now()
        with pytest.raises(dialogue_context.DialogueContextUnavailable):
            dialogue_context._restore(request,memory,{recent+':user'})
    finally:compact.current_turn.reset(token)


@pytest.mark.parametrize('transport',['json','sse'])
def test_real_hybrid_load_routes_ten_turns_but_generates_only_selected_goal(tmp_path,monkeypatch,transport):
    from app.core import clock
    from app.schemas.chat import ChatRequest
    from evals.v2.safety import deny_network
    from tests.react_supervisor_fixtures import tool_call
    from tests.react_readonly_fixtures import environment,chunks,semantic_outline,semantic_task,sse_body
    with deny_network() as audit,environment(tmp_path,monkeypatch,'recent-http') as env:
        enable(monkeypatch)
        monkeypatch.setattr(settings,'wiki_enabled',True)
        base=clock.utc_now()
        for i in range(12):
            env.mp.setattr(clock,'utc_now',lambda i=i:base+timedelta(seconds=i))
            tid,_=compact.begin_turn(ChatRequest(message='旧目标'+str(i),conversation_id='demo-conversation'))
            compact.finish_turn(tid,'OLD_ANSWER_'+str(i),{'answer':'OLD_ANSWER_'+str(i)})
        env.mp.setattr(clock,'utc_now',lambda:base+timedelta(seconds=20))
        service=compact.CompactMemory(SimpleNamespace(initialize=AsyncMock(side_effect=AssertionError('no Redis read'))))
        env.load.side_effect=service.load
        source='给我制定饮食方案'
        reply='不要肉，改成素食'
        env.mock_rag(chunks(source))
        llm=env.mock_llm(final='原饮食示例',policy=lambda index,messages:{'role':'assistant','content':'','tool_calls':[
            tool_call('diet-'+str(index),'advise_nutrition',{'query':source})]})
        routed=[]
        async def structured(messages,schema,**kwargs):
            assert kwargs.get('name')=='return_request_understanding'
            payload=json.loads(messages[-1]['content'])
            routed.append(messages)
            t=semantic_task('nutrition_advice','question',payload['query'])
            t.update(delivery_type='plan',history_requirement='none')
            if payload['query']==reply:
                target=payload['result_sources'][0]
                t.update(result_reference={'message_id':target['message_id'],'selection':'latest','quote':reply},
                         context_ids=[target['user_message_id'],target['message_id']],history_requirement='required')
            value=schema.model_validate(semantic_outline(payload['query'],[t])|{'relation':'follow_up' if payload['query']==reply else 'new'})
            if kwargs.get('validator'):kwargs['validator'](value)
            return value
        monkeypatch.setattr(llm,'complete_structured',structured)
        first=env.post('first-recent',message=source)
        assert first['metrics']['react_status']=='complete',first
        assert first['metrics']['recent_memory']['loaded_turns']==10
        assert first['metrics']['recent_memory']['token_limit']==12000
        payload=json.loads(routed[0][-1]['content'])
        assert len(payload['history'])==20 and payload['recent_window']['candidate_turns']==10
        assert 'OLD_ANSWER_11' in json.dumps(routed[0]) and 'OLD_ANSWER_0"' not in json.dumps(routed[0])
        assert 'OLD_ANSWER_' not in json.dumps([c for c in llm.calls if not c['tools']][-1]['messages'])
        env.mp.setattr(clock,'utc_now',lambda:base+timedelta(seconds=21))
        llm.final='素食饮食示例'
        result=env.post('second-recent',message=reply) if transport=='json' else sse_body(env.stream('second-recent',message=reply))
        assert result['metrics']['react_status']=='complete' and result['answer']=='素食饮食示例',result
        prompt=json.dumps([c for c in llm.calls if not c['tools']][-1]['messages'],ensure_ascii=False)
        assert source in prompt and reply in prompt and 'OLD_ANSWER_' not in prompt
        assert not any((c['tools'] or [{}])[0].get('function',{}).get('name')=='return_wiki_read_plan' for c in llm.calls)
        count=len(llm.calls)
        assert env.post('second-recent',message=reply)['answer']==result['answer']
        assert len(llm.calls)==count
    assert audit=={'blocked_connections':0,'blocked_dns':0}
