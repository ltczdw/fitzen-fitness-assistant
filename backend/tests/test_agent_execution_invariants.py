import asyncio
import gc
import weakref
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import settings
from app.services.agent.execution_state import RequestIdConflict, _conversation_locks, conversation_lock
from app.services.agent.planner_validation import validate_rule_plan
from app.services.agent.task_planner import plan_tasks_rule, task_write_authorization
from app.services.agent.task_scheduler import select_ready_task
from app.services.agent.routing import PlannerRoutingMixin
from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
from app.services.security.write_guard import guard_planner_tasks
from app.services.rag.chunking import split_document


@pytest.mark.parametrize(('message', 'flags', 'expected'), [
    ('别把卧推记录一下', {'is_negated': True}, 'rejected_unauthorized'),
    ('补录卧推60kg，4组8次', {}, 'authorized'),
    ('记录卧推60kg，行不行？', {'is_hypothetical': True}, 'rejected_unauthorized'),
    ('记录其他动作，卧推60kg，4组8次', {}, 'authorized'),
])
def test_one_write_authority_drives_guard_and_planner(message, flags, expected):
    from app.services.agent.semantics import remember_safety
    task = {'id': 'write', 'task_type': 'workout_record', 'action': 'record', 'text': message, **flags}
    remember_safety(message, {'level': 'normal', 'evidence': '', 'categories': [], 'reason': 'scripted'})
    guard_planner_tasks(message, [task], now=__import__('datetime').datetime.now(__import__('datetime').timezone.utc),
                        semantic_available=True)
    assert task['payload']['write_authorization'] == expected
    assert task_write_authorization(message, {**flags, 'semantic_available': True}) == expected
    assert task['execution_status'] == 'pending'


def test_rule_fallback_keeps_authorized_record_runnable(monkeypatch):
    from app.services.agent import routing
    from app.services.agent.semantics import remember_safety, remember_understanding, request_understanding
    message = '记录饮食鸡蛋，再安排晚餐'
    monkeypatch.setattr(routing, 'get_pending_food', lambda *_: None)
    monkeypatch.setattr(settings, 'planner_shadow_enabled', False)
    token = request_understanding.set({})
    try:
        remember_safety(message, {'level': 'normal', 'evidence': '', 'categories': [], 'reason': 'scripted'})
        remember_understanding(message, {"validated_tasks": [
            {'task_type': 'nutrition_record', 'action': 'record', 'text': '记录饮食鸡蛋',
             'is_negated': False, 'is_hypothetical': False, 'is_third_party': False},
            {'task_type': 'nutrition_advice', 'action': 'question', 'text': '再安排晚餐',
             'is_negated': False, 'is_hypothetical': False, 'is_third_party': False},
        ]})
        graph = SimpleNamespace(parse_input=AsyncMock(return_value={'tasks': [
            {'task_type': 'nutrition_record', 'action': 'record', 'text': '记录饮食鸡蛋',
             'is_negated': False, 'is_hypothetical': False, 'is_third_party': False},
            {'task_type': 'nutrition_advice', 'action': 'question', 'text': '再安排晚餐',
             'is_negated': False, 'is_hypothetical': False, 'is_third_party': False},
        ]}))
        class Agent(PlannerRoutingMixin):
            def __init__(self): self.graph = graph
        request = SimpleNamespace(message=message, user_id='demo-user', conversation_id='fixture')
        decision = asyncio.run(Agent()._route(request, {}))
        record = next(t for t in decision.task_specs if t['task_type'] == 'nutrition_record')
        advice = next(t for t in decision.task_specs if t['task_type'] == 'nutrition_advice')
        assert record['payload']['write_authorization'] == 'authorized'
        assert select_ready_task(decision.task_specs, [])['id'] == record['id']
        assert record['id'] in advice['depends_on']
    finally:
        request_understanding.reset(token)


def test_no_model_rule_fallback_never_authorizes_a_write(monkeypatch):
    from app.services.agent import routing
    from app.services.agent.semantics import request_understanding
    message = '记录饮食鸡蛋，再安排晚餐'
    monkeypatch.setattr(routing, 'get_pending_food', lambda *_: None)
    monkeypatch.setattr(settings, 'planner_shadow_enabled', False)
    token = request_understanding.set({})
    try:
        graph = SimpleNamespace(parse_input=AsyncMock(return_value=None))
        class Agent(PlannerRoutingMixin):
            def __init__(self): self.graph = graph
        request = SimpleNamespace(message=message, user_id='demo-user', conversation_id='fixture')
        decision = asyncio.run(Agent()._route(request, {}))
        writes = [t for t in decision.task_specs if t.get('task_type') in {'nutrition_record', 'workout_record'}]
        assert not writes or all(t.get('payload', {}).get('write_authorization') != 'authorized' for t in writes)
    finally:
        request_understanding.reset(token)


def test_graph_resolves_successful_dependency_and_reports_failed_one():
    tasks = [{'id':'advice','task_type':'nutrition_advice','worker':'nutrition','action':'question','text':'安排晚餐','depends_on':['record'],'status':'blocked','execution_status':'blocked','blocked_reason':'record_dependency_not_ready'}]
    complete = FitnessMultiAgentGraph._post_worker_node({'tasks':tasks,'worker_results':[{'task_id':'record','status':'complete','answer':'已记录'}]})
    assert complete['tasks'][0]['execution_status']=='pending'
    assert select_ready_task(complete['tasks'], ['record']) is complete['tasks'][0]
    failed = FitnessMultiAgentGraph._post_worker_node({'tasks':tasks,'worker_results':[{'task_id':'record','status':'needs_confirmation','answer':'请补充重量'}]})
    assert failed['tasks'][0]['execution_status']=='blocked'
    final = FitnessMultiAgentGraph._finalize_node({'tasks':failed['tasks'],'worker_results':[{'status':'needs_confirmation','answer':'请补充重量'}]})
    assert '饮食建议' in final['final_answer'] and '未执行' in final['final_answer']


@pytest.mark.parametrize(('worker','task_type','action','text'),[
    ('workout_record','workout_record','record','别把卧推记录一下'),
    ('nutrition','nutrition_record','record','别把鸡蛋记录一下'),
])
def test_pending_rejected_write_is_stopped_at_worker_before_record_access(worker,task_type,action,text,monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.agent import multi_agent_graph as graph_module
    graph=FitnessMultiAgentGraph()
    extraction=AsyncMock();save=AsyncMock()
    monkeypatch.setattr(graph.workout_extraction,'run_many',extraction)
    monkeypatch.setattr(graph_module,'extract_foods_with_llm',extraction)
    monkeypatch.setattr(graph,'_run_authorized_model_tool',save)
    task={'id':'write','task_type':task_type,'worker':worker,'action':action,'text':text,
          'status':'rejected_unauthorized','execution_status':'pending',
          'payload':{'write_authorization':'rejected_unauthorized'}}
    assert select_ready_task([task],[]) is task
    result=asyncio.run(graph.workers[worker].ainvoke({'user_id':'demo-user','conversation_id':'fixture',
        'raw_message':text,'task_text':text,'task_type':task_type,'task_action':action,
        'task_payload':task['payload'],'request_id':'fixture'}))['result']
    assert result['status']=='rejected_unauthorized'
    extraction.assert_not_called();save.assert_not_called()


def test_cancelled_model_tool_round_propagates_and_stops_after_one_write():
    from app.services.llm.client import LLMClient
    async def run():
        client=LLMClient();client.configured=lambda:True
        client._complete_message=AsyncMock(side_effect=[{'content':'','tool_calls':[{'id':'call-1','function':{'name':'save','arguments':'{}'}}]},asyncio.CancelledError()])
        execute=AsyncMock(return_value={'saved':True})
        with pytest.raises(asyncio.CancelledError):
            await client._complete_with_tools_impl([{'role':'user','content':'fixture'}],tools=[{'type':'function','function':{'name':'save'}}],tool_executor=execute)
        assert execute.await_count==1
        assert client._complete_message.await_count==2
    asyncio.run(run())


def test_cancellation_does_not_create_degraded_failure_diagnostic(monkeypatch):
    from app.services.agent import runtime
    saved=AsyncMock(return_value='ready')
    monkeypatch.setattr(runtime,'save_diagnostics',saved)
    class Agent:
        @asynccontextmanager
        async def _request_scope(self,request): yield None
        async def _run_planner_shadow(self,message): return None
        async def _respond_impl(self,*args): raise asyncio.CancelledError()
    from app.schemas.chat import ChatRequest
    async def run():
        with pytest.raises(asyncio.CancelledError): await runtime.execute_request(Agent(),ChatRequest(message='fixture'))
    asyncio.run(run())
    assert saved.call_args.args[3]=='cancelled'
    metrics=saved.call_args.args[2]
    assert metrics['degraded'] is False and metrics['failure_reasons']==[]


def test_request_id_conflict_maps_to_http_409(monkeypatch):
    from fastapi import FastAPI
    from app.api import chat as chat_module
    from app.main import request_id_conflict_handler
    async def conflict(_): raise RequestIdConflict('request_id_payload_conflict')
    monkeypatch.setattr(chat_module,'require_ready',lambda:None)
    monkeypatch.setattr(chat_module.agent,'respond',conflict)
    app=FastAPI();app.post('/chat')(chat_module.chat)
    app.add_exception_handler(RequestIdConflict,request_id_conflict_handler)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            response=await client.post('/chat',json={'message':'first payload','request_id':'same-id'})
            assert response.status_code==409
    asyncio.run(run())


def test_request_scope_converts_compact_memory_replay_conflict(monkeypatch):
    from app.services.agent.agent import FitnessAgent
    from app.services.agent import agent as agent_module
    monkeypatch.setattr(settings,'memory_read_backend','compact_v1')
    def conflict(_): raise ValueError('request_id_payload_conflict')
    monkeypatch.setattr(agent_module,'begin_turn',conflict)
    async def run():
        with pytest.raises(RequestIdConflict):
            async with FitnessAgent()._request_scope(SimpleNamespace(message='other',request_id='same',conversation_id='fixture')):
                pass
    asyncio.run(run())


def test_conversation_locks_are_weakly_retained_per_event_loop():
    loop=asyncio.new_event_loop()
    try:
        lock=conversation_lock(loop,'ephemeral')
        ref=weakref.ref(lock)
        del lock;gc.collect()
        assert ref() is None
        assert _conversation_locks[loop].get('ephemeral') is None
    finally:loop.close()


def test_all_conversation_scrub_must_be_explicit():
    from app.db.quality_repository import scrub_in_session
    with pytest.raises(ValueError,match='explicitly'):
        scrub_in_session(None,None)


def test_markdown_fences_only_close_with_the_same_marker_and_length():
    chunks=split_document('~~~text\n```\n~~~~\n# Real heading\nBody.')
    assert len(chunks)==2
    assert chunks[1].section_path=='Real heading'
    assert '```\n~~~~' in chunks[0].content
