import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from sqlalchemy import select

from app.core.config import settings
from app.db.memory_models import MemoryTurn, MemoryOutbox, UserFact
from app.schemas.chat import ChatRequest
from app.services.llm.client import LLMClient, LLMGatewayError, LLMErrorCategory
from app.services.llm.structured import complete_structured, StructuredOutputError
from app.services.memory.deepseek_summary import DeepSeekSummaryClient, SummaryResult
from app.services.memory import compact
from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def no_provider_network():
    with deny_network():yield


def result(items=True):
    return {'items':[{'text':'用户偏好晨练','category':'preference','speaker':'user',
                      'importance':.8,'confidence':.9,'source_ids':['u1']}] if items else [],
            'importance_scores':[{'message_id':'u1','score':.8},{'message_id':'a1','score':.1}]}


def wire(arguments=None, name='return_memory_summary'):
    return {'content':'This prose is intentionally ignored.', 'tool_calls':[{
        'id':'result-one','type':'function','function':{'name':name,
        'arguments':json.dumps(result() if arguments is None else arguments,ensure_ascii=False)}}]}


class FakeWireClient:
    def __init__(self,responses):self.responses=list(responses);self.calls=[]
    async def complete_message(self,messages,**kwargs):
        self.calls.append((copy.deepcopy(messages),kwargs))
        reply=self.responses.pop(0)
        if isinstance(reply,Exception):raise reply
        return reply
    async def complete_structured(self,messages,schema,**kwargs):
        return await complete_structured(self,messages,schema,**kwargs)


def summary(fake):
    client=DeepSeekSummaryClient(fake)
    client.probe=AsyncMock(return_value=True)
    return client


MESSAGES=[{'message_id':'u1','role':'user','content':'我偏好晨练'},
          {'message_id':'a1','role':'assistant','content':'收到'}]


def test_schema_and_forced_function_sent_actual_args_parsed_without_prose():
    fake=FakeWireClient([wire()])
    output=asyncio.run(summary(fake).summarize(MESSAGES))
    assert output['items'][0]['text']=='用户偏好晨练'
    assert len(fake.calls)==1
    messages,options=fake.calls[0]
    function=options['tools'][0]['function']
    assert function['parameters']==SummaryResult.model_json_schema()
    assert function['name']=='return_memory_summary'
    assert options['tool_choice']['function']['name']=='return_memory_summary'
    assert 'speaker' in messages[0]['content']
    assert 'importance_scores必须是' in messages[0]['content']


def test_one_semantic_correction_keeps_original_user_input():
    fake=FakeWireClient([{'content':'no tool_calls'},wire()])
    output=asyncio.run(summary(fake).summarize(MESSAGES))
    assert output['items'] and len(fake.calls)==2
    # The original user packet is byte-for-byte unchanged; only the system
    # instruction gains a bounded correction and the result keeps real source
    # ids. Do not pin a synonymous prompt phrase.
    assert fake.calls[0][0][1]==fake.calls[1][0][1]
    assert fake.calls[1][0][0]['content'].startswith(fake.calls[0][0][0]['content'])
    assert fake.calls[1][0][0]['content'] != fake.calls[0][0][0]['content']
    assert output['items'][0]['source_ids']==['u1']
    assert output['processed_ids']==['u1','a1']


@pytest.mark.parametrize('bad',[
    {'content':'just prose'},
    wire(result(),name='save_workout_record'),
    {'tool_calls':[wire()['tool_calls'][0],wire()['tool_calls'][0]]},
    wire({'items':[],'importance_scores':{'u1':.1,'a1':.2}}),
    wire({'items':[{'text':'private-value-do-not-log'}],'importance_scores':[]}),
])
def test_repeated_bad_outputs_stop_and_error_does_not_expose_values(bad):
    fake=FakeWireClient([bad,bad])
    with pytest.raises(StructuredOutputError) as caught:
        asyncio.run(summary(fake).summarize(MESSAGES))
    assert len(fake.calls)==2
    assert caught.value.category==LLMErrorCategory.PROTOCOL
    assert not caught.value.retryable
    assert 'private-value' not in str(caught.value)
    assert 'private-value' not in json.dumps(caught.value.validation_errors)


@pytest.mark.parametrize('scores',[
    [{'message_id':'u1','score':.5}],
    [{'message_id':'u1','score':.5},{'message_id':'a1','score':.1},{'message_id':'u1','score':.2}],
    [{'message_id':'u1','score':.5},{'message_id':'a1','score':.1},{'message_id':'invented','score':.2}],
])
def test_score_coverage_and_duplicates_rejected_without_more_calls(scores):
    value=result();value['importance_scores']=scores
    fake=FakeWireClient([wire(value)])
    with pytest.raises(StructuredOutputError):asyncio.run(summary(fake).summarize(MESSAGES))
    assert len(fake.calls)==1


def test_assistant_or_invented_sources_do_not_create_user_facts():
    value=result()
    template=value['items'][0]
    value['items']=[dict(template,source_ids=['a1']),dict(template,text='invented fact',source_ids=['invented'])]
    fake=FakeWireClient([wire(value)])
    output=asyncio.run(summary(fake).summarize(MESSAGES))
    assert output['items']==[]
    assert output['processed_ids']==['u1','a1']


def test_validation_log_contains_paths_but_never_unknown_keys_or_values(caplog):
    value=result()
    value['items'][0].pop('speaker')
    value['items'][0]['private-field-do-not-log']='private-value-do-not-log'
    fake=FakeWireClient([wire(value),wire(value)])
    with pytest.raises(StructuredOutputError):asyncio.run(summary(fake).summarize(MESSAGES))
    assert 'structured_output_invalid' in caplog.text
    assert 'speaker' in caplog.text and '<unknown-field>' in caplog.text
    assert 'private-field-do-not-log' not in caplog.text
    assert 'private-value-do-not-log' not in caplog.text


def mock_http(monkeypatch,handler):
    monkeypatch.setattr(settings,'llm_api_key','mock-not-real')
    monkeypatch.setattr(settings,'llm_base_url','https://summary-typed.invalid/v1')
    monkeypatch.setattr(settings,'llm_model','synthetic-model')
    monkeypatch.setattr(settings,'llm_retry_delays','0,0,0')
    original=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
    client=LLMClient();client.admission.token=AsyncMock()
    return client


@pytest.mark.parametrize('status',[401,402,503])
def test_http_retries_and_semantic_correction_share_two_physical_posts(monkeypatch,status):
    calls=[]
    def handle(request):
        calls.append(json.loads(request.content))
        if len(calls)==1:return httpx.Response(status)
        return httpx.Response(200,json={'choices':[{'message':{'content':'bad result'}}],
                                      'usage':{'prompt_tokens':4,'completion_tokens':2}})
    client=mock_http(monkeypatch,handle)
    with pytest.raises(LLMGatewayError):
        asyncio.run(client.complete_structured([{'role':'user','content':'synthetic input'}],SummaryResult,name='return_memory_summary'))
    assert len(calls)==(2 if status==503 else 1)


def test_truncated_tool_arguments_fail_once_with_specific_protocol_error(monkeypatch):
    calls=[]
    def handle(request):
        calls.append(request)
        return httpx.Response(200,json={'choices':[{'finish_reason':'length','message':{
            'content':None,'tool_calls':[{'type':'function','function':{
                'name':'return_memory_summary','arguments':'{"items":['}}]}}],
            'usage':{'prompt_tokens':4,'completion_tokens':2}})
    client=mock_http(monkeypatch,handle)
    with pytest.raises(LLMGatewayError,match='structured_response_truncated') as caught:
        asyncio.run(client.complete_structured([{'role':'user','content':'synthetic input'}],SummaryResult,name='return_memory_summary'))
    assert caught.value.category==LLMErrorCategory.PROTOCOL
    assert len(calls)==1


def test_six_bounded_messages_and_prior_items_fit_default_background_budget(monkeypatch):
    from app.services.llm.budget import model_scope
    messages=[{'message_id':f'm{i}','role':'user' if i%2==0 else 'assistant','content':'资料'*200} for i in range(6)]
    previous={'items':[{'text':'历史'*90,'category':'transient','speaker':'user',
                       'importance':.5,'confidence':.8,'source_ids':[f'old-{i}']} for i in range(8)]}
    value={'items':[],'importance_scores':[{'message_id':m['message_id'],'score':.1} for m in messages]}
    calls=[]
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'finish_reason':'tool_calls','message':wire(value)}],
            'usage':{'prompt_tokens':20,'completion_tokens':2}})
    client=mock_http(monkeypatch,handle)
    async def run():
        with model_scope('native-summary-size',kind='background'):
            return await summary(client).summarize(messages,previous)
    assert asyncio.run(run())['items']==[]
    assert len(calls)==1
    assert '"properties"' not in calls[0]['messages'][0]['content']
    assert calls[0]['tools'][0]['function']['parameters']['properties']['items']


def queued(factory):
    request=ChatRequest(message='请记住我偏好晨练',request_id='summary-empty-fixture',conversation_id='summary-empty-fixture')
    tid,_=compact.begin_turn(request)
    from app.services.agent.semantics import validate_understanding
    semantic=validate_understanding(request.message, {"tasks":[{
        "task_type":"training_qa","action":"answer","text":request.message,
        "is_negated":False,"is_hypothetical":False,"is_third_party":False}],
        "memory_requests":[{"action":"remember","quote":"我偏好晨练","key":"preference:training_time","confidence":1}]})
    compact.finish_turn(tid,'已记住',{'answer':'已记住','intent':'training_qa'}, semantic_values=[(request.message,semantic)])
    with factory() as session,session.begin():
        row=session.get(MemoryTurn,tid);row.index_status='pending';compact.enqueue(session,row)
    return tid


@pytest.mark.parametrize('filtered',[False,True])
def test_empty_summary_finishes_queue_once_without_embedding_and_keeps_history_facts(compact_db,monkeypatch,filtered):
    from app.services.memory import deepseek_summary as module
    monkeypatch.setattr(settings,'memory_background_enabled',True)
    monkeypatch.setattr(settings,'memory_summary_enabled',True)
    tid=queued(compact_db)
    with compact_db() as session:
        original=session.get(MemoryTurn,tid).messages
    value=result(items=filtered)
    if filtered:value['items'][0]['confidence']=.1
    # Use the real summary parser and safety filters with a synthetic native tool result.
    with compact_db() as session:
        ids=[m['message_id'] for m in session.get(MemoryTurn,tid).messages]
    value['importance_scores']=[{'message_id':mid,'score':.1} for mid in ids]
    if filtered:value['items'][0]['source_ids']=[ids[0]]
    fake=FakeWireClient([wire(value)])
    monkeypatch.setattr(module,'deepseek_summary',summary(fake))
    embed=Mock(side_effect=AssertionError('empty summary must not embed'))
    monkeypatch.setattr(compact.embedding_client,'embed',embed)
    assert asyncio.run(compact.consume_one()) is True
    assert asyncio.run(compact.consume_one()) is False
    assert len(fake.calls)==1
    embed.assert_not_called()
    with compact_db() as session:
        row=session.get(MemoryTurn,tid);job=session.scalar(select(MemoryOutbox))
        assert row.messages==original and row.status=='completed'
        assert row.index_status=='skipped' and row.summary_embedding is None
        assert job.status=='done' and job.attempts==1 and job.last_error_code is None
        assert session.scalar(select(UserFact)).value=='我偏好晨练'


def test_empty_completion_refuses_stale_attempt(compact_db):
    tid=queued(compact_db)
    event=compact.claim_event()
    assert compact.complete_empty_index({**event,'attempt':event['attempt']+1}) is False
    with compact_db() as session:
        assert session.scalar(select(MemoryOutbox)).status=='running'
        assert session.get(MemoryTurn,tid).index_status=='pending'
