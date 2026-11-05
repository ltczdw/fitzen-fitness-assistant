import asyncio
from contextlib import asynccontextmanager
import json
from unittest.mock import AsyncMock
import httpx
import pytest

from app.schemas.chat import ChatRequest,ChatResponse
from app.services.agent import runtime
from app.services.agent.transport import stream_response
from app.services.agent.events import emit
from app.services.telemetry import active_metrics,RequestMetrics,model_attempt,mark_degraded
from app.core.config import settings
from app.services.llm.client import LLMClient

class FakeAgent:
 def __init__(self):self.writes=0;self.cached=None;self.shadow=0
 async def _run_planner_shadow(self,message):self.shadow+=1
 @asynccontextmanager
 async def _request_scope(self,request):yield self.cached
 async def _respond_impl(self,request,parsed_input=None):
  self.writes+=1
  emit('plan',{'tasks':[{'id':'t1'}]})
  emit('task_done',{'task_id':'t1','status':'complete'})
  return ChatResponse(answer='完成记录和建议。',intent='workout_record')

def test_json_sse_share_execution_and_do_not_repeat_writes(monkeypatch):
 saved=AsyncMock(return_value='ready');monkeypatch.setattr(runtime,'save_diagnostics',saved)
 async def run():
  agent=FakeAgent();request=ChatRequest(message='记录今天训练')
  response=await runtime.execute_request(agent,request)
  assert agent.writes==1 and agent.shadow==1
  events=[event async for event in stream_response(agent,request)]
  assert agent.writes==2 and agent.shadow==2
  assert ''.join(e['data']['content'] for e in events if e['event']=='token')==response.answer
  assert [e['event'] for e in events]==['status','plan','task_done','meta','token','done']
  assert events[-1]['data']['intent']==response.intent
  agent.cached=response.model_dump()
  replay=await runtime.execute_request(agent,request)
  assert replay.replayed and replay.run_id==response.run_id and replay.metrics==response.metrics
  assert agent.writes==2 and agent.shadow==2 and saved.await_count==2
 asyncio.run(run())

def test_disconnect_cancels_work_without_retry(monkeypatch):
 monkeypatch.setattr(runtime,'save_diagnostics',AsyncMock(return_value='ready'))
 async def run():
  started=asyncio.Event();cancelled=asyncio.Event();agent=FakeAgent()
  async def slow(*args):
   agent.writes+=1;started.set();emit('plan',{'tasks':[]})
   try:await asyncio.Event().wait()
   finally:cancelled.set()
  agent._respond_impl=slow
  stream=stream_response(agent,ChatRequest(message='安排训练'))
  assert (await anext(stream))['event']=='status'
  assert (await anext(stream))['event']=='plan'
  await stream.aclose()
  assert cancelled.is_set() and agent.writes==1
 asyncio.run(run())

def test_metrics_cumulative_usage_missing_and_context_isolation():
 first=RequestMetrics();token=active_metrics.set(first)
 try:
  with model_attempt('one',True) as call:
   for _ in range(2):call.observe({'usage':{'prompt_tokens':30,'completion_tokens':8,'prompt_tokens_details':{'cached_tokens':5}}})
  with model_attempt('two') as call:call.observe({'usage':{'input_tokens':7,'prompt_tokens_details':'bad'}})
  with pytest.raises(RuntimeError):
   with model_attempt('three'):raise RuntimeError('offline')
  mark_degraded('retry')
  result=first.summary()
  assert (result['input_tokens'],result['output_tokens'],result['cached_tokens'])==(37,8,5)
  assert result['model_calls']==3 and result['usage_missing_calls']==2 and not result['usage_complete']
  assert result['calls'][-1]['status']=='error' and result['degraded']
 finally:active_metrics.reset(token)
 assert RequestMetrics().summary()['model_calls']==0

@pytest.mark.parametrize('mode',['text','tools','stream'])
def test_real_gateway_reads_provider_usage(monkeypatch,mode):
 monkeypatch.setattr(settings,'llm_model','fixture-model')
 monkeypatch.setattr(settings,'llm_api_key','test-key');monkeypatch.setattr(settings,'llm_base_url','https://fixture.invalid/v1')
 original=httpx.AsyncClient;seen=[]
 def handle(request):
  payload=json.loads(request.content);seen.append(payload)
  usage={'prompt_tokens':12,'completion_tokens':4}
  if mode=='stream':
   assert payload['stream_options']=={'include_usage':True}
   frames=[{'choices':[{'delta':{'content':'ok'}}]},{'choices':[],'usage':usage},{'choices':[],'usage':usage}]
   return httpx.Response(200,text=''.join('data: '+json.dumps(f)+'\n\n' for f in frames)+'data: [DONE]\n\n')
  return httpx.Response(200,json={'choices':[{'message':{'role':'assistant','content':'ok'}}],'usage':usage})
 monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handle),**kw))
 async def run():
  metrics=RequestMetrics();token=active_metrics.set(metrics)
  try:
   client=LLMClient();messages=[{'role':'user','content':'hello'}]
   if mode=='text':assert await client._complete_once(messages,.2)=='ok'
   elif mode=='tools':assert (await client._complete_message_once(messages,.2,[],None))['content']=='ok'
   else:assert ''.join([part async for part in client._stream_once(messages,.2)])=='ok'
   result=metrics.summary();assert result['model_calls']==1 and result['input_tokens']==12 and result['output_tokens']==4
  finally:active_metrics.reset(token)
 asyncio.run(run());assert len(seen)==1

def test_real_agent_json_and_sse_use_same_graph_and_guard(monkeypatch):
 from types import SimpleNamespace
 from app.services.agent.agent import FitnessAgent
 from app.services.agent import execution
 from app.services.agent.execution_state import PlannerDecision
 from app.services.agent.intent import IntentResult
 from app.services.agent.semantics import RequestUnderstanding
 agent=FitnessAgent()
 source='记录今天的卧推'
 understanding=RequestUnderstanding.model_validate({
  'tasks':[{'task_type':'workout_record','action':'record','text':source,
            'is_negated':False,'is_hypothetical':False,'is_third_party':False,
            'history_requirement':'none','reply_mode':'grounded',
            'reply_purpose':'general'}],
  'safety':{'level':'normal','evidence':'','categories':[],'reason':'scripted normal'},
 })
 async def complete_structured(messages,schema,*,name=None,temperature=0.0,validator=None,**kwargs):
  assert schema is RequestUnderstanding and name=='return_request_understanding'
  if validator is not None:validator(understanding)
  return understanding
 graph=SimpleNamespace(run=AsyncMock(return_value={'final_answer':'已经完成训练记录。','current_task':'workout_record','worker_results':[{'task_id':'t1','status':'complete'}]}),llm=SimpleNamespace(configured=lambda:True,complete_structured=complete_structured))
 agent.graph=graph
 agent._route=AsyncMock(return_value=PlannerDecision(IntentResult('workout_record',1.0),{},({'id':'t1','task_type':'workout_record','depends_on':[]},)))
 agent._persist_turn=AsyncMock();agent._run_planner_shadow=AsyncMock()
 monkeypatch.setattr(settings,'memory_read_backend','legacy')
 monkeypatch.setattr(execution.memory_service,'load',AsyncMock(return_value={'memory_backend':'fixture'}))
 monkeypatch.setattr(runtime,'save_diagnostics',AsyncMock(return_value='ready'))
 async def run():
  req=ChatRequest(message='记录今天的卧推')
  response=await agent.respond(req)
  events=[e async for e in agent.respond_stream(req)]
  assert graph.run.await_count==2 and agent._persist_turn.await_count==2
  assert ''.join(e['data']['content'] for e in events if e['event']=='token')==response.answer
  assert events[-1]['data']['intent']==response.intent=='workout_record'
  graph.run.return_value['worker_results'][0]['status']='needs_confirmation'
  clarification=await agent.respond(req)
  assert not clarification.metrics['degraded']
 asyncio.run(run())

def test_gateway_retry_counts_each_actual_attempt(monkeypatch):
 monkeypatch.setattr(settings,'llm_model','fixture-model')
 monkeypatch.setattr(settings,'llm_api_key','test');monkeypatch.setattr(settings,'llm_base_url','https://retry.invalid/v1');monkeypatch.setattr(settings,'llm_retry_delays','0')
 original=httpx.AsyncClient;calls=[]
 def handle(request):
  calls.append(request)
  if len(calls)==1:return httpx.Response(503)
  return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}],'usage':{'prompt_tokens':4,'completion_tokens':2}})
 monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handle),**kw))
 async def run():
  metrics=RequestMetrics();token=active_metrics.set(metrics)
  try:
   client=LLMClient();client.admission.token=AsyncMock()
   assert await client._complete_impl([{'role':'user','content':'hi'}],.2)=='ok'
   result=metrics.summary()
   assert result['model_calls']==2 and result['usage_missing_calls']==1 and result['degraded']
   assert result['input_tokens']==4 and result['calls'][0]['status']=='error'
  finally:active_metrics.reset(token)
 asyncio.run(run())

def test_embedding_usage_counts_batches_without_fake_completion_tokens(monkeypatch):
 from app.services.rag.embedding_client import EmbeddingClient
 monkeypatch.setattr(settings,'retrieval_embedding_base_url','https://embed.invalid/v1');monkeypatch.setattr(settings,'retrieval_embedding_api_key','test')
 original=httpx.Client
 def handle(request):
  size=len(json.loads(request.content)['input'])
  return httpx.Response(200,json={'data':[{'index':i,'embedding':[1.0]+[0.0]*1023} for i in range(size)],'usage':{'prompt_tokens':size,'total_tokens':size}})
 monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(handle),**kw))
 metrics=RequestMetrics();token=active_metrics.set(metrics)
 try:
  assert len(EmbeddingClient().embed(['text']*21))==21
  result=metrics.summary()
  assert result['calls_by_capability']['embedding']==2 and result['total_tokens']==21 and result['usage_complete']
  assert all(c['output_tokens'] is None for c in result['calls'])
 finally:active_metrics.reset(token)

