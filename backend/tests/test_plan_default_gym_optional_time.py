import asyncio,json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx,pytest
from app.services.agent.task_information import PlanInformation,check_plan_information
from app.services.agent.training_structure import parse_untimed_plan,render_untimed_plan
from tests.test_workout_save_repair_20261102 import c3_env,C3HTTPHarness,mock_provider_transport,semantic_response,native_tools
from tests.test_react_record_continuation import tool_response


def information(source,**extra):
    return {'scope':'session','facts':[
        {'field':'goal','source_id':'current','quote':'增肌'},
        {'field':'experience','source_id':'current','quote':'训练一年'}],**extra}


@pytest.mark.parametrize('mode',['auto','request','omit'])
def test_equipment_default_is_policy_and_time_is_never_required(mode):
    source='增肌，训练一年，请给训练计划'+({'request':'并安排时间','omit':'但不要时间'}.get(mode,''))
    value=information(source,timing_action=mode,timing_quote={'request':'安排时间','omit':'不要时间'}.get(mode,''))
    llm=SimpleNamespace(complete_structured=AsyncMock(return_value=PlanInformation.model_validate(value)))
    details,question=asyncio.run(check_plan_information(llm,{'raw_message':source,'task_text':source}))
    assert question is None and details['known']['equipment']=='健身房'
    assert details['defaulted_fields']==['equipment'] and 'session_time' not in details['known']
    assert details['timing_requested']==(mode=='request')


@pytest.mark.parametrize('equipment',['只有弹力带','徒手','不去健身房'])
def test_explicit_equipment_preserved_without_time_question(equipment):
    source='增肌，训练一年，'+equipment+'，请给训练计划'
    value=information(source)
    value['facts'].append({'field':'equipment','source_id':'current','quote':equipment})
    llm=SimpleNamespace(complete_structured=AsyncMock(return_value=PlanInformation.model_validate(value)))
    details,question=asyncio.run(check_plan_information(llm,{'raw_message':source,'task_text':source}))
    assert question is None and details['known']['equipment']==equipment
    assert not details.get('defaulted_fields') and details['timing_requested'] is False


def test_omitting_time_display_does_not_discard_an_explicit_limit():
    source='增肌，训练一年，每次不超过30分钟，不要时间表，给我训练计划'
    value=information(source,timing_action='omit',timing_quote='不要时间表')
    value['facts'].append({'field':'session_time','source_id':'current','quote':'每次不超过30分钟',
                           'quantity_quote':'30分钟','seconds':1800})
    llm=SimpleNamespace(complete_structured=AsyncMock(return_value=PlanInformation.model_validate(value)))
    details,question=asyncio.run(check_plan_information(llm,{'raw_message':source,'task_text':source}))
    assert question is None and details['timing_requested'] is False
    assert details['session_time_seconds']==1800


def test_old_per_day_preview_caps_remain_without_requesting_a_new_duration():
    from app.services.agent.multi_agent_graph import _preview_session_limits
    old={'days':[{'label':'推','minutes':30},{'label':'拉','minutes':20}]}
    assert _preview_session_limits(old,2,'preview',None,None)==[1800,1200]
    assert _preview_session_limits({'days':[{'label':'推','minutes':None}]},1,'preview',None,None) is None
    assert _preview_session_limits({'days':[{'label':'推','minutes':None},{'label':'拉','minutes':None}]},
        2,'preview',900,'拉')==[None,900]


def test_pending_questions_drop_only_retired_equipment_and_time_fields():
    from app.services.agent.planning_questions import current_questions
    from app.services.agent.task_information import question
    old={'ready':question('training_plan',['equipment','session_time'],task_id='ready').model_dump(),
         'needs':question('training_plan',['goal','session_time','experience'],task_id='needs').model_dump()}
    projected=current_questions(old)
    assert 'ready' not in projected
    assert [f['name'] for f in projected['needs']['fields']]==['goal','experience']
    assert len(old['ready']['fields'])==2


def test_untimed_display_keeps_explicit_internal_timing_validation():
    from app.services.agent.training_timing import parse_timed_plan
    value={'sessions':[{'label':'训练日','warmup':{'seconds':120,'activity':'动态活动'},
        'main':{'seconds':900,'exercises':[{'name':'深蹲','sets':3,'reps':10}]},
        'breaks':{'seconds':120},'cooldown':{'seconds':120,'activity':'拉伸'},
        'time_consistency':'consistent'}]}
    plan=parse_timed_plan(json.dumps(value),limit_seconds=1800,expected_sessions=1)
    answer=render_untimed_plan(plan)
    assert '深蹲' in answer and '| 3 | 10 |' in answer
    assert '分钟' not in answer and '秒' not in answer and '用时' not in answer
    with pytest.raises(ValueError):parse_timed_plan(json.dumps(value),limit_seconds=1200,expected_sessions=1)


def structure():
    return {'sessions':[{'label':label,'warmup':['动态活动'],'main':{'exercises':[
        {'name':name,'sets':3,'reps':10}]},'cooldown':['轻柔拉伸'],'without_timing':True}
        for label,name in [('推','卧推'),('拉','划船'),('腿','深蹲')]]}


@pytest.mark.parametrize('mutation',['seconds','hold_seconds','extra_session','false_check','numeric_check','empty_activity'])
def test_untimed_contract_rejects_timing_fields_or_incomplete_structure(mutation):
    value=structure()
    if mutation=='seconds':value['sessions'][0]['seconds']=100
    elif mutation=='hold_seconds':value['sessions'][0]['main']['exercises'][0]['hold_seconds']=30
    elif mutation=='extra_session':value['sessions'].pop()
    elif mutation=='false_check':value['sessions'][0]['without_timing']=False
    elif mutation=='numeric_check':value['sessions'][0]['without_timing']=1
    else:value['sessions'][0]['warmup']=[' ']
    with pytest.raises(ValueError):parse_untimed_plan(json.dumps(value),expected_sessions=3)


@pytest.mark.parametrize('timed',[False,True])
@pytest.mark.parametrize('failure',['budget','provider'])
def test_generation_preserves_typed_failures(monkeypatch,timed,failure):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.llm.budget import ModelBudgetError
    from app.services.llm.client import LLMGatewayError,LLMErrorCategory
    from tests.test_training_timing import wire_worker,raw_plan
    graph=FitnessMultiAgentGraph()
    wire_worker(monkeypatch,graph,raw_plan())
    info={'scope':'session','known':{'goal':'增肌','experience':'新手','equipment':'健身房'},
          'timing_requested':timed,'session_time_seconds':1200 if timed else None}
    monkeypatch.setattr('app.services.agent.task_information.check_plan_information',
                        AsyncMock(return_value=(info,None)))
    error=ModelBudgetError('token_limit') if failure=='budget' else LLMGatewayError(
        'fixture',LLMErrorCategory.UPSTREAM,status_code=503,code='provider_503')
    names=[]
    async def stop(messages,**kwargs):
        names.append(kwargs['tools'][0]['function']['name'])
        raise error
    monkeypatch.setattr(graph.llm,'complete_message',stop)
    with pytest.raises(type(error)):
        asyncio.run(graph.workers['plan_adjust'].ainvoke({'raw_message':'制定训练计划','task_text':'制定训练计划'}))
    assert names==['return_timed_training_plan' if timed else 'return_training_plan']


@pytest.mark.parametrize('backend',['sqlite','postgres'])
@pytest.mark.parametrize('transport',['json','sse'])
def test_real_default_plan_preview_then_explicit_time_without_limit(c3_env,monkeypatch,backend,transport):
    from app.db import profile_repository
    from app.services.agent.agent import agent
    from app.services.rag.retrieval import RetrievedChunk
    profile_repository.save_profile('demo-user',{'goal':'增肌','experience':'中级'})
    source='给我制定三分化训练计划'
    edit='把这个计划改成二分化'
    follow='给这个计划安排时间'
    calls=[];queries=[]
    active={'count':3}
    def current_structure():
        value=structure()
        if active['count']==2:
            value['sessions']=[value['sessions'][0],value['sessions'][2]]
            value['sessions'][0]['label']='上肢'
            value['sessions'][1]['label']='下肢'
        return value
    def respond(request):
        body=json.loads(request.content)
        names=[t['function']['name'] for t in body.get('tools') or []]
        calls.append(names)
        if 'return_request_understanding' in names:
            data=json.loads(body['messages'][-1]['content']);query=data['query']
            queries.append(query)
            result=semantic_response([{'task_type':'training_plan','action':'question','text':query}])
            f=result['choices'][0]['message']['tool_calls'][0]['function'];args=json.loads(f['arguments'])
            task=args['tasks'][0]
            task.update(plan_result=True,result_scope='independent',plan_request_quote=query,delivery_type='plan')
            if query in (edit,follow):
                task.update(delivery_type='preview',preview_ref=data['current_plan_preview']['preview_id'])
            f['arguments']=json.dumps(args,ensure_ascii=False)
            return httpx.Response(200,json=result)
        if 'return_plan_information' in names:
            data=json.loads(body['messages'][-1]['content']);text=data['task']
            facts=[{'field':field,'source_id':'profile:'+field,'quote':data['sources']['profile:'+field]}
                   for field in ('goal','experience')]
            value={'scope':'cycle','scope_quote':'三分化','cycle_sessions':3,'cycle_quote':'三分化',
                   'cycle_quantity_quote':'三','facts':facts} if '三分化' in text else {
                       'scope':'session','facts':facts,'timing_action':'request','timing_quote':'安排时间'}
            if '二分化' in text:
                value={'scope':'cycle','scope_quote':'二分化','cycle_sessions':2,
                       'cycle_quote':'二分化','cycle_quantity_quote':'二','facts':facts}
                active['count']=2
            return httpx.Response(200,json=tool_response('return_plan_information',value))
        if 'return_training_plan' in names:
            assert '健身房' in json.dumps(body,ensure_ascii=False)
            return httpx.Response(200,json=tool_response('return_training_plan',current_structure()))
        if 'return_timed_training_plan' in names:
            value={'sessions':[{'label':s['label'],'warmup':{'seconds':300,'activity':'动态活动'},
                'main':{'seconds':1800,'exercises':s['main']['exercises']},'breaks':{'seconds':300},
                'cooldown':{'seconds':300,'activity':'轻柔拉伸'},'time_consistency':'consistent'} for s in current_structure()['sessions']]}
            return httpx.Response(200,json=tool_response('return_timed_training_plan',value))
        if 'plan_training' in names:
            return httpx.Response(200,json=native_tools([('plan','plan_training',{'query':queries[-1]})]))
        raise AssertionError(names)
    mock_provider_transport(monkeypatch,[respond])
    hit=RetrievedChunk('训练原则','逐步增加训练量，保持动作稳定。',.9,'official',
        {'rerank_score':.9,'rerank_status':'model'},'default-plan')
    monkeypatch.setattr(agent.graph,'_readonly_retrieval',AsyncMock(return_value=[hit]))
    harness=C3HTTPHarness(monkeypatch,c3_env)
    def send(message,rid):
        if transport=='json':return harness.post_json(message,request_id=rid,mode='react_readonly')
        status,events,_=harness.stream(message,request_id=rid,mode='react_readonly')
        return status,{**events[-1],'answer':''.join(e.get('content','') for e in events)}
    rid=str(uuid4());status,first=send(source,rid)
    assert status==200 and not first['clarifications'],first
    assert all(x in first['answer'] for x in ('卧推','划船','深蹲'))
    assert '分钟' not in first['answer'] and '秒' not in first['answer'] and '用时' not in first['answer']
    count=len(calls);_,replay=send(source,rid)
    assert replay['answer']==first['answer'] and len(calls)==count
    status,edited=send(edit,str(uuid4()))
    assert status==200 and not edited['clarifications'],edited
    assert '上肢' in edited['answer'] and '下肢' in edited['answer']
    assert '分钟' not in edited['answer'] and '用时' not in edited['answer']
    status,second=send(follow,str(uuid4()))
    assert status==200 and not second['clarifications'] and '分钟' in second['answer'],second
    assert '未指定单次时长上限' in second['answer'] and '86400' not in second['answer']
