from unittest.mock import AsyncMock

import pytest

from app.services.rag.evidence_conflict import EvidenceAnalysis
from evals.v2.safety import deny_network
from tests.react_phase_a_support import tool_call
from tests.react_phase_b_codex_support import environment, chunks, semantic_outline, semantic_task, sse_body

SOURCE='跑步的时候小腿痛'


def test_all_deliverable_domain_types_have_a_read_carrier():
    from app.services.agent.task_models import TaskType, serves_read_goal
    from app.services.agent.react_tools import TOOL_CATALOG
    for goal in TaskType:
        if goal is not TaskType.CHITCHAT:
            assert any(spec.permission=='read' and serves_read_goal(spec.task_type,goal)
                       for spec in TOOL_CATALOG.values()),goal
    assert not serves_read_goal('nutrition_advice','safety_triage')
    assert not serves_read_goal('training_plan','safety_triage')
    assert not serves_read_goal('safety_triage','training_qa')


def setup(env,monkeypatch,kind,*,ref=True,safety='medical',tool='answer_training'):
    from app.services.rag import evidence_conflict,quality
    env.mock_rag(chunks(SOURCE))
    checks=[]
    assess=quality.assess_rag_evidence
    def quality_spy(query,hits,**kwargs):
        checks.append(kwargs.get('task_type'))
        return assess(query,hits,**kwargs)
    monkeypatch.setattr(quality,'assess_rag_evidence',quality_spy)
    analysis=AsyncMock(return_value=EvidenceAnalysis(status='not_applicable'))
    monkeypatch.setattr(evidence_conflict,'analyze_evidence_conflicts',analysis)
    def policy(index,messages):
        args={'query':SOURCE}
        if ref:args['source_ref']={'start':0,'end':len(SOURCE),'exact_text':SOURCE}
        return {'role':'assistant','content':'','tool_calls':[tool_call('answer-'+str(index),tool,args)]}
    llm=env.mock_llm(policy=policy,final='症状相关一般信息（合成测试），不作诊断。',
        semantic=lambda source:semantic_outline(source,[semantic_task(kind,'question',source)],
            safety={'level':safety,'evidence':source if safety!='normal' else '', 'categories':[]}))
    return llm,analysis,checks


@pytest.mark.parametrize('kind',['safety_triage','training_qa'])
@pytest.mark.parametrize('ref',[True,False])
@pytest.mark.parametrize('transport',['json','sse'])
def test_symptom_labels_reach_worker_once_with_same_safety_contract(tmp_path,monkeypatch,kind,ref,transport):
    with deny_network() as audit,environment(tmp_path,monkeypatch,'symptom') as env:
        before=env.snapshot()
        llm,analysis,checks=setup(env,monkeypatch,kind,ref=ref)
        result=env.post('symptom',message=SOURCE) if transport=='json' else sse_body(env.stream('symptom',message=SOURCE))
        assert result['metrics']['react_status']=='complete',result
        assert result['answer']=='症状相关一般信息（合成测试），不作诊断。'
        assert result['safety_level']=='medical'
        assert result['metrics']['react_supervisor_turns']==1 and result['metrics']['react_tool_steps']==1
        assert checks==['safety_triage']
        assert analysis.await_args.kwargs['high_risk'] is True
        assert analysis.await_args.kwargs['task_type']=='safety_triage'
        assert len(env.rag_calls)==1 and len(llm.calls)==2
        assert env.states[-1]['intent_candidates'][0]['task_type']==kind
        calls=len(llm.calls)
        replay=env.post('symptom',message=SOURCE)
        assert replay['answer']==result['answer'] and len(llm.calls)==calls
        assert env.snapshot()==before
    assert audit=={'blocked_connections':0,'blocked_dns':0}


@pytest.mark.parametrize('transport',['json','sse'])
def test_urgent_still_stops_before_any_read_or_answer_worker(tmp_path,monkeypatch,transport):
    with deny_network() as audit,environment(tmp_path,monkeypatch,'urgent') as env:
        llm,analysis,_=setup(env,monkeypatch,'safety_triage',safety='urgent')
        result=env.post('urgent',message=SOURCE) if transport=='json' else sse_body(env.stream('urgent',message=SOURCE))
        assert result['safety_level']=='urgent'
        assert not llm.calls and not env.handlers and not env.rag_calls
        analysis.assert_not_awaited()
    assert audit=={'blocked_connections':0,'blocked_dns':0}


def test_wrong_domain_remains_blocked_and_reports_tool_goal_mismatch(tmp_path,monkeypatch):
    with deny_network() as audit,environment(tmp_path,monkeypatch,'wrong-domain') as env:
        llm,analysis,_=setup(env,monkeypatch,'safety_triage',tool='advise_nutrition')
        result=env.post('wrong-domain',message=SOURCE)
        assert result['metrics']['react_status']=='limited'
        assert not env.handlers and not env.rag_calls
        assert result['metrics']['failure_details'][0]['code']=='no_matching_goal',result
        assert '匹配' in result['answer']
        analysis.assert_not_awaited()
    assert audit=={'blocked_connections':0,'blocked_dns':0}


def test_completed_safety_result_can_be_referenced_without_losing_its_domain(tmp_path,monkeypatch):
    from app.db.memory_models import MemoryTurn
    from app.services.memory import compact
    with deny_network() as audit,environment(tmp_path,monkeypatch,'symptom-reference') as env:
        llm,analysis,_=setup(env,monkeypatch,'safety_triage')
        first=env.post('first-symptom',message=SOURCE)
        assert first['metrics']['react_status']=='complete'
        with env.factory() as s:
            row=s.query(MemoryTurn).filter_by(request_id='first-symptom').one()
            mid=row.messages[-1]['message_id']
            assert row.messages[-1]['app_meta']['result_targets'][0]['task_type']=='safety_triage'
        async def load(user,cid,query=''):
            return {'memory_backend':'compact_v1','recent_result_dialogue':compact.latest_result_dialogue(cid,0)}
        env.load.side_effect=load
        def semantic(source):
            t=semantic_task('safety_triage','question',source)
            t['result_reference']={'message_id':mid,'selection':'latest','quote':source}
            return semantic_outline(source,[t]) | {'relation':'follow_up'}
        llm.semantic=semantic
        llm.policy=lambda index,messages:{'role':'assistant','content':'','tool_calls':[
            tool_call('again-'+str(index),'answer_training',{'query':SOURCE,
                'source_ref':{'start':0,'end':len('再说一次'),'exact_text':'再说一次'}})]}
        again=env.post('repeat-symptom',message='再说一次')
        assert again['metrics']['react_status']=='complete',again
        assert analysis.await_args.kwargs['high_risk'] is True
    assert audit=={'blocked_connections':0,'blocked_dns':0}


def test_actual_auxiliary_read_then_symptom_tool_sequence_finishes(tmp_path,monkeypatch):
    with deny_network() as audit,environment(tmp_path,monkeypatch,'actual-sequence') as env:
        llm,analysis,_=setup(env,monkeypatch,'safety_triage')
        answer_policy=llm.policy
        llm.policy=lambda index,messages:({'role':'assistant','content':'','tool_calls':[
            tool_call('profile','read_profile',{}),tool_call('history','read_workout_history',{'days':7})]}
            if index==0 else answer_policy(index,messages))
        result=env.post('actual-sequence',message=SOURCE)
        assert result['metrics']['react_status']=='complete',result
        assert result['metrics']['react_supervisor_turns']==2
        assert result['metrics']['react_tool_steps']==3
        assert analysis.await_args.kwargs['high_risk'] is True
        assert len(env.rag_calls)==1
    assert audit=={'blocked_connections':0,'blocked_dns':0}


@pytest.mark.parametrize('kind',['training_qa','safety_triage'])
def test_medical_query_keeps_stricter_evidence_floor_for_both_labels(tmp_path,monkeypatch,kind):
    from app.services.rag.retrieval import RetrievedChunk
    with deny_network() as audit,environment(tmp_path,monkeypatch,'evidence-floor') as env:
        llm,analysis,_=setup(env,monkeypatch,kind)
        items=[RetrievedChunk('资料'+str(i),'症状资料片段'+str(i),0.5,'mock',
            {'rerank_status':'model','rerank_score':score},'symptom-'+str(i))
            for i,score in enumerate([0.683,0.6762,0.55,0.4,0.35])]
        env.mock_rag(items)
        result=env.post('evidence-floor',message=SOURCE)
        assert result['metrics']['react_status']=='rejected',result
        assert '证据不足' in result['answer']
        assert len(llm.calls)==1 and len(env.rag_calls)==1
        analysis.assert_not_awaited()
        gates=[x['quality_gate'] for x in result['metrics']['retrieval'] if 'quality_gate' in x]
        assert gates[0]['confidence']==0.5429 and gates[0]['threshold']==pytest.approx(0.58)
    assert audit=={'blocked_connections':0,'blocked_dns':0}
