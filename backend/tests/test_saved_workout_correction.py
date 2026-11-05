"""Real ReAct/C1/SQLite correction, with provider-bound mock responses."""
import copy
import json
from uuid import uuid4

import pytest
from sqlalchemy import text

from tests.react_phase_c2_3_support import (
    c3_env, C3HTTPHarness, C3SQLFault, mock_provider_transport, workout_rows, all_business_rows,
)
from tests.test_workout_source_bound_writes import semantic_response, native_tools, native, record, ref_of

ORIGINAL='帮我记录昨天哑推胸15*8*4'
CORRECTION='说错了是哑铃推胸'


def seed(monkeypatch,engine):
    responses=[semantic_response([{'task_type':'workout_record','action':'record',
                'text':ORIGINAL,'fact_text':ORIGINAL}]),
        native_tools([('save','record_workout',{'source_ref':ref_of(ORIGINAL,ORIGINAL)})]),
        native({'records':[record('哑推胸',15,4,8)]})]
    seen=mock_provider_transport(monkeypatch,responses)
    harness=C3HTTPHarness(monkeypatch,engine,write_enabled=True)
    status,saved=harness.post_json(ORIGINAL,request_id=str(uuid4()))
    assert status==200 and len(workout_rows(engine))==1,saved
    return harness,responses,seen,saved


def correction_semantics(source_message_id, *, flag=None, message=CORRECTION,exercise_quote='哑铃推胸'):
    value=semantic_response([{'task_type':'workout_record','action':'record','text':message,
                             'history_requirement':'required'}])
    function=value['choices'][0]['message']['tool_calls'][0]['function']
    args=json.loads(function['arguments'])
    args['relation']='correction'
    args['dialogue_ids']=[source_message_id]
    task=args['tasks'][0]
    task['context_ids']=[source_message_id]
    task['workout_correction']={'source_message_id':source_message_id,
                              'exercise_quote':exercise_quote,'original_exercise':None}
    if flag:task[flag]=True
    function['arguments']=json.dumps(args,ensure_ascii=False)
    return value


def append_correction(responses,saved,*,tool='correct_workout',flag=None,narrow=False):
    responses.extend([correction_semantics(saved['message_id'],flag=flag),
        native_tools([('correct',tool,{'source_ref':ref_of(CORRECTION,'哑铃推胸' if narrow else CORRECTION)})])])


@pytest.mark.parametrize('backend',['sqlite','postgres'])
@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('narrow',[False,True])
def test_rename_preserves_fields_mirrors_and_replays(c3_env,monkeypatch,backend,stream,narrow):
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    before=workout_rows(c3_env)
    append_correction(responses,saved,narrow=narrow)
    rid=str(uuid4())
    if stream:
        status,events,_=harness.stream(CORRECTION,request_id=rid)
        result={**events[-1],'answer':''.join(e.get('content','') for e in events)}
    else:status,result=harness.post_json(CORRECTION,request_id=rid)
    assert status==200,result
    after=workout_rows(c3_env)
    assert after==[{**before[0],'exercise':'哑铃推胸'}],result
    assert '已将训练记录的动作名' in result['answer'] and '保持不变' in result['answer'],result
    assert result['metrics']['react_write']=='completed' and 'workout_name_update_v1' in result['metrics']['receipt_refs']
    with c3_env.connect() as connection:
        names=connection.execute(text('SELECT exercise_name FROM workout_session_exercises WHERE source_record_id=:id'),{'id':before[0]['id']}).scalars().all()
    assert names==['哑铃推胸']
    assert len(seen)==5
    state=all_business_rows(c3_env)
    status,replayed=harness.post_json(CORRECTION,request_id=rid)
    assert status==200 and replayed['answer']==result['answer'],replayed
    assert replayed['metrics']['receipt_refs']==result['metrics']['receipt_refs']
    assert all_business_rows(c3_env)==state and len(seen)==5


@pytest.mark.parametrize('backend',['sqlite','postgres'])
def test_correction_cannot_be_dispatched_as_insert(c3_env,monkeypatch,backend):
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    before=all_business_rows(c3_env)
    append_correction(responses,saved,tool='record_workout')
    _,result=harness.post_json(CORRECTION,request_id=str(uuid4()))
    assert all_business_rows(c3_env)==before and '已将训练记录' not in result['answer']


@pytest.mark.parametrize('backend',['sqlite','postgres'])
@pytest.mark.parametrize('flag',['is_negated','is_hypothetical','is_third_party'])
def test_correction_honors_write_authorization(c3_env,monkeypatch,backend,flag):
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    before=all_business_rows(c3_env)
    append_correction(responses,saved,flag=flag)
    _,result=harness.post_json(CORRECTION,request_id=str(uuid4()))
    assert all_business_rows(c3_env)==before and '已将训练记录' not in result['answer']


@pytest.mark.parametrize('backend',['sqlite','postgres'])
def test_atomic_rollback_on_correction_proof_failure(c3_env,monkeypatch,backend):
    from app.services.agent import react_graph
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    monkeypatch.setattr(react_graph,'MAX_SUPERVISOR_TURNS',1)
    before=all_business_rows(c3_env)
    append_correction(responses,saved)
    fault=C3SQLFault(c3_env,table='agent_execution_events',action='insert',
        occurrence=1,match=lambda statement,parameters:'workout-name-correction' in str(parameters))
    fault.install()
    try:_,result=harness.post_json(CORRECTION,request_id=str(uuid4()))
    finally:fault.remove()
    assert fault.raised
    assert all_business_rows(c3_env)==before and '已将训练记录' not in result['answer']


@pytest.mark.parametrize('backend',['sqlite','postgres'])
@pytest.mark.parametrize('phase',['tool_result','final_reply'])
def test_resume_after_real_commit_does_not_repeat_mutation(c3_env,monkeypatch,backend,phase):
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    append_correction(responses,saved)
    rid=str(uuid4())
    fault=C3SQLFault(c3_env,table='agent_execution_events',action='insert',occurrence=1,
        match=lambda statement,parameters:phase in str(parameters) and rid in str(parameters))
    fault.install()
    try:harness.post_json(CORRECTION,request_id=rid)
    finally:fault.remove()
    assert fault.raised and workout_rows(c3_env)[0]['exercise']=='哑铃推胸'
    with c3_env.connect() as connection:
        proofs=connection.execute(text("SELECT count(*) FROM agent_execution_events WHERE request_id=:r AND event_key LIKE 'workout-name-correction:v1:%'"),{'r':rid}).scalar_one()
    assert proofs==1
    calls=len(seen)
    status,result=harness.post_json(CORRECTION,request_id=rid)
    assert status==200 and result['metrics'].get('react_write')=='completed',result
    assert len(seen)==calls and '已将训练记录' in result['answer']
    assert len(workout_rows(c3_env))==1


@pytest.mark.parametrize('backend',['sqlite','postgres'])
@pytest.mark.parametrize('mode',['foreign_user','foreign_conversation','hidden','deleted','epoch'])
def test_correction_source_requires_live_owned_visible_turn(c3_env,monkeypatch,backend,mode):
    from app.services.agent.workout_correction import WorkoutCorrectionBridge,CorrectionError
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    uid='demo-user'
    cid=harness.conversation_id
    if mode=='foreign_user':uid='other-user'
    elif mode=='foreign_conversation':cid='other-conversation'
    else:
        with c3_env.begin() as connection:
            if mode=='epoch':
                connection.execute(text('UPDATE user_profiles SET memory_epoch=memory_epoch+1 WHERE user_id=:u'),{'u':uid})
            else:
                column='hidden_at' if mode=='hidden' else 'deleted_at'
                connection.execute(text(f'UPDATE memory_turns SET {column}=CURRENT_TIMESTAMP WHERE turn_id=:t'),{'t':saved['message_id'].rpartition(':')[0]})
    before=all_business_rows(c3_env)
    with pytest.raises(CorrectionError):
        WorkoutCorrectionBridge(__import__('sqlalchemy').orm.sessionmaker(bind=c3_env)).resolve(uid,cid,
            {'source_message_id':saved['message_id'],'exercise_quote':'哑铃推胸','original_exercise':None})
    assert all_business_rows(c3_env)==before


@pytest.mark.parametrize('backend',['sqlite','postgres'])
@pytest.mark.parametrize('target',[None,'哑推胸'])
def test_multiple_source_actions_require_unique_selection(c3_env,monkeypatch,backend,target):
    from app.services.agent.workout_correction import WorkoutCorrectionBridge,CorrectionError
    from sqlalchemy.orm import sessionmaker
    source='保存哑推胸15*8*4；哑铃划船10*12*3'
    responses=[semantic_response([{'task_type':'workout_record','action':'record','text':source,
        'fact_text':source,'workout_sources':[{'exercise':'哑推胸','quote':'哑推胸15*8*4'},
                                            {'exercise':'哑铃划船','quote':'哑铃划船10*12*3'}]}]),
        native_tools([('save','record_workout',{'source_ref':ref_of(source,source)})]),
        native({'records':[record('哑推胸',15,4,8),record('哑铃划船',10,3,12)]})]
    seen=mock_provider_transport(monkeypatch,responses)
    harness=C3HTTPHarness(monkeypatch,c3_env,write_enabled=True)
    _,saved=harness.post_json(source,request_id=str(uuid4()))
    assert len(workout_rows(c3_env))==2,saved
    bridge=WorkoutCorrectionBridge(sessionmaker(bind=c3_env))
    correction={'source_message_id':saved['message_id'],'exercise_quote':'哑铃推胸','original_exercise':target}
    if target is None:
        with pytest.raises(CorrectionError,match='correction_target_ambiguous'):
            bridge.resolve('demo-user',harness.conversation_id,correction)
    else:
        selected=bridge.resolve('demo-user',harness.conversation_id,correction)
        assert selected['expected']['exercise']=='哑推胸'
    sem=correction_semantics(saved['message_id'])
    function=sem['choices'][0]['message']['tool_calls'][0]['function']
    args=json.loads(function['arguments'])
    args['tasks'][0]['workout_correction']['original_exercise']=target
    function['arguments']=json.dumps(args,ensure_ascii=False)
    responses.extend([sem,native_tools([('correct','correct_workout',{
        'source_ref':ref_of(CORRECTION,CORRECTION)})])])
    before=workout_rows(c3_env)
    _,result=harness.post_json(CORRECTION,request_id=str(uuid4()))
    after=workout_rows(c3_env)
    assert len(after)==2
    if target is None:
        assert after==before and '请明确' in result['answer'],result
    else:
        assert after==[{**before[0],'exercise':'哑铃推胸'},before[1]],result


@pytest.mark.parametrize('backend',['sqlite','postgres'])
def test_concurrent_change_is_preserved_and_requires_review(c3_env,monkeypatch,backend):
    from app.services.agent.workout_correction import WorkoutCorrectionBridge
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    original=WorkoutCorrectionBridge.resolve
    def resolve(self,*args,**kwargs):
        params=original(self,*args,**kwargs)
        with c3_env.begin() as connection:
            connection.execute(text('UPDATE workout_records SET note=:n WHERE id=:id'),
                               {'n':'concurrent edit','id':params['record_id']})
        return params
    monkeypatch.setattr(WorkoutCorrectionBridge,'resolve',resolve)
    append_correction(responses,saved)
    _,result=harness.post_json(CORRECTION,request_id=str(uuid4()))
    assert workout_rows(c3_env)[0]['exercise']=='哑推胸'
    assert '已发生变化' in result['answer'],result
    with c3_env.connect() as connection:
        assert connection.execute(text('SELECT note FROM workout_records')).scalar_one()=='concurrent edit'


@pytest.mark.parametrize('backend',['sqlite','postgres'])
def test_name_only_update_preserves_heterogeneous_set_rows(c3_env,monkeypatch,backend):
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    with c3_env.begin() as connection:
        detail=[{'weight_kg':15+i,'reps':8+i,'completed':True} for i in range(4)]
        connection.execute(text('UPDATE workout_records SET set_details=:d,note=:n'),
                           {'d':json.dumps(detail),'n':'retain note'})
        connection.execute(text('UPDATE workout_session_sets SET weight_kg=weight_kg+set_index,reps=reps+set_index'))
    def sets():
        with c3_env.connect() as connection:
            return [dict(r._mapping) for r in connection.execute(text('SELECT * FROM workout_session_sets ORDER BY id'))]
    before_sets=sets()
    before=workout_rows(c3_env)
    append_correction(responses,saved)
    _,result=harness.post_json(CORRECTION,request_id=str(uuid4()))
    assert '已将训练记录' in result['answer'],result
    assert workout_rows(c3_env)==[{**before[0],'exercise':'哑铃推胸'}]
    assert sets()==before_sets


@pytest.mark.parametrize('backend',['sqlite','postgres'])
def test_second_correction_uses_latest_committed_receipt(c3_env,monkeypatch,backend):
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    append_correction(responses,saved)
    _,updated=harness.post_json(CORRECTION,request_id=str(uuid4()))
    assert workout_rows(c3_env)[0]['exercise']=='哑铃推胸',updated
    followup='名称改成哑铃卧推'
    responses.extend([correction_semantics(updated['message_id'],message=followup,exercise_quote='哑铃卧推'),
        native_tools([('correct-again','correct_workout',{'source_ref':ref_of(followup,followup)})])])
    _,result=harness.post_json(followup,request_id=str(uuid4()))
    assert len(workout_rows(c3_env))==1 and workout_rows(c3_env)[0]['exercise']=='哑铃卧推',result
    assert result['metrics']['react_write']=='completed' and len(seen)==7


@pytest.mark.parametrize('backend',['sqlite','postgres'])
@pytest.mark.parametrize('mode',['auto','react_readonly','legacy'])
def test_entry_modes_share_the_correction_writer(c3_env,monkeypatch,backend,mode):
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    append_correction(responses,saved)
    _,result=harness.post_json(CORRECTION,request_id=str(uuid4()),mode=mode)
    assert len(workout_rows(c3_env))==1 and workout_rows(c3_env)[0]['exercise']=='哑铃推胸',result
    assert result['metrics']['execution_mode']=='react_write' and len(seen)==5


@pytest.mark.parametrize('backend',['sqlite','postgres'])
def test_public_ledger_cannot_forge_correction_proof(c3_env,monkeypatch,backend):
    from sqlalchemy.orm import sessionmaker
    from app.services.agent import execution_ledger_store as ledger
    from app.services.agent.workout_correction import WorkoutCorrectionBridge
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    store=ledger.ExecutionLedgerStore(sessionmaker(bind=c3_env))
    state=store.load_request('demo-user',saved['request_id'])
    current=state['checkpoint']
    checkpoint=ledger.Checkpoint(supervisor_turns=current['supervisor_turns'],
        tool_steps=current['tool_steps'],event_sequence=current['event_sequence']+1,
        budget_scope_id=state['identity']['budget_scope_id'])
    event=ledger.EventInput(event_key=ledger._CORRECTION_EVENT_PREFIX+'fake',
        tool_call_id='fake',batch_id='fake',phase='confirmation',observation={},
        task_id='fake',op_id=None,task_status='pending',operation_state=None,confirmed=False)
    before=all_business_rows(c3_env)
    with pytest.raises(ledger.ExecutionLedgerError) as caught:
        store.append_observation('demo-user',saved['request_id'],event,checkpoint=checkpoint,
            expected_revision=state['revision'],lease=ledger.LeaseProof('fake','fake',1))
    assert caught.value.reason=='reserved_event_key'
    bridge=WorkoutCorrectionBridge(sessionmaker(bind=c3_env))
    with bridge.factory() as session,pytest.raises(ledger.ExecutionLedgerError) as caught:
        bridge.store.ledger_append_correction_event(session,'demo-user',saved['request_id'],event,
            checkpoint=checkpoint,expected_revision=state['revision'],
            lease=ledger.LeaseProof('fake','fake',1),capability=object())
    assert caught.value.reason=='reserved_capability_required'
    assert all_business_rows(c3_env)==before


@pytest.mark.parametrize('backend',['sqlite','postgres'])
@pytest.mark.parametrize('mutation',['status','commit_state','task_id','record_id'])
def test_corrupt_confirmation_is_never_replayed_as_success(c3_env,monkeypatch,backend,mutation):
    from sqlalchemy.orm import sessionmaker
    from app.db.execution_ledger_models import ExecutionEventModel
    from sqlalchemy import select
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    append_correction(responses,saved)
    rid=str(uuid4())
    _,result=harness.post_json(CORRECTION,request_id=rid)
    assert result['metrics']['react_write']=='completed'
    with sessionmaker(bind=c3_env)() as session:
        event=session.scalar(select(ExecutionEventModel).where(
            ExecutionEventModel.request_id==rid,
            ExecutionEventModel.event_key.like('workout-name-correction:v1:%')))
        value=copy.deepcopy(event.observation)
        if mutation=='record_id':value['result']['record_id']=True
        else:value[mutation]={'status':'unavailable','commit_state':None,'task_id':'foreign'}[mutation]
        event.observation=value
        # JSON change tracking treats Python True == 1; force the intentional
        # corrupt persisted type rather than accidentally testing unchanged data.
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(event,'observation')
        session.commit()
        if mutation=='record_id':
            session.refresh(event)
            assert event.observation['result']['record_id'] is True
    before=all_business_rows(c3_env)
    calls=len(seen)
    _,replayed=harness.post_json(CORRECTION,request_id=rid)
    assert replayed['metrics'].get('react_write')!='completed' and '已将训练记录' not in replayed['answer']
    assert len(seen)==calls and all_business_rows(c3_env)==before


@pytest.mark.parametrize('backend',['sqlite','postgres'])
def test_dispatched_state_cannot_hide_existing_correction_commit(c3_env,monkeypatch,backend):
    from sqlalchemy.orm import sessionmaker
    from app.db.execution_ledger_models import ExecutionOperationModel
    from sqlalchemy import select
    from app.services.agent import execution_ledger_store as ledger
    from app.services.agent.react_write_runtime import _bridge_ops_proven
    harness,responses,seen,saved=seed(monkeypatch,c3_env)
    append_correction(responses,saved)
    rid=str(uuid4())
    _,result=harness.post_json(CORRECTION,request_id=rid)
    assert result['metrics']['react_write']=='completed'
    factory=sessionmaker(bind=c3_env)
    with factory() as session:
        op=session.scalar(select(ExecutionOperationModel).where(ExecutionOperationModel.request_id==rid))
        op.status='dispatch_intent'
        session.commit()
    state=ledger.ExecutionLedgerStore(factory).load_request('demo-user',rid)
    assert _bridge_ops_proven('demo-user',rid,state,factory,allow_reserved=True) is False
