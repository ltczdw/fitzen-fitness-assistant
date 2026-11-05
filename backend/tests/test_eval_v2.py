"""Evaluator invariants, not a duplicate test of current application outputs."""
import asyncio
import copy
import json
import socket
from pathlib import Path

import pytest

from evals.v2.dataset import load_cases,validate,ROOT
from evals.v2.scoring import rag_metrics,calibrate,summarize
from evals.v2.safety import deny_network


def test_v2_integrity_and_core_coverage():
    cases=load_cases()
    corpus=json.loads((ROOT/'corpus.json').read_text(encoding='utf-8'))
    assert validate(cases,corpus)==[]
    core=[c for c in cases if c['core']]
    assert len(core)==80
    assert any(c['kind']=='urgent' for c in core)
    assert any(c['kind']=='tool_denied' for c in core)
    assert {c['expected']['answerable'] for c in core if c['kind']=='rag'}=={True,False}


def test_validator_rejects_family_leakage():
    cases=copy.deepcopy(load_cases()[:2])
    cases[1]['split']='test' if cases[0]['split']!='test' else 'development'
    assert any('split leakage' in p for p in validate(cases,require_counts=False))


def test_validator_rejects_duplicate_not_just_duplicate_id():
    first=load_cases()[0]
    duplicate=copy.deepcopy(first);duplicate['id']='changed-id'
    assert any('identical scenario' in p for p in validate([first,duplicate],require_counts=False))


def test_validator_rejects_invented_evidence():
    case=copy.deepcopy(next(c for c in load_cases() if c['kind']=='rag'))
    case['expected']['evidence'][0]['snippet']='invented evidence, not in source'
    corpus=json.loads((ROOT/'corpus.json').read_text(encoding='utf-8'))
    assert any('evidence not in frozen corpus' in p for p in validate([case],corpus,require_counts=False))


def rag_row(cid,confidence,gold,*,split='test',mode='degraded_rules',hit=True):
    return {'id':cid,'family':cid,'category':'rag','split':split,'severity':'normal','status':'passed',
            'actual':{'gold_answerable':gold,'confidence':confidence,'threshold':.48,'mode':mode,
                      'answerable':confidence>=.48,'evidence_recall':1.0 if hit else 0.0,
                      'citation_precision':.5 if hit else None,'mrr':1.0 if hit else 0.0}}


def test_mrr_and_precision_keep_missed_queries_in_denominator():
    result=rag_metrics([rag_row('hit',.6,True),rag_row('miss',0,True,hit=False)])
    assert result['mrr']==.5
    assert result['citation_precision']==.25
    assert result['answer_faithfulness'] is None
    assert result['generation_evaluated'] is False


def test_calibration_ignores_frozen_test_and_development_rows():
    baseline=[rag_row('cal-yes',.40,True,split='calibration'),rag_row('cal-no',.20,False,split='calibration')]
    augmented=baseline+[rag_row('test',.99,False),rag_row('dev',0,True,split='development')]
    assert calibrate(baseline)==calibrate(augmented)
    assert calibrate(baseline)['degraded_rules']['samples']==2


def test_missing_evidence_cannot_be_rescued_by_threshold_calibration():
    rows=[rag_row('missing-positive',0,True,split='calibration',mode='none',hit=False),
          rag_row('missing-negative',0,False,split='calibration',mode='none',hit=False)]
    result=calibrate(rows)['none']
    assert result['status']=='not_calibratable_no_evidence'
    assert 'threshold' not in result


def test_default_bad_gate_is_not_rescued_by_calibration():
    rows=[rag_row('test-positive',.40,True),rag_row('test-negative',.20,False),
          rag_row('cal-positive',.40,True,split='calibration'),rag_row('cal-negative',.20,False,split='calibration')]
    summary=summarize(rows)
    assert summary['calibration']['degraded_rules']['balanced_accuracy']==1
    assert summary['default_acceptance']['rag_answerable_acceptance_at_least_0.85'] is False
    assert summary['release_gate_passed'] is False


def test_network_deny_permits_asyncio_wakeup_but_blocks_app_localhost():
    with deny_network() as audit:
        async def run():
            await asyncio.sleep(0)
            return 1
        assert asyncio.run(run())==1
        with socket.socket() as client:
            with pytest.raises(RuntimeError,match='evaluation_network_prohibited'):
                client.connect(('127.0.0.1',8001))
        with pytest.raises(RuntimeError,match='evaluation_network_prohibited'):
            socket.create_connection(('example.invalid',443))
    assert audit['blocked_connections']==2


def test_real_ledger_case_runs_in_temporary_database():
    from evals.v2.harness import execute
    from app.db import session as db
    previous=db.SessionLocal
    case=next(c for c in load_cases() if c['kind']=='reliability' and c['expected']['scenario']=='ledger_plan')
    with deny_network():result=execute(case)
    assert result['workout_rows']==0
    assert result['write_calls']==0
    assert db.SessionLocal is previous


def test_graph_scenario_semantic_preserves_explicit_write_and_flags():
    from evals.v2.harness import GraphScenario, MonkeyPatch
    from app.services.agent.semantics import RequestUnderstanding

    message = "记录卧推60kg4组8次，然后告诉我怎么恢复"
    case = {"input": {"message": message}, "conditions": {"mock_workout": {
        "exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}},
        "expected": {"safety_level": "normal"}}
    planner = {"tasks": [
        {"task_type": "workout_record", "action": "record",
         "text": "记录卧推60kg4组8次", "is_negated": True,
         "is_hypothetical": False, "is_third_party": False},
        {"task_type": "training_qa", "action": "question",
         "text": "然后告诉我怎么恢复", "is_negated": False,
         "is_hypothetical": False, "is_third_party": False},
    ]}
    with MonkeyPatch.context() as mp:
        scenario = GraphScenario(mp, case, planner=planner)
        value = asyncio.run(scenario.agent.graph.llm.complete_structured(
            [], RequestUnderstanding, name="return_request_understanding"))
    assert [task.task_type for task in value.tasks] == ["workout_record", "training_qa"]
    assert value.tasks[0].action == "record" and value.tasks[0].is_negated is True
    assert value.tasks[1].action == "question"
    assert value.safety.level == "normal"


def test_eval_urgent_uses_model_safety_not_input_guard():
    from evals.v2.harness import execute

    case = next(c for c in load_cases() if c["kind"] == "urgent")
    with deny_network():
        result = execute(case)
    assert result["safety_level"] == "urgent"
    assert result["model_calls"] == 0 and result["write_calls"] == 0
    assert result["model_used"] is False


def test_failed_check_retains_actual_instead_of_being_skipped():
    from evals.v2.harness import verify,CheckFailure
    with pytest.raises(CheckFailure) as caught:
        verify({'workout_rows':2},{'workout_rows':0},['workout_rows'])
    assert caught.value.actual=={'workout_rows':2}
    assert 'expected=0' in caught.value.failures[0]
