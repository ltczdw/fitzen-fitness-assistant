import json

import pytest

from app.services.rag.evidence_conflict import (
    EvidenceClaimModel, EvidenceConditionModel, EvidenceCitationModel, EvidenceConflictModel,
    _select_inputs, _validated_analysis, controlled_evidence_answer, finalize_evidence_answer,
)
from evals.v2.safety import deny_network
from tests.test_rag_evidence_conflict import _chunk
from tests.react_supervisor_fixtures import tool_call
from tests.react_readonly_fixtures import environment, semantic_outline, semantic_task, sse_body

SOURCE='给我制定一个减脂饮食计划'
BACKGROUND='We all need different amounts of energy (or calories) from food to be a healthy weight.'


def hits():
    return [_chunk('均衡饮食','蔬菜水果搭配全谷物与适量蛋白质。'+BACKGROUND,'food'),
            _chunk('食物搭配','三餐选择多样化食物。','meals')]


def judgement(supported=True):
    claims=[EvidenceClaimModel(claim='个性化能量需要',relation='condition_difference',
        evidence=[EvidenceCitationModel(chunk_id='food',quote=BACKGROUND),
                  EvidenceCitationModel(chunk_id='food',quote='蔬菜水果搭配全谷物与适量蛋白质。')],
        conditions=[EvidenceConditionModel(text='未经验证的模型解释',chunk_id='food',quote=BACKGROUND)])]
    if supported:
        claims.append(EvidenceClaimModel(claim='均衡三餐',relation='consistent',evidence=[
            EvidenceCitationModel(chunk_id='food',quote='蔬菜水果搭配全谷物与适量蛋白质。'),
            EvidenceCitationModel(chunk_id='meals',quote='三餐选择多样化食物。')]))
    return EvidenceConflictModel(claims=claims)


@pytest.mark.parametrize('supported',[True,False])
def test_one_generic_condition_is_not_a_comparison_and_not_new_evidence(supported):
    inputs,_=_select_inputs(hits())
    analysis=_validated_analysis(judgement(supported),inputs)
    assert not analysis.has_condition_difference and analysis.has_insufficient
    assert analysis.invalid_relations==1 and analysis.invalid_references==0
    assert analysis.status=='partial'
    answer=controlled_evidence_answer(analysis)
    if supported:
        assert answer is None
        rendered=finalize_evidence_answer('基于有支持部分给出均衡三餐示例。',analysis)
        assert rendered.startswith('基于有支持部分')
        assert '不能合并为同一条件' not in rendered and '模型引用未通过原文核验' not in rendered
    else:
        assert '暂未生成具体建议' in answer


def test_high_risk_unpaired_conditions_keep_conservative_refusal():
    analysis=_validated_analysis(judgement(),_select_inputs(hits())[0])
    answer=controlled_evidence_answer(analysis,high_risk=True)
    assert '暂未生成具体建议' in answer and '不进行医疗诊断' in answer


@pytest.mark.parametrize('same_chunk',[True,False])
def test_two_verified_conditions_and_conclusions_remain_controlled(same_chunk):
    content='初学者每周训练三天。进阶者每周训练五天。'
    items=[_chunk('训练建议',content,'a'),_chunk('第二资料','需要足够恢复。','b')]
    second='a' if same_chunk else 'b'
    if not same_chunk:items[1]=_chunk('第二资料','进阶者每周训练五天。','b')
    raw=EvidenceConflictModel(claims=[EvidenceClaimModel(claim='对照',relation='condition_difference',
        evidence=[EvidenceCitationModel(chunk_id='a',quote='每周训练三天'),
                  EvidenceCitationModel(chunk_id=second,quote='每周训练五天')],
        conditions=[EvidenceConditionModel(text='模型解释不可展示',chunk_id='a',quote='初学者',kind='applicability'),
                    EvidenceConditionModel(text='模型解释不可展示',chunk_id=second,quote='进阶者',kind='applicability')])])
    analysis=_validated_analysis(raw,_select_inputs(items)[0])
    assert analysis.has_condition_difference and not analysis.invalid_relations
    result=finalize_evidence_answer('将两侧合并成四天',analysis)
    assert '不能合并为同一条件' in result and '四天' not in result
    assert '模型解释不可展示' not in result


@pytest.mark.parametrize('transport',['json','sse'])
def test_diet_with_supported_evidence_survives_unpaired_condition(tmp_path,monkeypatch,transport):
    with deny_network() as audit,environment(tmp_path,monkeypatch,'conditions') as env:
        before=env.snapshot()
        env.mock_rag(hits())
        llm=env.mock_llm(policy=lambda index,messages:{'role':'assistant','content':'','tool_calls':[
            tool_call('diet-'+str(index),'advise_nutrition',{'query':SOURCE})]},
            semantic=lambda source:semantic_outline(source,[semantic_task('nutrition_advice','question',source)]),
            final='均衡三餐示例：早餐搭配全谷物，午晚餐搭配蔬菜和适量蛋白质。')
        original=llm.complete_structured
        async def structured(messages,schema,**kwargs):
            if kwargs.get('name')=='return_evidence_conflict_judgement':return judgement()
            return await original(messages,schema,**kwargs)
        monkeypatch.setattr(llm,'complete_structured',structured)
        result=env.post('diet-conditions',message=SOURCE) if transport=='json' else sse_body(env.stream('diet-conditions',message=SOURCE))
        assert result['answer'].startswith('均衡三餐示例'),result
        assert '不能合并为同一条件下的结论' not in result['answer']
        assert result['metrics']['react_status']=='complete' and result['citations']
        generations=[c for c in llm.calls if not c['tools']]
        assert len(generations)==1
        prompt=json.dumps(generations[0]['messages'],ensure_ascii=False)
        assert 'insufficient' in prompt and '已限制为insufficient' in prompt
        count=len(llm.calls)
        assert env.post('diet-conditions',message=SOURCE)['answer']==result['answer']
        assert len(llm.calls)==count and env.snapshot()==before
    assert audit=={'blocked_connections':0,'blocked_dns':0}
