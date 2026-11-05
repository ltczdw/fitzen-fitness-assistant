"""Evidence refusal must end a read goal without retrying unchanged retrieval."""
import pytest
from unittest.mock import AsyncMock
from evals.v2.safety import deny_network
from tests.react_phase_a_support import tool_call
from tests.react_phase_b_codex_support import (
    environment, chunks, semantic_outline, semantic_task, sse_body,
)

SOURCE = '如果花生过敏，日常需要注意哪些食物'


@pytest.mark.parametrize('transport', ['json', 'sse'])
@pytest.mark.parametrize('kind,tool', [('nutrition_advice','advise_nutrition'),
                                      ('training_qa','answer_training')])
@pytest.mark.parametrize('evidence', ['empty','poor','sufficient'])
def test_read_goal_evidence_outcome_is_terminal(tmp_path, monkeypatch, transport, kind, tool, evidence):
    with deny_network() as audit, environment(tmp_path, monkeypatch, 'terminal') as env:
        before = env.snapshot()
        env.mock_rag([] if evidence == 'empty' else chunks(SOURCE, sufficient=evidence == 'sufficient'))
        def semantic(source):
            return semantic_outline(source, [semantic_task(kind,'question',source,hypothetical=True)],
                safety={'level':'medical','evidence':source,'categories':[]})
        def policy(index, messages):
            assert index == 0, 'an evidence outcome must not request another supervisor turn'
            return {'role':'assistant','content':'','tool_calls':[
                tool_call('answer',tool,{'query':SOURCE})]}
        llm = env.mock_llm(policy=policy, semantic=semantic, final='这是合成资料支持的回答。')
        result = (env.post('terminal-request',message=SOURCE) if transport == 'json' else
                  sse_body(env.stream('terminal-request',message=SOURCE)))
        expected = 'complete' if evidence == 'sufficient' else 'rejected'
        assert result['metrics']['react_status'] == expected, result
        assert result['metrics']['react_termination_reason'] == 'goals_complete'
        assert result['metrics']['react_supervisor_turns'] == 1
        assert result['metrics']['react_tool_steps'] == 1
        assert result['safety_level'] == 'medical'
        assert len(env.rag_calls) == 1
        assert len(llm.calls) == (2 if evidence == 'sufficient' else 1)
        assert result['answer'] == ('这是合成资料支持的回答。' if evidence == 'sufficient' else
            env.handlers[-1]['observation']['summary'])
        assert bool(result['citations']) == (evidence == 'sufficient')
        calls = len(llm.calls)
        replay = env.post('terminal-request',message=SOURCE)
        assert replay['answer'] == result['answer'] and len(llm.calls) == calls
        assert env.snapshot() == before
    assert audit == {'blocked_connections':0,'blocked_dns':0}


@pytest.mark.parametrize('tool',['advise_nutrition','answer_training','plan_training'])
def test_retrieval_outage_is_not_misreported_as_completed_evidence_check(tmp_path,monkeypatch,tool):
    from app.services.agent import react_readonly_workers as workers
    from tests.react_phase_b_codex_support import obs
    from app.services.agent import task_information
    with deny_network() as audit, environment(tmp_path,monkeypatch,'outage') as env:
        env.mock_rag([])
        async def outage(query):
            raise RuntimeError('synthetic retrieval outage')
        env.mock_rag([],callback=outage)
        llm=env.mock_llm()
        if tool=='plan_training':
            monkeypatch.setattr(task_information,'check_plan_information',AsyncMock(return_value=({},None)))
        result=obs(getattr(workers,tool)({'query':SOURCE},{'source_text':SOURCE}))
        assert result['status']=='unavailable' and result['error']=='knowledge_retrieval_failed'
        assert not llm.calls
    assert audit=={'blocked_connections':0,'blocked_dns':0}
