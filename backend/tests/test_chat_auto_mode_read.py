"""Omitted mode (auto) keeps a pure read on react_readonly and reads real SQL.

The provider is scripted; the identity check, tool schema, readonly catalog and
the saved-workout SQL read stay real. The user's recorded workout is seeded
directly, then the same user asks what they did today.
"""
import json

from tests.react_phase_a_support import tool_call
from tests.react_phase_b_codex_support import (
    environment, semantic_outline, semantic_task, tool_results,
)

SOURCE = "我今天练了什么？"
ANSWER = "你今天练了卧推 60kg 4组8次。"


def simulated_semantics(source):
    payload = semantic_outline(source, [semantic_task("training_qa", "question", source)])
    payload["tasks"][0]["history_requirement"] = "none"
    payload["safety"] = {"level": "normal", "evidence": "", "categories": [], "reason": "synthetic"}
    return payload


def test_omitted_mode_auto_reads_workout_history(tmp_path, monkeypatch):
    with environment(tmp_path, monkeypatch, "auto-read") as env:
        from app.core.config import settings

        monkeypatch.setattr(settings, "chat_default_mode", "auto", raising=False)
        env.seed_workout("demo-user", "卧推", 0)
        env.reset_counts()
        before = env.snapshot()
        seen_tools = []

        def policy(index, messages):
            if index == 0:
                return {"role": "assistant", "content": "",
                        "tool_calls": [tool_call("r1", "read_workout_history", {"days": 1})]}
            if index == 1:
                observed = tool_results(messages)
                seen_tools.append(observed[-1])
                assert any("卧推" in json.dumps(row, ensure_ascii=False)
                           for row in observed[-1]["result"]["workouts"])
                return {"role": "assistant", "content": "",
                        "tool_calls": [tool_call("r2", "answer_training", {"query": SOURCE})]}
            return {"role": "assistant", "content": ANSWER}

        env.mock_llm(policy=policy, semantic=simulated_semantics, final=ANSWER)
        body = env.post("auto-read-1", message=SOURCE, mode=None)
        state = env.states[-1]

        print("SIM_READ=" + json.dumps({
            "answer": body.get("answer"),
            "declared_mode": body["metrics"].get("declared_mode"),
            "execution_mode": body["metrics"].get("execution_mode"),
            "react_status": body["metrics"].get("react_status"),
            "citations": len(body.get("citations") or []),
            "workouts_read": len(seen_tools[0]["result"]["workouts"]) if seen_tools else 0},
            ensure_ascii=False))
        assert body["metrics"]["declared_mode"] == "auto", body["metrics"]
        assert body["metrics"]["execution_mode"] == "react_readonly", body["metrics"]
        assert body["metrics"]["react_status"] == "complete", body["metrics"]
        assert body["answer"] == ANSWER, body
        assert seen_tools and seen_tools[0]["result"]["workouts"], seen_tools
        assert env.legacy.await_count == 0
        assert env.parse.await_count == 0
        assert env.snapshot() == before
        assert any("卧推" in json.dumps(message, ensure_ascii=False)
                   for message in state["messages"] if message.get("role") == "tool")
