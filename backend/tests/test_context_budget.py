import unittest
from types import SimpleNamespace

from app.services.context.budget import ContextBlock, ContextBudgetExceeded, ContextBudgetManager
from app.services.context.token_counter import TokenCounter


class ContextBudgetTests(unittest.TestCase):
    def setUp(self):
        self.manager = ContextBudgetManager(window=1024, output=128, margin=0,
                                            counter=TokenCounter(tokenizer_path=""))

    def test_trims_summary_before_whole_recent_turn_and_rag(self):
        blocks = [
            ContextBlock("system", "system", [{"role": "system", "content": "规则" * 10}], 1000, required=True),
            ContextBlock("query", "query", [{"role": "user", "content": "问题" * 10}], 999, required=True),
            ContextBlock("rag", "rag", [{"role": "system", "content": "资料" * 80}], 60),
            ContextBlock("turn", "recent", [{"role": "user", "content": "旧问题" * 40},
                                               {"role": "assistant", "content": "旧回答" * 40}], 40),
            ContextBlock("summary", "summary", [{"role": "system", "content": "摘要" * 150}], 20),
        ]
        result = self.manager.build(blocks)
        self.assertIn("summary", result.report["dropped_blocks"])
        self.assertNotIn("rag", result.report["dropped_blocks"])
        roles_and_text = [(item["role"], item["content"]) for item in result]
        turn_parts = [item for item in roles_and_text if item[1].startswith("旧")]
        self.assertIn(len(turn_parts), {0, 2})

    def test_rag_parent_context_is_reduced_before_chunk_dropped(self):
        block = ContextBlock("rag", "rag", [{"role": "system", "content": "子块" * 80 + "父文档" * 150}], 60,
                             reduced_messages=[{"role": "system", "content": "子块" * 80}])
        result = self.manager.build([
            ContextBlock("system", "system", [{"role": "system", "content": "规则"}], 1000, required=True),
            ContextBlock("query", "query", [{"role": "user", "content": "问题"}], 999, required=True),
            block,
        ])
        self.assertIn("rag", result.report["reduced_blocks"])
        self.assertNotIn("rag", result.report["dropped_blocks"])

    def test_never_trims_required_safety_and_current_question(self):
        manager = ContextBudgetManager(window=300, output=64, margin=0,
                                       counter=TokenCounter(tokenizer_path=""))
        with self.assertRaises(ContextBudgetExceeded):
            manager.build([
                ContextBlock("system", "system", [{"role": "system", "content": "安全" * 80}], 1000, required=True),
                ContextBlock("query", "query", [{"role": "user", "content": "当前问题" * 40}], 999, required=True),
            ])

    def test_counter_uses_conservative_mode_without_model_tokenizer(self):
        counter = TokenCounter(tokenizer_path="")
        self.assertEqual(counter.mode, "conservative-utf8")
        self.assertEqual(counter.text("健身"), len("健身".encode("utf-8")))

    def test_assemble_follows_safety_rag_recent_summary_trim_order(self):
        manager = ContextBudgetManager(window=1200, output=128, margin=0,
                                       counter=TokenCounter(tokenizer_path=""))
        hits = [SimpleNamespace(title="权威动作指南", source="official", score=0.9,
                                metadata={"url": "https://example.invalid/guide"},
                                content="动作证据" * 70 + "\n\n父文档摘要：" + "扩展背景" * 120)]
        state = {
            "short_memory": [
                {"role": "user", "content": "我上一轮问的是深蹲姿势", "message_id": "u1"},
                {"role": "assistant", "content": "先控制动作幅度", "message_id": "a1"},
            ],
            "semantic_memory": [{"text": "很久以前偏好晚间训练" * 20, "category": "preference",
                                 "speaker": "user", "importance": 0.9}],
            "long_memory": {"injury": {"text": "医生要求膝盖疼痛时不能负重"}},
        }
        result = manager.assemble(system="安全规则", query="这个动作还能继续练吗", state=state, hits=hits)
        self.assertIn("semantic-0", result.report["dropped_blocks"])
        self.assertNotIn("long-injury", result.report["dropped_blocks"])
        self.assertLessEqual(result.report["used_tokens"], result.report["input_budget"])
        self.assertEqual(result[-1]["content"], "这个动作还能继续练吗")

    def test_assemble_keeps_a_complete_referenced_recent_turn(self):
        manager = ContextBudgetManager(window=900, output=128, margin=0,
                                       counter=TokenCounter(tokenizer_path=""))
        state = {"short_memory": [
            {"role": "user", "content": "我上一轮问的是卧推动作", "message_id": "u1"},
            {"role": "assistant", "content": "肩胛骨保持稳定", "message_id": "a1"},
        ], "semantic_memory": [{"text": "不重要的旧摘要" * 80, "category": "transient",
                                "speaker": "assistant", "importance": 0.1}]}
        result = manager.assemble(system="规则", query="继续解释这个", state=state, hits=[])
        contents = [item["content"] for item in result]
        self.assertIn("我上一轮问的是卧推动作", contents)
        self.assertIn("肩胛骨保持稳定", contents)


if __name__ == "__main__":
    unittest.main()
