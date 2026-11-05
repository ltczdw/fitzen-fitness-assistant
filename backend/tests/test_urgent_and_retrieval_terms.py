"""E003/G003 回归：紧急表达按词族识别与否定语境；检索查询词项单一权威规则。"""
import unittest


class E003UrgentDetectionTests(unittest.TestCase):
    """Model-owned safety semantics; application only binds source/enums."""

    @staticmethod
    def _task(source):
        return {"task_type": "safety_triage", "action": "question", "text": source,
                "is_negated": False, "is_hypothetical": False, "is_third_party": False}

    def _assess(self, source, level, categories=()):
        from app.services.agent import semantics
        value = semantics.validate_understanding(source, {
            "tasks": [self._task(source)],
            "safety": {"level": level, "evidence": source,
                       "categories": list(categories), "reason": "scripted"},
        })
        token = semantics.request_understanding.set({})
        self.addCleanup(semantics.request_understanding.reset, token)
        semantics.remember_understanding(source, value)
        return semantics.safety_for(source)

    def test_urgent_expression_families_are_model_judged(self):
        urgent_cases = [
            "运动时晕厥了", "训练中突然晕倒了", "现在胸部剧痛", "胸口一阵剧痛",
            "突然出现单侧无力", "他口齿不清，面部下垂", "跑步时呼吸困难",
            "伤口大量出血", "运动后意识不清",
        ]
        for message in urgent_cases:
            with self.subTest(message=message):
                self.assertEqual(self._assess(message, "urgent", ["current_symptom"])["level"],
                                 "urgent")

    def test_model_context_can_be_normal_without_local_word_override(self):
        normal_cases = [
            "手指不麻木了", "没有胸痛", "手不再麻木了", "动作变形怎么纠正",
            "如何缓解腰部麻木", "避免动作变形", "膝盖没有明显肿胀",
        ]
        for message in normal_cases:
            with self.subTest(message=message):
                self.assertEqual(self._assess(message, "normal", ["negated", "consultation"])["level"],
                                 "normal")

    def test_current_urgent_is_retained_over_old_resolution(self):
        source = "以前手指不麻木，现在开始麻木了"
        self.assertEqual(self._assess(source, "urgent", ["current_symptom"])["level"], "urgent")

    def test_history_hypothetical_and_third_party_use_model_level(self):
        for message in ("去年训练时晕倒过一次", "如果有人突然晕倒该怎么办", "我朋友刚才晕倒了"):
            with self.subTest(message=message):
                self.assertEqual(self._assess(message, "medical", ["medical_history"])["level"],
                                 "medical")

    def test_urgent_gate_blocks_writes_with_source_bound_model_judgment(self):
        from datetime import datetime, timezone
        from app.services.security.write_guard import guard_planner_tasks
        tasks = [{
            "id": "task-1", "task_type": "workout_record", "action": "record",
            "text": "记录卧推60kg4组8次", "source_start": 0, "source_end": 13,
        }]
        source = "记录卧推60kg4组8次，我现在胸部剧痛"
        self._assess(source, "urgent", ["current_symptom"])
        guard_planner_tasks(source, tasks, now=datetime.now(timezone.utc))
        self.assertEqual(tasks[0]["write_authorization"], "rejected_unauthorized")
        self.assertEqual(tasks[0]["write_rejection_reason"], "urgent_safety_gate")

    def test_intent_reuses_request_local_urgent_judgment(self):
        from app.services.agent.intent import classify_intent_rule
        source = "运动时晕厥了"
        self._assess(source, "urgent", ["current_symptom"])
        intent = classify_intent_rule(source)
        self.assertEqual(intent.name, "symptom_triage")
        self.assertEqual(intent.safety_level, "urgent")


class G003QueryTermsTests(unittest.TestCase):
    """查询词项单一权威规则：重叠二元词，过滤与索引口径一致。"""

    def test_overlapping_bigrams_keep_the_hit_term(self):
        from app.services.rag.bm25 import query_terms, tokenize
        terms = query_terms("有疼痛")
        self.assertIn("疼痛", terms, f"非重叠切词问题仍存在: {terms}")
        self.assertIn("有疼", terms)
        # 与 BM25 索引分词的二元词集合一致
        index_bigrams = {token for token in tokenize("有疼痛") if len(token) == 2}
        self.assertTrue(index_bigrams.issubset(set(terms)))

    def test_filter_keeps_lexical_hit_for_prefix_changed_query(self):
        # 复现原缺陷形态：BM25 命中的候选被过滤分支按字符偏移丢弃。
        from app.services.rag.bm25 import query_terms
        body = "膝盖疼痛时建议休息和冰敷"
        terms = query_terms("有疼痛")
        self.assertTrue(any(term in body for term in terms),
                        "BM25 命中候选仍会被过滤分支丢弃")

    def test_single_char_queries_do_not_open_single_char_matching(self):
        from app.services.rag.bm25 import query_terms
        self.assertEqual(query_terms("疼痛"), ["疼痛"])  # 无单字泄漏
        self.assertEqual(query_terms("痛"), ["痛"])  # 单字查询仅输出自身

    def test_quality_coverage_shares_the_rule(self):
        from app.services.rag.quality import _query_terms
        self.assertIn("疼痛", _query_terms("有疼痛"))


class G004GateMetricTests(unittest.TestCase):
    """默认门控行为必须被验收看见；指标函数与校准/测试分离的合同。"""

    def _result(self, confidence, should_reject, task="training_qa", split="test"):
        from scripts.eval_rag import SampleResult
        return SampleResult(
            id="x", category="factual", difficulty="easy", split=split,
            should_reject=should_reject, recall=1.0, mrr=1.0,
            expected_coverage_ok=True, citation_precision=0.5, traceable=True,
            gate_task=task, gate_mode="degraded_rules",
            gate_answerable_default=confidence >= 0.48,
            gate_answerable_calibrated=confidence >= 0.36,
            gate_confidence=confidence, number_support=None, term_coverage=None)

    def test_gate_metrics_reject_all_refusal_gaming(self):
        from scripts.eval_rag import _gate_metrics
        rows = [self._result(0.9, False), self._result(0.2, True)]
        metrics = _gate_metrics(rows, "gate_answerable_default")
        self.assertEqual(metrics["answerable_acceptance"], 1.0)
        self.assertEqual(metrics["no_evidence_refusal"], 1.0)
        # 全部拒答时接受率为 0，不能通过拒答率 1.0 蒙混。
        rows = [self._result(0.2, True), self._result(0.2, True)]
        metrics = _gate_metrics(rows, "gate_answerable_default")
        self.assertEqual(metrics["answerable_acceptance"], 0.0)
        self.assertEqual(metrics["no_evidence_refusal"], 1.0)

    def test_calibration_scan_only_uses_calibration_rows(self):
        from scripts.eval_rag import _calibrate_threshold, SampleResult

        def make(confidence, should_reject, split):
            return SampleResult(
                id=f"id-{confidence}-{should_reject}-{split}", category="factual",
                difficulty="easy", split=split, should_reject=should_reject,
                recall=1.0, mrr=1.0, expected_coverage_ok=True, citation_precision=0.5,
                traceable=True, gate_task="training_qa", gate_mode="degraded_rules",
                gate_answerable_default=False, gate_answerable_calibrated=False,
                gate_confidence=confidence, number_support=None, term_coverage=None)

        # test split 的极端分布若参与搜索会改变最优阈值；它必须被忽略。
        rows = ([make(0.9, False, "calibration"), make(0.1, True, "calibration")] * 5
                + [make(0.5, False, "test"), make(0.5, True, "test")] * 5)
        threshold, agreement = _calibrate_threshold(rows)
        self.assertEqual(threshold, 0.20)
        self.assertEqual(round(agreement, 3), 1.0)

    def test_sample_split_is_stable_and_overridable(self):
        from scripts.eval_rag import _sample_split
        sample = {"id": "rag-001"}
        self.assertEqual(_sample_split(sample), _sample_split(dict(sample)))
        self.assertIn(_sample_split(sample), {"calibration", "test"})
        self.assertEqual(_sample_split({"id": "x", "split": "test"}), "test")

    def test_safety_bump_stays_within_reachable_score_range(self):
        # G004 不可达组合修复：纯词法降级模式下安全任务阈值不再上浮到
        # 可达上限（0.55）之上；模型重排模式照常上浮。
        from app.core.config import settings
        from app.services.rag.quality import _DEGRADED_ACHIEVABLE_MAX, assess_rag_evidence

        class _Hit:
            def __init__(self, content, metadata):
                self.title = "指南"
                self.content = content
                self.metadata = metadata

        lexical_hits = [_Hit("胸痛时应立即停止运动并就医",
                             {"lexical_score": 5.0, "vector_score": -1}),
                        _Hit("康复期应避免剧烈疼痛动作",
                             {"lexical_score": 3.0, "vector_score": -1})]
        assessment = assess_rag_evidence("训练时胸部剧痛怎么办", lexical_hits,
                                         task_type="safety_triage")
        self.assertEqual(assessment.mode, "degraded_rules")
        self.assertEqual(assessment.threshold, float(settings.rag_answer_confidence_threshold),
                         "纯词法模式下安全阈值仍被上浮到不可达区间")
        self.assertLessEqual(assessment.threshold, _DEGRADED_ACHIEVABLE_MAX)

        model_hits = [_Hit("胸痛时应立即停止运动并就医",
                           {"rerank_status": "model", "rerank_score": 0.9}),
                      _Hit("康复期应避免剧烈疼痛动作",
                           {"rerank_status": "model", "rerank_score": 0.4})]
        model_assessment = assess_rag_evidence("训练时胸部剧痛怎么办", model_hits,
                                               task_type="safety_triage")
        self.assertEqual(model_assessment.mode, "model_rerank")
        self.assertEqual(model_assessment.threshold,
                         min(0.95, float(settings.rag_answer_confidence_threshold) + 0.10))

    def test_no_hits_is_always_refused(self):
        from app.services.rag.quality import assess_rag_evidence
        for task in ("training_qa", "safety_triage"):
            assessment = assess_rag_evidence("任何问题", [], task_type=task)
            self.assertFalse(assessment.answerable, task)
            self.assertEqual(assessment.mode, "none")


class R006NetworkDenyTests(unittest.TestCase):
    """禁网审计覆盖应用 DNS 解析，同时不破坏 asyncio 内部唤醒机制。"""

    def test_dns_resolution_is_blocked_and_counted(self):
        import asyncio
        import socket

        from evals.v2.safety import deny_network

        async def attempt():
            loop = asyncio.get_running_loop()
            try:
                await loop.getaddrinfo("example.com", 443)
                return "resolved"
            except RuntimeError as exc:
                return str(exc)

        with deny_network() as audit:
            outcome = asyncio.run(attempt())
        self.assertEqual(outcome, "evaluation_dns_prohibited")
        self.assertGreaterEqual(audit.get("blocked_dns", 0), 1)
        self.assertEqual(audit["blocked_connections"], 0)

    def test_plain_socket_connect_still_blocked(self):
        import socket

        from evals.v2.safety import deny_network

        with deny_network() as audit:
            sock = socket.socket()
            try:
                sock.connect(("127.0.0.1", 80))
            except RuntimeError as exc:
                message = str(exc)
            finally:
                sock.close()
        self.assertEqual(message, "evaluation_network_prohibited")
        self.assertGreaterEqual(audit["blocked_connections"], 1)


class E007DependencyContractTests(unittest.TestCase):
    """显式跨业务依赖边：原文明确引用结果时必须保留，缺失会被评测检出。"""

    def setUp(self):
        from app.services.agent.semantics import request_understanding
        self._token = request_understanding.set({})

    def tearDown(self):
        from app.services.agent.semantics import request_understanding
        request_understanding.reset(self._token)

    @staticmethod
    def _write(task_type, text, depends_on=()):
        return {"id": None, "task_type": task_type, "action": "record", "text": text,
                "is_negated": False, "is_hypothetical": False, "is_third_party": False,
                "side_effect": True, "depends_on": list(depends_on)}

    @staticmethod
    def _read(task_type, text, depends_on=()):
        return {"id": None, "task_type": task_type, "action": "question", "text": text,
                "is_negated": False, "is_hypothetical": False, "is_third_party": False,
                "side_effect": False, "depends_on": list(depends_on)}

    def test_explicit_cross_business_reference_is_preserved_through_v5(self):
        from app.services.agent.planner_validation import validate_rule_plan
        from app.services.agent.task_planner import plan_tasks_rule
        from app.services.agent.semantics import remember_understanding

        message = "记录早餐两个鸡蛋，然后根据今天的蛋白质摄入制定明天的训练计划"
        remember_understanding(message, {"validated_tasks": [
            self._write("nutrition_record", "记录早餐两个鸡蛋"),
            self._read("training_plan", "根据今天的蛋白质摄入制定明天的训练计划", depends_on=["task-1"]),
        ], "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}})
        result = validate_rule_plan(message, plan_tasks_rule(message))
        plan_tasks = [t for t in result.tasks if t["task_type"] == "training_plan"]
        records = [t for t in result.tasks if t["task_type"] == "nutrition_record"]
        self.assertEqual(len(plan_tasks), 1)
        self.assertEqual(len(records), 1)
        self.assertIn(records[0]["id"], plan_tasks[0].get("depends_on", []),
                      f"显式跨业务依赖边被 V5 清空: {result.tasks}")

    def test_same_business_advice_keeps_its_record_edge(self):
        from app.services.agent.planner_validation import validate_rule_plan
        from app.services.agent.task_planner import plan_tasks_rule
        from app.services.agent.semantics import remember_understanding

        message = "记录两个鸡蛋，然后安排晚餐"
        remember_understanding(message, {"validated_tasks": [
            self._write("nutrition_record", "记录两个鸡蛋"),
            self._read("nutrition_advice", "安排晚餐", depends_on=["task-1"]),
        ], "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}})
        result = validate_rule_plan(message, plan_tasks_rule(message))
        advice = next(t for t in result.tasks if t["task_type"] == "nutrition_advice")
        record = next(t for t in result.tasks if t["task_type"] == "nutrition_record")
        self.assertIn(record["id"], advice.get("depends_on", []))
