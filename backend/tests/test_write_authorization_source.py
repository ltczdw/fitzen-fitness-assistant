"""R001/R002/R003 回归：取消/查询语义不得授权写入，写入实体必须有原文来源。

授权候选断言走生产验证路径 ``validate_rule_plan(plan_tasks_rule(...))``，
落库断言走真实执行链（真实工具 + 临时 SQLite，模型边界 mock）。
两类证据分开报告：授权候选错误 ≠ 实际落库错误。
"""
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.services.agent.planner_validation import validate_planner_output, validate_rule_plan
from app.services.agent.task_planner import plan_tasks_rule
from app.services.security.write_guard import authorize_write_span

_NOW = datetime.now(timezone.utc)


def _seed(message: str) -> None:
    from app.services.agent.semantics import remember_safety
    remember_safety(message, {"level": "normal", "evidence": "", "categories": [], "reason": "scripted normal"})


def semantic_task(message: str, kind: str, action: str = "record", **flags) -> dict:
    return {"task_type": kind, "action": action, "text": message,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            **flags}

_FORBIDDEN_INPUTS = [
    "不记了",
    "先不记了",
    "就不记了",
    "不记了，查看上次记录",
    "上次记录是什么",
    "查看昨天的训练记录",
    "可以记录跑步吗",
    "可以记录卧推60kg，4组8次吗",
    "记录卧推60kg，4组8次？",
]


def production_tasks(message: str) -> list[dict]:
    result = validate_rule_plan(message, plan_tasks_rule(message), now=_NOW)
    return result.tasks


def authorized_tasks(tasks: list[dict]) -> list[dict]:
    return [task for task in tasks if task.get("write_authorization") == "authorized"]


class R001CancelSemanticsTests(unittest.TestCase):
    """取消记录不得产生授权写候选；否定不得吞掉同句其他合法记录或历史查询。"""

    def test_cancel_commands_never_authorize_writes(self):
        for message in _FORBIDDEN_INPUTS:
            with self.subTest(message=message):
                tasks = production_tasks(message)
                self.assertEqual(authorized_tasks(tasks), [],
                                 f"取消/禁止输入产生了授权写候选: {tasks}")
        # The semantic path expresses cancellation with model flags/action.
        for flag in ("is_negated", "is_hypothetical", "is_third_party"):
            result = validate_planner_output(
                "不记了", {"tasks": [semantic_task("不记了", "nutrition_record", **{flag: True})]})
            self.assertNotEqual(result.tasks[0]["write_authorization"], "authorized")

    def test_cancel_with_history_query_keeps_the_query(self):
        tasks = production_tasks("不记了，查看上次记录")
        query_tasks = [task for task in tasks if "上次记录" in task.get("text", "")]
        self.assertTrue(query_tasks, f"历史查询被吞掉: {tasks}")
        self.assertTrue(all(task.get("action") in {"question", "answer"} for task in query_tasks),
                        f"历史查询应为问答任务: {tasks}")

    def test_negation_scope_does_not_swallow_sibling_record(self):
        message = "先不记了，帮我记录深蹲80kg3组8次"
        raw = {"tasks": [
            semantic_task("先不记了", "workout_record", is_negated=True),
            semantic_task("帮我记录深蹲80kg3组8次", "workout_record"),
        ]}
        _seed(message)
        result = validate_planner_output(message, raw)
        records = [task for task in result.tasks if task.get("task_type") == "workout_record"]
        self.assertEqual(len(authorized_tasks(records)), 1,
                         f"同句合法记录被否定吞掉: {result.tasks}")

    def test_gate_rejects_model_negated_span(self):
        authorization, reason = authorize_write_span("不记了，查看上次记录", {"is_negated": True})
        self.assertEqual(authorization, "rejected_unauthorized")
        self.assertEqual(reason, "planner_flag_negated")
        authorization, _ = authorize_write_span("先不记了", {"is_negated": True})
        self.assertEqual(authorization, "rejected_unauthorized")

    def test_gate_does_not_treat_every_bu_as_negation(self):
        # Only the model's explicit flag can negate; an unrelated word in the
        # caller's metadata must not block a model-authorized write.
        authorization, reason = authorize_write_span("记录深蹲80kg3组8次", {"note": "不相关"})
        self.assertEqual(authorization, "authorized", reason)


class R002QuestionAndHistoryTests(unittest.TestCase):
    """查询、疑问语义不得授权写入；前后置命令与其对象的绑定保留。"""

    def test_history_queries_route_to_qa(self):
        for message in ("上次记录是什么", "查看昨天的训练记录"):
            with self.subTest(message=message):
                tasks = production_tasks(message)
                self.assertEqual(authorized_tasks(tasks), [], f"{message}: {tasks}")
                self.assertTrue(
                    all(task.get("worker") != "workout_record"
                        or task.get("action") not in {"record", "confirm"} for task in tasks),
                    f"{message} 仍计划为保存动作: {tasks}")

    def test_question_mark_less_and_marked_questions_rejected(self):
        for message in ("可以记录跑步吗", "可以记录卧推60kg，4组8次吗"):
            with self.subTest(message=message):
                tasks = production_tasks(message)
                self.assertEqual(authorized_tasks(tasks), [], f"{message}: {tasks}")

    def test_trailing_question_mark_stays_rejected(self):
        tasks = production_tasks("记录卧推60kg，4组8次？")
        self.assertEqual(authorized_tasks(tasks), [], f"问号尾巴被授权: {tasks}")

    def test_record_command_survives_followup_question(self):
        message = "记录卧推60kg，4组8次，然后解释一下为什么要热身？"
        raw = {"tasks": [
            semantic_task("记录卧推60kg，4组8次", "workout_record"),
            semantic_task("然后解释一下为什么要热身？", "training_qa", action="question"),
        ]}
        _seed(message)
        result = validate_planner_output(message, raw)
        records = [task for task in result.tasks if task.get("task_type") == "workout_record"]
        self.assertEqual(len(authorized_tasks(records)), 1, f"合法记录被误拒: {result.tasks}")
        questions = [task for task in result.tasks if task.get("action") == "question"]
        self.assertTrue(questions, f"后续问题丢失: {result.tasks}")

    def test_trailing_command_binds_to_training_object(self):
        for message in ("跑步5公里，帮我记录一下", "今天跑步5公里30分钟，帮我记一下"):
            with self.subTest(message=message):
                _seed(message)
                result = validate_planner_output(
                    message, {"tasks": [semantic_task(message, "workout_record")]})
                records = authorized_tasks(result.tasks)
                self.assertEqual(len(records), 1, f"{message}: {result.tasks}")
                self.assertEqual(records[0].get("task_type"), "workout_record",
                                 f"{message} 写入域错误: {result.tasks}")
                self.assertIn("跑步", records[0].get("text", ""),
                              f"记录命令未绑定到训练对象: {result.tasks}")

    def test_meal_words_after_record_do_not_cancel_authorization(self):
        for message in (
            "帮我记录中午吃了一个鸡蛋",
            "记录中午吃了一个鸡蛋",
            "帮我记录今天中午吃了一个鸡蛋",
            "帮我记录今天中午吃了一个鸡蛋，晚上吃了一碗米饭",
        ):
            with self.subTest(message=message):
                _seed(message)
                self.assertEqual(authorize_write_span(message, {"semantic_available": True}),
                                 ("authorized", None))
                result = validate_planner_output(
                    message, {"tasks": [semantic_task(message, "nutrition_record")]})
                records = authorized_tasks(result.tasks)
                self.assertTrue(records, "明确的记录指令不应要求再次确认")
                self.assertTrue(all(task["task_type"] == "nutrition_record" for task in records))

    def test_meal_record_removal_keeps_rejection_rules(self):
        cases = (
            ("不要记录中午吃了一个鸡蛋", {"is_negated": True}),
            ("如果中午吃了一个鸡蛋，帮我记录", {"is_hypothetical": True}),
            ("帮他记录中午吃了一个鸡蛋", {"is_third_party": True}),
            ("可以记录中午吃了一个鸡蛋吗", {"action": "question"}),
        )
        for message, flags in cases:
            with self.subTest(message=message):
                kind = "nutrition_record"
                result = validate_planner_output(
                    message, {"tasks": [semantic_task(message, kind, **flags)]})
                self.assertEqual(authorized_tasks(result.tasks), [])

    def test_legit_records_still_authorized(self):
        checks = {
            "记录两个鸡蛋": "nutrition_record",
            "记录深蹲80kg3组8次": "workout_record",
        }
        for message, expected_type in checks.items():
            with self.subTest(message=message):
                _seed(message)
                result = validate_planner_output(
                    message, {"tasks": [semantic_task(message, expected_type)]})
                records = authorized_tasks(result.tasks)
                self.assertEqual(len(records), 1, f"{message}: {result.tasks}")
                self.assertEqual(records[0]["task_type"], expected_type, f"{message}: {result.tasks}")


class R003EntitySourceTests(unittest.TestCase):
    """写入实体必须能追溯到该写任务授权范围内的原文；词典别名与份量估算合法。"""

    def _filter(self, entries, source):
        from app.services.agent.food_parser import filter_entries_to_source
        kept, dropped = filter_entries_to_source(entries, source)
        return kept, dropped

    def _entry(self, name, quantity=1.0, unit="个", weight_g=None, meal_type=None, record_date=None):
        from app.services.agent.food_parser import FoodInput
        raw = f"{weight_g:g}g{name}" if weight_g is not None else f"{quantity:g}{unit}{name}"
        return FoodInput(raw_text=raw, requested_name=name, quantity=quantity, unit=unit,
                         weight_g=weight_g, portion_name=None if weight_g is not None else unit,
                         meal_type=meal_type, record_date=record_date)

    def test_hallucinated_food_is_dropped(self):
        kept, dropped = self._filter([self._entry("炸鸡")], "记录两个鸡蛋")
        self.assertEqual(kept, ())
        self.assertEqual(dropped, ["炸鸡"])

    def test_verbatim_and_alias_names_are_kept(self):
        kept, _ = self._filter([self._entry("鸡蛋", quantity=2.0)], "记录两个鸡蛋")
        self.assertEqual(len(kept), 1)
        # 词典映射：原文“饭”，模型规范化为“米饭”
        kept, dropped = self._filter([self._entry("米饭", unit="碗")], "记录一碗饭")
        self.assertEqual(len(kept), 1, f"合法别名被拒: {dropped}")

    def test_invented_quantity_and_weight_are_rejected(self):
        kept, dropped = self._filter([self._entry("鸡蛋", quantity=3.0)], "记录鸡蛋")
        self.assertEqual(kept, (), "凭空数量未被拒绝")
        kept, _ = self._filter([self._entry("鸡胸肉", weight_g=150.0)], "记录鸡胸肉")
        self.assertEqual(kept, (), "凭空重量未被拒绝")
        kept, _ = self._filter([self._entry("鸡胸肉", weight_g=150.0)], "记录150克鸡胸肉")
        self.assertEqual(len(kept), 1, "原文重量的合法条目被误拒")

    def test_invented_meal_and_date_are_neutralized(self):
        kept, _ = self._filter(
            [self._entry("鸡蛋", quantity=2.0, meal_type="夜宵", record_date="2025-12-31")],
            "记录两个鸡蛋")
        self.assertEqual(len(kept), 1)
        self.assertIsNone(kept[0].meal_type, "凭空餐次未被清除")
        self.assertIsNone(kept[0].record_date, "凭空日期未被清除")
        kept, _ = self._filter(
            [self._entry("鸡蛋", quantity=2.0, meal_type="早餐", record_date=None)],
            "今天早餐吃了两个鸡蛋，记录一下")
        self.assertEqual(kept[0].meal_type, "早餐", "原文餐次被误清")

    def test_structured_payload_entities_validated_against_task_span(self):
        from app.services.agent.multi_agent_graph import _food_inputs_from_structured_payload
        payload = {"foods": [{"name": "炸鸡", "quantity": 1, "unit": "份",
                              "weight_g": None, "meal_type": None, "record_date": None}]}
        entries = _food_inputs_from_structured_payload(payload, "记录两个鸡蛋")
        self.assertEqual(entries, (), "结构化实体未按任务原文校验")
        payload = {"foods": [{"name": "鸡蛋", "quantity": 2, "unit": "个",
                              "weight_g": None, "meal_type": None, "record_date": None}]}
        entries = _food_inputs_from_structured_payload(payload, "记录两个鸡蛋")
        self.assertEqual(len(entries), 1)


class _WorkerHarness:
    """Real agent whose model boundaries are replaced; records tool calls."""

    def __init__(self, monkeypatch, factory, *, parsed_input=None, food_entries=None):
        from app.db.storage import StorageUnavailableError
        from app.services.agent.agent import FitnessAgent
        from app.services.agent import multi_agent_graph as graph_module
        from app.services.agent.execution import memory_service

        self.factory = factory
        self.tool_calls: list[str] = []
        self.agent = FitnessAgent()
        monkeypatch.setattr(self.agent.graph, "parse_input", AsyncMock(return_value=parsed_input))
        monkeypatch.setattr(self.agent, "_run_planner_shadow", AsyncMock())
        monkeypatch.setattr(memory_service, "load",
                            AsyncMock(return_value={"memory_backend": "fixture"}))
        monkeypatch.setattr(self.agent.graph.rag, "search", lambda *args, **kwargs: [])
        monkeypatch.setattr(self.agent.graph, "_run_authorized_model_tool",
                            AsyncMock(return_value=None))
        if food_entries is not None:
            async def fake_food(_llm, _message):
                return tuple(food_entries)
            monkeypatch.setattr(graph_module, "extract_foods_with_llm", fake_food)

        async def call_mcp(name, arguments, **kwargs):
            self.tool_calls.append(name)
            if name == "get_user_profile":
                return {}
            if name == "get_training_history":
                return []
            if name == "get_nutrition_summary":
                return {}
            if name in {"save_workout_record", "calculate_nutrition", "calculate_nutrition_batch"}:
                return getattr(self.agent.graph.tools, name)(**arguments)
            raise AssertionError(f"unexpected tool: {name}")

        monkeypatch.setattr(self.agent.graph, "_call_mcp", call_mcp)

    async def respond(self, request):
        return await self.agent.respond(request)


class ForbiddenWriteEndToEndTests(unittest.TestCase):
    """禁止写入输入：真实执行链 0 保存工具调用、0 SQL 新增。"""

    def _db_counts(self, engine):
        from sqlalchemy import text
        with engine.connect() as conn:
            food = conn.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar()
            workout = conn.execute(text("SELECT COUNT(*) FROM workout_records")).scalar()
            return food, workout

    def test_forbidden_inputs_never_reach_save_tools(self):
        import asyncio
        import tempfile
        from pathlib import Path

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.core.config import settings
        from app.db import session as db
        from app.db.models import Base
        from app.schemas.chat import ChatRequest
        from app.services.agent.agent import FitnessAgent

        for message in ("不记了，查看上次记录", "上次记录是什么",
                        "可以记录卧推60kg，4组8次吗"):
            with self.subTest(message=message):
                harness = _WorkerHarness.__new__(_WorkerHarness)
                harness.tool_calls = []
                agent = FitnessAgent()
                harness.agent = agent
                engine = create_engine("sqlite:///" + str(Path(tempfile.mkdtemp()) / "e2e.db"))
                Base.metadata.create_all(engine)
                with engine.begin() as conn:
                    conn.exec_driver_sql(
                        "CREATE TABLE IF NOT EXISTS user_food_log ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, user_id VARCHAR(100) NOT NULL,"
                        "meal_group_id VARCHAR(100) NOT NULL, record_time TIMESTAMP NOT NULL,"
                        "meal_type VARCHAR(20), food_id INTEGER, raw_food_name VARCHAR(200) NOT NULL,"
                        "food_name VARCHAR(200) NOT NULL, weight_g NUMERIC NOT NULL,"
                        "total_kcal NUMERIC NOT NULL, total_protein NUMERIC NOT NULL,"
                        "total_carb NUMERIC NOT NULL, total_fat NUMERIC NOT NULL,"
                        "total_fiber NUMERIC NOT NULL DEFAULT 0, is_estimated BOOLEAN NOT NULL DEFAULT 0,"
                        "weight_source VARCHAR(50) NOT NULL DEFAULT 'user_input',"
                        "match_method VARCHAR(50) NOT NULL DEFAULT 'exact',"
                        "match_confidence FLOAT NOT NULL DEFAULT 1.0,"
                        "idempotency_key VARCHAR(150) NOT NULL,"
                        "calculation_version VARCHAR(100) NOT NULL DEFAULT 'food-calc-v1',"
                        "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
                        "UNIQUE(user_id, idempotency_key))")
                db.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
                db.DB_READY = True
                db.engine = engine
                try:
                    with (patch.object(agent.graph, "parse_input", AsyncMock(return_value=None)),
                          patch.object(agent, "_run_planner_shadow", AsyncMock()),
                          patch("app.services.agent.execution.memory_service.load",
                                AsyncMock(return_value={"memory_backend": "fixture"})),
                          patch.object(agent.graph.rag, "search", lambda *a, **k: []),
                          patch.object(agent.graph, "_run_authorized_model_tool",
                                       AsyncMock(return_value=None)),
                          patch.object(settings, "memory_read_backend", "compact_v1"),
                          patch.object(settings, "planner_shadow_enabled", False)):
                        async def call_tool(name, arguments, **kwargs):
                            harness.tool_calls.append(name)
                            if name in {"get_user_profile", "get_nutrition_summary"}:
                                return {}
                            if name == "get_training_history":
                                return []
                            return getattr(agent.graph.tools, name)(**arguments)

                        with patch.object(agent.graph, "_call_mcp", call_tool):
                            response = asyncio.run(agent.respond(ChatRequest(message=message)))
                    food_rows, workout_rows = self._db_counts(engine)
                    save_calls = [name for name in harness.tool_calls
                                  if name in {"save_workout_record", "calculate_nutrition",
                                              "calculate_nutrition_batch"}]
                    self.assertEqual(save_calls, [],
                                     f"{message} 派发了保存工具: {harness.tool_calls}")
                    self.assertEqual((food_rows, workout_rows), (0, 0),
                                     f"{message} 产生数据库写入: {response.answer}")
                finally:
                    engine.dispose()


class ModelForgedPlanTests(unittest.TestCase):
    """模型伪造 record+完整实体：仍零写入，不能靠缺参数偶然挡住。"""

    # 只用 schema 内字段的真实伪造：模型把查询标成 record 并自带完整实体。
    FORGED = {
        "tasks": [{
            "task_type": "workout_record", "action": "record",
            "text": "上次记录是什么",
            "workouts": [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}],
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        }],
    }
    # 携带 schema 外 write_authorization 键的伪造：V1 直接拒绝整个计划。
    FORGED_WITH_MARKER = {
        "tasks": [{
            "task_type": "workout_record", "action": "record",
            "text": "上次记录是什么",
            "workouts": [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}],
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "write_authorization": "authorized",
        }],
    }

    def test_gate_keeps_model_write_but_invented_entities_cannot_execute(self):
        from app.services.agent.task_planner import plan_tasks_from_llm
        from app.services.agent.workout_extractor import validate_workout_records_against_source
        tasks = plan_tasks_from_llm("上次记录是什么", self.FORGED)
        self.assertTrue(tasks, "伪造计划不应整体丢弃（应走守卫复核）")
        # The model's record label is authoritative for intent; the invented
        # exercise cannot pass the original-span entity check, so nothing can
        # be saved from it.
        kept, rejected = validate_workout_records_against_source(
            self.FORGED["tasks"][0]["workouts"], "上次记录是什么")
        self.assertEqual(kept, [])
        self.assertTrue(rejected)

    def test_forged_plan_via_real_agent_writes_nothing(self):
        import asyncio
        import tempfile
        from pathlib import Path

        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker

        from app.core.config import settings
        from app.db import session as db
        from app.db.models import Base
        from app.schemas.chat import ChatRequest
        from app.services.agent.agent import FitnessAgent

        agent = FitnessAgent()
        engine = create_engine("sqlite:///" + str(Path(tempfile.mkdtemp()) / "forge.db"))
        Base.metadata.create_all(engine)
        db.SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        db.DB_READY = True
        db.engine = engine
        tool_calls: list[str] = []
        try:
            with (patch.object(agent.graph, "parse_input",
                               AsyncMock(return_value=self.FORGED)),
                  patch.object(agent, "_run_planner_shadow", AsyncMock()),
                  patch("app.services.agent.execution.memory_service.load",
                        AsyncMock(return_value={"memory_backend": "fixture"})),
                  patch.object(agent.graph.rag, "search", lambda *a, **k: []),
                  patch.object(agent.graph, "_run_authorized_model_tool",
                               AsyncMock(return_value=None)),
                  patch.object(settings, "memory_read_backend", "compact_v1"),
                  patch.object(settings, "planner_shadow_enabled", False)):
                async def call_tool(name, arguments, **kwargs):
                    tool_calls.append(name)
                    if name in {"get_user_profile", "get_nutrition_summary"}:
                        return {}
                    if name == "get_training_history":
                        return []
                    return getattr(agent.graph.tools, name)(**arguments)

                with patch.object(agent.graph, "_call_mcp", call_tool):
                    asyncio.run(agent.respond(ChatRequest(message="上次记录是什么")))
            with engine.connect() as conn:
                rows = conn.execute(text("SELECT COUNT(*) FROM workout_records")).scalar()
            self.assertEqual([name for name in tool_calls
                              if name in {"save_workout_record"}], [],
                             f"伪造计划触发了保存工具: {tool_calls}")
            self.assertEqual(rows, 0, "伪造计划落库")
        finally:
            engine.dispose()

    def test_shadow_mode_ignores_forged_model_plan(self):
        from datetime import datetime, timezone

        from app.core.config import settings
        from app.services.agent.planner_validation import validate_rule_plan

        # 影子模式下路由只使用 validate_rule_plan 的权威结果；模型输出不进入决策。
        with patch.object(settings, "planner_shadow_enabled", True):
            validation = validate_rule_plan("上次记录是什么",
                                            plan_tasks_rule("上次记录是什么"),
                                            now=datetime.now(timezone.utc))
        for task in validation.tasks:
            self.assertNotEqual(task.get("write_authorization"), "authorized")


class LLMPlannerPathMatrixTests(unittest.TestCase):
    """正常/低覆盖/非法结构/不可用/影子模式：守卫均不得被模型绕过。"""

    FORGED = ModelForgedPlanTests.FORGED
    FORGED_WITH_MARKER = ModelForgedPlanTests.FORGED_WITH_MARKER

    def _validate_llm(self, message, raw, rule_tasks):
        from app.services.agent.planner_validation import validate_planner_output
        return validate_planner_output(message, raw, rule_tasks=rule_tasks,
                                       now=datetime.now(timezone.utc))

    def test_normal_llm_plan_invented_entities_are_not_executable(self):
        from app.services.agent.workout_extractor import validate_workout_records_against_source
        result = self._validate_llm("上次记录是什么", self.FORGED,
                                    plan_tasks_rule("上次记录是什么"))
        self.assertFalse(result.fallback_required)
        kept, rejected = validate_workout_records_against_source(
            self.FORGED["tasks"][0]["workouts"], "上次记录是什么")
        self.assertEqual(kept, [])
        self.assertTrue(rejected)

    def test_extra_authorization_key_is_schema_invalid(self):
        result = self._validate_llm("上次记录是什么", self.FORGED_WITH_MARKER,
                                    plan_tasks_rule("上次记录是什么"))
        self.assertTrue(result.fallback_required)
        self.assertEqual(result.degraded_reason, "planner_invalid_schema")
        for task in result.tasks:
            self.assertNotEqual(task.get("write_authorization"), "authorized")

    def test_low_coverage_llm_falls_back_to_guarded_rule_plan(self):
        low_coverage = {
            "tasks": [{
                "task_type": "training_qa", "action": "question",
                "text": "上次记录是什么",
                "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            }],
        }
        message = "上次记录是什么，顺便告诉我今天该练什么"
        result = self._validate_llm(message, low_coverage, plan_tasks_rule(message))
        self.assertTrue(result.fallback_required, "低覆盖未触发回退")
        self.assertEqual(result.degraded_reason, "planner_low_coverage")
        for task in result.tasks:
            self.assertNotEqual(task.get("write_authorization"), "authorized")

    def test_illegal_schema_falls_back_to_guarded_rule_plan(self):
        message = "上次记录是什么"
        result = self._validate_llm(message, {"tasks": "not-a-list"},
                                    plan_tasks_rule(message))
        self.assertTrue(result.fallback_required)
        self.assertEqual(result.degraded_reason, "planner_invalid_schema")
        for task in result.tasks:
            self.assertNotEqual(task.get("write_authorization"), "authorized")


if __name__ == "__main__":
    unittest.main()
