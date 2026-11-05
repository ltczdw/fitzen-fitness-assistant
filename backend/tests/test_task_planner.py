"""Planner consumes model semantics; unavailable models never infer tasks."""
import pytest
from app.services.agent.semantics import request_understanding, validate_understanding, remember_understanding
from app.services.agent.task_planner import plan_tasks, plan_tasks_from_llm
from app.services.agent.task_scheduler import select_ready_task


@pytest.fixture(autouse=True)
def isolated_scope():
    token = request_understanding.set({})
    yield
    request_understanding.reset(token)


def task(kind, text, action="question", **extra):
    return {"task_type": kind, "action": action, "text": text, "is_negated": False,
            "is_hypothetical": False, "is_third_party": False, **extra}


@pytest.mark.parametrize("message,kind", [
    ("制定每周三次训练计划", "training_plan"),
    ("给我安排每周两次卧推训练计划", "training_plan"),
    ("设计四周深蹲训练方案", "training_plan"),
    ("如果明天训练，怎么安排热身", "training_qa"),
    ("请解释卧推热身方法", "training_qa"),
    ("安排训练前热身动作", "training_qa"),
    ("怎么练卧推", "training_qa"),
    ("卧推60公斤4组8次怎么练", "training_qa"),
    ("明天卧推怎么练", "training_qa"),
])
def test_model_question_type_is_preserved(message, kind):
    value = validate_understanding(message, {"tasks": [task(kind, message)]})
    remember_understanding(message, value)
    planned = plan_tasks(message)
    assert [t["task_type"] for t in planned] == [kind]
    assert not planned[0]["side_effect"]


@pytest.mark.parametrize("first,second,record,advice", [
    ("把我记录今天中午吃了一碗饭和两个鸡蛋", "今晚吃什么好", "nutrition_record", "nutrition_advice"),
    ("帮我把午餐记一下", "再帮我规划晚餐", "nutrition_record", "nutrition_advice"),
    ("把今天训练记一下", "再安排明天的训练", "workout_record", "training_plan"),
    ("记录卧推60公斤4组8次", "制定每周三次训练计划", "workout_record", "training_plan"),
])
def test_model_split_keeps_order_and_record_dependency(first, second, record, advice):
    message = first + "，" + second
    planned = plan_tasks(message, parsed_input={"tasks": [task(record, first, "record"), task(advice, second)]})
    assert [t["task_type"] for t in planned] == [record, advice]
    assert planned[0]["side_effect"]
    assert planned[1]["depends_on"] == [planned[0]["id"]]


def test_repeated_worker_tasks_remain_distinct():
    message = "记录早餐一碗粥，记录午餐一份牛肉"
    planned = plan_tasks(message, parsed_input={"tasks": [
        task("nutrition_record", "记录早餐一碗粥", "record"),
        task("nutrition_record", "记录午餐一份牛肉", "record")]})
    assert [t["id"] for t in planned] == ["task-1", "task-2"]
    assert all(t["side_effect"] for t in planned)


def test_single_model_workout_task_is_not_split_locally():
    message = "补记昨天训练：哑铃卧推20kg4组10次、深蹲40kg3组12次"
    planned = plan_tasks(message, parsed_input={"tasks": [task("workout_record", message, "record")]})
    assert len(planned) == 1 and planned[0]["worker"] == "workout_record"


def test_entity_payload_and_nonwrite_actions_survive_mapping():
    message = "午餐一碗米饭"
    parsed = {"tasks": [task("nutrition_record", message, "record", foods=[{"name": "米饭", "quantity": 1, "unit": "碗"}])]}
    assert plan_tasks(message, parsed_input=parsed)[0]["payload"]["foods"][0]["name"] == "米饭"
    parsed["tasks"][0]["action"] = "question"
    assert not plan_tasks(message, parsed_input=parsed)[0]["side_effect"]


def test_incomplete_or_forged_model_plan_has_no_rule_repair():
    message = "记录午餐一个鸡蛋，再安排晚餐"
    for fragment in ("记录午餐一个鸡蛋", "模型编造的食物"):
        raw = {"tasks": [task("nutrition_record", fragment, "record")]}
        assert plan_tasks_from_llm(message, raw) is None
        assert all(not t["side_effect"] for t in plan_tasks(message, parsed_input=raw))


@pytest.mark.parametrize("message", ["记录卧推60公斤4组8次", "记录两个鸡蛋然后规划晚餐", "制定每周三次训练计划"])
def test_missing_model_never_guesses_task_type(message):
    planned = plan_tasks(message)
    assert len(planned) == 1 and planned[0]["task_type"] == "training_qa"
    assert planned[0]["action"] == "question" and not planned[0]["side_effect"]


def test_scheduler_selects_dependency_ready_task():
    tasks = [{"id": "task-1", "status": "pending", "depends_on": ["task-2"]},
             {"id": "task-2", "status": "pending", "depends_on": []}]
    assert select_ready_task(tasks, [])["id"] == "task-2"
    assert select_ready_task(tasks, ["task-2"])["id"] == "task-1"
