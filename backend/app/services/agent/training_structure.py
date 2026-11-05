"""Training prescriptions without a duration or timetable contract."""
import json
from typing import Annotated,Literal
from pydantic import BaseModel,ConfigDict,Field,StrictInt,field_validator

Activity=Annotated[str,Field(min_length=1,max_length=500)]


class Value(BaseModel):
    model_config=ConfigDict(extra="forbid",str_strip_whitespace=True)


class Exercise(Value):
    name: str=Field(min_length=1,max_length=150,description="只写动作名称，不写时长或休息时间。")
    sets: StrictInt=Field(gt=0,le=100)
    reps: StrictInt|None=Field(default=None,gt=0,le=1000,
        description="每组次数；无法用次数表达的动作留空，不填写秒数或时间。")


class Main(Value):
    exercises: list[Exercise]=Field(min_length=1,max_length=12)


class Session(Value):
    label: str=Field(min_length=1,max_length=100,description="训练日/部位或序号，不含用时。")
    warmup: list[Activity]=Field(min_length=1,max_length=12,description="只列热身活动，不写分钟或秒数。")
    main: Main
    cooldown: list[Activity]=Field(min_length=1,max_length=12,description="只列收尾活动，不写分钟或秒数。")
    without_timing: Literal[True]=Field(description="确认所有自然语言字段均没有时间安排、时长或休息秒数。")

    @field_validator('without_timing',mode='before')
    @classmethod
    def literal_true(cls,value):
        if value is not True:raise ValueError('without_timing_must_be_true')
        return value


class UntimedTrainingPlan(Value):
    sessions: list[Session]=Field(min_length=1,max_length=21)


def parse_untimed_plan(raw,*,expected_sessions,expected_weekdays=None):
    def unique(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError("duplicate_plan_field")
            result[key]=value
        return result
    plan=(raw.model_copy(deep=True) if isinstance(raw,UntimedTrainingPlan)
          else UntimedTrainingPlan.model_validate(json.loads(raw,object_pairs_hook=unique)))
    if len(plan.sessions)!=expected_sessions:raise ValueError("session_count_mismatch")
    if len({s.label for s in plan.sessions})!=len(plan.sessions):raise ValueError("duplicate_session_label")
    if expected_weekdays and (len(expected_weekdays)!=len(plan.sessions) or any(
        day not in session.label for day,session in zip(expected_weekdays,plan.sessions))):
        raise ValueError("session_weekday_mismatch")
    return plan


def render_untimed_plan(plan):
    """Render only activities and set/repetition prescriptions, with no clock fields."""
    clean=lambda value:str(value).replace('|','／').replace('\n','；').replace('\r','')
    lines=["训练计划："]
    for session in plan.sessions:
        warmup=session.warmup if isinstance(session.warmup,list) else [session.warmup.activity]
        cooldown=session.cooldown if isinstance(session.cooldown,list) else [session.cooldown.activity]
        lines += ['',clean(session.label),'','热身：'+'、'.join(map(clean,warmup)),
                  '','| 动作 | 组数 | 每组次数 |','|---|---|---|']
        for exercise in session.main.exercises:
            lines.append(f'| {clean(exercise.name)} | {exercise.sets} | {exercise.reps if exercise.reps is not None else "按动作质量完成"} |')
        lines += ['','收尾：'+'、'.join(map(clean,cooldown))]
    return '\n'.join(lines)


def output_contract():
    return ("\n本次使用无时间训练结构，调用return_training_plan。按expected_sessions返回全部训练日，"
            "保留expected_weekdays的顺序。只提供动作、组数、次数、热身和收尾活动；不输出单次用时、阶段时间表、"
            "总分钟数、休息秒数或计时动作秒数。器械defaulted_fields中的equipment是应用默认健身房，"
            "不能声称用户说过。用户明确的器械限制优先。without_timing=true是对所有描述的自检，不得把时间藏在名称里。"
            "不追问训练时长。schema："+json.dumps(UntimedTrainingPlan.model_json_schema(),ensure_ascii=False))
