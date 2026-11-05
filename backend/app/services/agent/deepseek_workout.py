from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.core import clock
from app.services.llm.client import LLMClient


logger = logging.getLogger(__name__)


class DeepSeekWorkoutItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exercise: str | None = Field(default=None, max_length=80)
    weight_kg: float | None = Field(default=None, ge=0, le=1000)
    sets: int | None = Field(default=None, ge=0, le=100)
    reps: int | None = Field(default=None, ge=1, le=1000)
    inferred_fields: list[str] = Field(default_factory=list, max_length=8)
    record_date: str | None = Field(default=None, max_length=20)


class DeepSeekWorkoutResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[DeepSeekWorkoutItem] = Field(default_factory=list, max_length=20)


class DeepSeekWorkoutExtractor:
    """Extract workout entities with the shared DeepSeek gateway."""

    _system_prompt = """你是健身助手的训练记录解析器，只负责从用户原话提取训练记录，不回答问题，不执行保存。
必须把用户原话中明确提到的每一个训练动作都输出为 records 中的一项，不能只返回第一个动作，也不能合并不同动作。
字段含义：exercise 是动作名称，weight_kg 是公斤数，sets 是组数，reps 是每组次数。
例如“10kg*10,4组”表示 weight_kg=10、reps=10、sets=4。
完整三项乘式按“重量×每组次数×组数”读取，例如“10kg*10次*4”“10kg×10×4”“10 x 10 x 4”“10X10X4”“10＊10＊4”“10*10*4”均为10kg、10次、4组；应用重量单位默认为kg。乘号可使用 *、x、X、×、＊及合理空格；明确标了组/次时按标注理解（如“10kg×3组×12次”为10kg、12次、3组），只有两项裸数时按“组数×次数”。信息足够直接提取，不要求用户重复确认。
“自重”“徒手”是明确的重量值0kg，不是缺失值：例如“引体向上自重*10*4”“自重引体向上×10×4”“徒手深蹲×12×3”分别为 weight_kg=0、reps=10/10/12、sets=4/4/3；“自重/徒手”占重量位，后面的两项数字仍按“每组次数×组数”读取。不要因为用户没有写数字重量就返回 weight_kg=null，也不要把一个动作的自重标记借给同句其他动作。
exercise保留原文实际动作名称，不限于动作库或固定词表，不擅自补器械名称或把未知动作替换成已知动作。
允许理解中文数字、kg/公斤、组/套、次/下/个，以及“每组8”“4组8次”等常见说法。
只能使用用户原话明确给出或训练语法直接表达的信息，不能猜测缺失数字；无法确定的训练数值填 null。日期严格按由应用时钟注入的相对日期映射填写 record_date（ISO 日期）；原文有相对日期时按映射换算，没有日期表达时填今天的 ISO 日期。
user_message是本次要提取的动作原文，authorized_task是应用绑定的同一保存目标原文，仅用于保全该目标中适用于当前动作的日期条件。不要提取authorized_task中位于user_message之外的动作或数字，不要把其它动作的日期借给当前动作；按每个动作自己的日期分别填写record_date。
只输出合法 JSON，不要 Markdown、解释或额外字段，格式必须是：
{"records":[{"exercise":string|null,"weight_kg":number|null,"sets":integer|null,"reps":integer|null,"inferred_fields":string[],"record_date":string|null}]}
"""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    @staticmethod
    def _prompt_for_current_date(reference_date: date | None = None) -> str:
        today = reference_date if reference_date is not None else clock.today()
        return (
            DeepSeekWorkoutExtractor._system_prompt
            + f"\n应用时区：{settings.app_timezone}。今天={today.isoformat()}；"
            + f"昨天={(today - timedelta(days=1)).isoformat()}；"
            + f"前天={(today - timedelta(days=2)).isoformat()}；"
            + f"明天={(today + timedelta(days=1)).isoformat()}；"
            + f"后天={(today + timedelta(days=2)).isoformat()}。"
        )

    async def extract(self, message: str, *, reference_date: date | None = None,
                      task_source: str | None = None) -> list[dict[str, Any]] | None:
        if not settings.workout_parser_enabled or not message.strip() or not self.llm.configured():
            return None
        try:
            raw = await self.llm.complete(
                [
                    {"role": "system", "content": self._prompt_for_current_date(reference_date)},
                    {"role": "user", "content": json.dumps({
                        "user_message": message[:4000],
                        **({"authorized_task": task_source} if task_source is not None else {}),
                    }, ensure_ascii=False)},
                ],
                temperature=0.0,
            )
            if not raw:
                return None
            text = str(raw).strip()
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
            if fenced:
                text = fenced.group(1)
            else:
                object_match = re.search(r"\{.*\}", text, re.DOTALL)
                if object_match:
                    text = object_match.group(0)
            result = DeepSeekWorkoutResponse.model_validate_json(text)
            return [item.model_dump() for item in result.records]
        except Exception as exc:
            logger.warning("deepseek_workout_parse_failed: %s", type(exc).__name__)
            return None
