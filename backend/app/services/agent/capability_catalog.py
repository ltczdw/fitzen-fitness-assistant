"""Application-owned capability registry and truthful entry-point copy.

The chat answer may select a capability kind, but the copy comes from here. A
non-null ``entry_screen`` must be a real frontend destination; if there is no
real page/button for an operation, the entry stays ``None`` and the copy says so
instead of inventing navigation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityEntry:
    kind: str
    copy: str
    allows_chat_write: bool
    receipt_required: bool
    entry_screen: str | None = None
    entry_label: str | None = None


CAPABILITY_REGISTRY: dict[str, CapabilityEntry] = {
    "record_workout": CapabilityEntry(
        kind="record_workout",
        copy="我可以帮你记录训练；只有你明确要求并通过校验时才会保存，没有真实保存回执时不会声称已保存。",
        allows_chat_write=True, receipt_required=True,
        entry_screen="training", entry_label="训练记录"),
    "record_nutrition": CapabilityEntry(
        kind="record_nutrition",
        copy="我可以帮你记录饮食；只有你明确要求并通过校验时才会保存，没有真实保存回执时不会声称已保存。",
        allows_chat_write=True, receipt_required=True,
        entry_screen="nutrition", entry_label="饮食记录"),
    "plan_preview": CapabilityEntry(
        kind="plan_preview",
        copy="我可以生成训练计划预览；预览不是已保存计划，当前对话不能把预览直接保存为正式计划。",
        allows_chat_write=False, receipt_required=False),
    "plan_save": CapabilityEntry(
        kind="plan_save",
        copy=("当前对话没有保存训练计划的工具，我不能在聊天里直接保存计划或声称已经保存；"
              "当前版本没有可确认的计划保存页面或入口。"),
        allows_chat_write=False, receipt_required=True),
    "profile": CapabilityEntry(
        kind="profile",
        copy="我可以读取你已保存的档案字段；不会在未确认时修改档案。修改请在个人信息页操作。",
        allows_chat_write=False, receipt_required=False,
        entry_screen="profile", entry_label="个人信息"),
    "memory": CapabilityEntry(
        kind="memory",
        copy="只有你明确要求时，我才能保存或更正长期记忆；不会自动记忆。",
        allows_chat_write=False, receipt_required=True),
    "delete_records": CapabilityEntry(
        kind="delete_records",
        copy=("我不能删除或清空你的训练或饮食数据，也不会声称已经删除成功；"
              "如需删除训练记录，可以在训练记录页对已有记录逐条操作。"),
        allows_chat_write=False, receipt_required=True,
        entry_screen="training", entry_label="训练记录"),
    "bulk_clear": CapabilityEntry(
        kind="bulk_clear",
        copy=("我不能批量清空训练或饮食数据，也不会声称已经删除成功；"
              "如需删除训练记录，可以在训练记录页对已有记录逐条操作。"),
        allows_chat_write=False, receipt_required=True,
        entry_screen="training", entry_label="训练记录"),
    "device_sync": CapabilityEntry(
        kind="device_sync",
        copy=("我不能直接同步 Apple Watch 或其他设备的训练、心率数据，也没有已连接或已导入的数据；"
              "你可以在训练记录页手动记录。"),
        allows_chat_write=False, receipt_required=False,
        entry_screen="training", entry_label="训练记录"),
    "general": CapabilityEntry(
        kind="general",
        copy=("我可以帮你记录训练和饮食、生成训练计划预览、读取已保存数据，并在明确授权时保存训练或饮食；"
              "不能直接同步设备、在聊天里保存计划，也不能批量清空记录。"),
        allows_chat_write=False, receipt_required=False),
}


def entry_for(kind: str | None) -> CapabilityEntry:
    return CAPABILITY_REGISTRY.get(str(kind or ""), CAPABILITY_REGISTRY["general"])


def capability_copy(kind: str | None) -> str:
    return entry_for(kind).copy


__all__ = ["CAPABILITY_REGISTRY", "CapabilityEntry", "capability_copy", "entry_for"]
