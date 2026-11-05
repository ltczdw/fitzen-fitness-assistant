from typing import Any, TypedDict

class FitnessState(TypedDict, total=False):
    information_sources: list[str]
    clarifications: list[dict]
    user_id: str
    conversation_id: str
    raw_message: str
    normalized_message: str
    tasks: list[dict[str, Any]]
    planned_tasks: list[dict[str, Any]]
    current_task: dict[str, Any]
    completed_tasks: list[str]
    step_count: int
    max_steps: int
    safety_level: str
    profile: dict[str, Any]
    training_history: list[dict[str, Any]]
    nutrition_summary: dict[str, Any]
    short_memory: list[dict[str, str]]
    retrieved_history: list[dict[str, Any]]
    memory_summary: str
    semantic_memory: list[dict[str, Any]]
    message_importance: dict[str, float]
    long_memory: dict[str, Any]
    persistent_constraints: dict[str, Any]
    current_states: dict[str, Any]
    versioned_view: bool
    memory_epoch: int
    memory_backend: str
    current_plan_preview: dict[str, Any]
    plan_preview: dict[str, Any]
    worker_results: list[dict[str, Any]]
    task_outcomes: list[dict[str, str]]
    final_answer: str
    citations: list[dict]
    model_used: bool
    status: str
    intent_result: dict[str, Any]
    parsed_input: dict[str, Any]
    request_id: str


class WorkerState(TypedDict, total=False):
    information_sources: list[str]
    user_id: str
    conversation_id: str
    raw_message: str
    profile: dict[str, Any]
    training_history: list[dict[str, Any]]
    nutrition_summary: dict[str, Any]
    short_memory: list[dict[str, str]]
    retrieved_history: list[dict[str, Any]]
    memory_summary: str
    semantic_memory: list[dict[str, Any]]
    message_importance: dict[str, float]
    long_memory: dict[str, Any]
    persistent_constraints: dict[str, Any]
    current_states: dict[str, Any]
    memory_backend: str
    current_plan_preview: dict[str, Any]
    intent_result: dict[str, Any]
    result: dict[str, Any]
    task_id: str
    task_type: str
    task_action: str
    task_text: str
    task_context: list[dict[str, Any]]
    task_payload: dict[str, Any]
    semantic_scope: dict[str, Any]
    shared_conditions: list[str]
    plan_siblings: list[dict[str, Any]]
    source_availability: dict[str, str]
    plan_contract: dict[str, Any]
    plan_target_count: int
    plan_target_only: bool
    request_id: str


