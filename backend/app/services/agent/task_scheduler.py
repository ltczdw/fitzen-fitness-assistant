from __future__ import annotations

from typing import Any, Iterable


def select_ready_task(tasks: Iterable[dict[str, Any]], completed_ids: Iterable[str]) -> dict[str, Any] | None:
    """Select the first pending task whose dependencies are complete.

    The graph executes one task at a time for deterministic writes. The task
    list is still dependency-aware, so independent same-type tasks are not
    deduplicated and can be parallelized later without changing the contract.
    """
    completed = set(completed_ids)
    for task in tasks:
        if task.get("execution_status", task.get("status", "pending")) != "pending":
            continue
        if all(dependency in completed for dependency in task.get("depends_on", [])):
            return task
    return None


def has_unresolved_dependency(tasks: Iterable[dict[str, Any]], completed_ids: Iterable[str]) -> bool:
    completed = set(completed_ids)
    pending = [task for task in tasks if task.get("execution_status", task.get("status", "pending")) == "pending"]
    return bool(pending) and not any(all(dependency in completed for dependency in task.get("depends_on", [])) for task in pending)

