from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal

from .base import AgentResult, AgentTask


TaskStatus = Literal["queued", "running", "completed", "failed", "waiting_approval"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TaskState:
    task_id: str
    input_text: str
    status: TaskStatus = "queued"
    agent_type: str = ""
    routing_reason: str = ""
    current_agent: str | None = None
    output: str = ""
    steps: list[AgentResult] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


class TaskStateStore:
    """Thread-safe in-memory task state for the first orchestrator release.

    The interface intentionally hides storage details so SQLite/PostgreSQL can
    replace it when resumable multi-process execution is added.
    """

    def __init__(self, max_tasks: int = 10000) -> None:
        self._max_tasks = max(1, max_tasks)
        self._states: dict[str, TaskState] = {}
        self._lock = Lock()

    def create(self, task: AgentTask) -> TaskState:
        with self._lock:
            if len(self._states) >= self._max_tasks:
                oldest_id = next(iter(self._states), None)
                if oldest_id is not None:
                    self._states.pop(oldest_id, None)
            state = TaskState(task_id=task.task_id, input_text=task.input_text)
            self._states[task.task_id] = state
            return state

    def get(self, task_id: str) -> TaskState | None:
        with self._lock:
            return self._states.get(task_id)

    def update(self, task_id: str, **changes: Any) -> TaskState:
        with self._lock:
            state = self._states[task_id]
            for key, value in changes.items():
                setattr(state, key, value)
            state.updated_at = utc_now()
            return state

    def list(self) -> list[TaskState]:
        with self._lock:
            return list(self._states.values())
