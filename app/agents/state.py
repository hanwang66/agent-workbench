from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
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
    """Thread-safe in-memory task state store used for tests and development."""

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


def _serialize_steps(steps: list[AgentResult]) -> str:
    return json.dumps(
        [
            {
                "agent_name": step.agent_name,
                "status": step.status,
                "output": step.output,
                "metadata": step.metadata,
                "artifacts": step.artifacts,
                "tool_traces": step.tool_traces,
                "error": step.error,
            }
            for step in steps
        ],
        ensure_ascii=False,
        default=str,
    )


def _deserialize_steps(raw_steps: str) -> list[AgentResult]:
    try:
        values = json.loads(raw_steps or "[]")
    except json.JSONDecodeError:
        values = []
    if not isinstance(values, list):
        return []

    steps: list[AgentResult] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        steps.append(
            AgentResult(
                agent_name=str(value.get("agent_name", "")),
                status=value.get("status", "failed"),  # type: ignore[arg-type]
                output=str(value.get("output", "")),
                metadata=dict(value.get("metadata") or {}),
                artifacts=list(value.get("artifacts") or []),
                tool_traces=list(value.get("tool_traces") or []),
                error=value.get("error"),
            )
        )
    return steps


class SQLiteTaskStateStore:
    """Durable task state store with the same interface as ``TaskStateStore``.

    SQLite keeps the default single-process deployment restart-safe without
    adding another service. The store boundary leaves room for PostgreSQL when
    multiple API workers need shared state.
    """

    def __init__(self, db_path: str | Path, max_tasks: int = 10000) -> None:
        self._max_tasks = max(1, max_tasks)
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = Lock()
        self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_states (
                    task_id TEXT PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    routing_reason TEXT NOT NULL,
                    current_agent TEXT,
                    output TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_states_created_at "
                "ON task_states(created_at)"
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> TaskState:
        return TaskState(
            task_id=str(row["task_id"]),
            input_text=str(row["input_text"]),
            status=row["status"],  # type: ignore[arg-type]
            agent_type=str(row["agent_type"]),
            routing_reason=str(row["routing_reason"]),
            current_agent=row["current_agent"],
            output=str(row["output"]),
            steps=_deserialize_steps(str(row["steps_json"])),
            error=row["error"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _save(self, state: TaskState) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO task_states (
                task_id, input_text, status, agent_type, routing_reason,
                current_agent, output, steps_json, error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.task_id,
                state.input_text,
                state.status,
                state.agent_type,
                state.routing_reason,
                state.current_agent,
                state.output,
                _serialize_steps(state.steps),
                state.error,
                state.created_at,
                state.updated_at,
            ),
        )

    def _trim(self) -> None:
        self._connection.execute(
            """
            DELETE FROM task_states
            WHERE task_id IN (
                SELECT task_id FROM task_states
                ORDER BY created_at ASC
                LIMIT MAX(0, (SELECT COUNT(*) FROM task_states) - ?)
            )
            """,
            (self._max_tasks,),
        )

    def create(self, task: AgentTask) -> TaskState:
        state = TaskState(task_id=task.task_id, input_text=task.input_text)
        with self._lock, self._connection:
            self._save(state)
            self._trim()
        return state

    def get(self, task_id: str) -> TaskState | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM task_states WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def update(self, task_id: str, **changes: Any) -> TaskState:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM task_states WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            state = self._from_row(row)
            for key, value in changes.items():
                setattr(state, key, value)
            state.updated_at = utc_now()
            self._save(state)
            return state

    def list(self) -> list[TaskState]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM task_states ORDER BY created_at ASC"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
