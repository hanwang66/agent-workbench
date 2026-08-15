from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4


AgentStatus = Literal["completed", "failed", "waiting_approval"]


@dataclass(slots=True)
class AgentTask:
    """The task envelope shared by the orchestrator and every worker."""

    input_text: str
    agent_type: str | None = None
    task_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    knowledge_base_id: str = "default"
    parameters: dict[str, Any] = field(default_factory=dict)
    parent_task_id: str | None = None


@dataclass(slots=True)
class AgentResult:
    """A normalized worker result that the orchestrator can persist or relay."""

    agent_name: str
    status: AgentStatus
    output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    tool_traces: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class AgentContext:
    """Context visible to one worker; workers do not call each other directly."""

    task: AgentTask
    shared_state: dict[str, Any] = field(default_factory=dict)
    previous_results: list[AgentResult] = field(default_factory=list)


class Agent(Protocol):
    name: str
    description: str
    capabilities: tuple[str, ...]

    async def run(self, task: AgentTask, context: AgentContext) -> AgentResult:
        ...
