from __future__ import annotations

from dataclasses import dataclass, field

from .base import AgentContext, AgentResult, AgentStatus, AgentTask
from .registry import AgentRegistry
from .state import TaskStateStore


@dataclass(slots=True)
class OrchestrationResult:
    task_id: str
    agent_type: str
    status: AgentStatus
    output: str
    routing_reason: str
    steps: list[AgentResult] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    error: str | None = None


class Orchestrator:
    """Centralized router and execution boundary for all workers.

    The first version deliberately uses deterministic routing. A model-driven
    planner can be added later without changing the worker interface.
    """

    _ROUTE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "coding",
            ("代码", "编码", "编程", "bug", "debug", "修复", "实现", "测试", "repository", "repo"),
        ),
        ("research", ("研究", "调研", "搜索", "资料", "research", "分析报告")),
        ("translation", ("翻译", "翻译成", "translate", "localize", "本地化")),
    )

    def __init__(self, registry: AgentRegistry, max_steps: int = 8, state_store: TaskStateStore | None = None) -> None:
        self._registry = registry
        self._max_steps = max(1, max_steps)
        self._state_store = state_store or TaskStateStore()

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    @property
    def state_store(self) -> TaskStateStore:
        return self._state_store

    def route(self, task: AgentTask) -> tuple[str, str]:
        if task.agent_type:
            return task.agent_type.strip().lower(), "explicit_agent_type"

        lowered = task.input_text.lower()
        for agent_name, keywords in self._ROUTE_KEYWORDS:
            if any(keyword in lowered for keyword in keywords) and self._registry.has(agent_name):
                return agent_name, f"keyword_route:{agent_name}"

        if self._registry.has("translation"):
            return "translation", "default_worker"

        return "", "no_matching_worker"

    async def run(self, task: AgentTask) -> OrchestrationResult:
        self._state_store.create(task)
        try:
            plan = self._normalize_plan(task)
            steps: list[AgentResult] = []
            shared_state: dict[str, object] = {"parent_task_id": task.task_id}
            routing_reasons: list[str] = []

            self._state_store.update(task.task_id, status="running")
            for index, (step_task, explicit_reason) in enumerate(plan, start=1):
                if index > self._max_steps:
                    raise RuntimeError("Orchestrator step limit exceeded")

                agent_name, routing_reason = self.route(step_task)
                if explicit_reason:
                    routing_reason = explicit_reason
                agent = self._registry.get(agent_name)
                self._state_store.update(
                    task.task_id,
                    agent_type=agent_name,
                    routing_reason=routing_reason,
                    current_agent=agent_name,
                )
                result = await agent.run(
                    task=step_task,
                    context=AgentContext(task=step_task, shared_state=shared_state, previous_results=list(steps)),
                )
                steps.append(result)
                shared_state[f"step_{index}"] = result.output
                routing_reasons.append(routing_reason)
                self._state_store.update(task.task_id, steps=list(steps))
                if result.status != "completed":
                    break

            final = steps[-1] if steps else AgentResult(
                agent_name="orchestrator",
                status="failed",
                error="No execution step was produced",
            )
            routing_reason = " -> ".join(routing_reasons)
            self._state_store.update(
                task.task_id,
                status=final.status,
                output=final.output,
                error=final.error,
                current_agent=None,
            )
            return OrchestrationResult(
                task_id=task.task_id,
                agent_type=steps[0].agent_name if steps else "",
                status=final.status,
                output=final.output,
                routing_reason=routing_reason,
                steps=steps,
                metadata={"step_count": len(steps), "agent_types": [step.agent_name for step in steps]},
                error=final.error,
            )
        except Exception as exc:
            self._state_store.update(task.task_id, status="failed", error=str(exc), current_agent=None)
            raise

    def _normalize_plan(self, task: AgentTask) -> list[tuple[AgentTask, str]]:
        raw_plan = task.parameters.get("plan")
        if raw_plan is None:
            return [(task, "")]
        if not isinstance(raw_plan, list) or not raw_plan:
            raise ValueError("parameters.plan must be a non-empty list")
        if len(raw_plan) > self._max_steps:
            raise ValueError(f"parameters.plan cannot contain more than {self._max_steps} steps")

        plan: list[tuple[AgentTask, str]] = []
        for index, raw_step in enumerate(raw_plan, start=1):
            if not isinstance(raw_step, dict):
                raise ValueError(f"parameters.plan step {index} must be an object")
            input_text = str(raw_step.get("task") or "").strip()
            agent_type = str(raw_step.get("agent_type") or "").strip() or None
            parameters = raw_step.get("parameters") or {}
            if not input_text or not isinstance(parameters, dict):
                raise ValueError(f"parameters.plan step {index} needs task and object parameters")
            plan.append(
                (
                    AgentTask(
                        input_text=input_text,
                        agent_type=agent_type,
                        task_id=f"{task.task_id}:step-{index}",
                        session_id=task.session_id,
                        knowledge_base_id=task.knowledge_base_id,
                        parameters=dict(parameters),
                        parent_task_id=task.task_id,
                    ),
                    f"explicit_plan_step:{index}",
                )
            )
        return plan
