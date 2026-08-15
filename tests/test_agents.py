from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from app.agents import AgentContext, AgentRegistry, AgentResult, AgentTask, CodingAgent, Orchestrator


class EchoAgent:
    name = "echo"
    description = "Echoes the task for orchestration tests."
    capabilities = ("echo",)

    async def run(self, task: AgentTask, context: AgentContext) -> AgentResult:
        return AgentResult(agent_name=self.name, status="completed", output=task.input_text)


class SuffixAgent:
    name = "suffix"
    description = "Adds a suffix for orchestration tests."
    capabilities = ("suffix",)

    async def run(self, task: AgentTask, context: AgentContext) -> AgentResult:
        previous = context.previous_results[-1].output if context.previous_results else ""
        return AgentResult(agent_name=self.name, status="completed", output=f"{previous}:{task.input_text}")


class UnusedClient:
    pass


class InspectingClient:
    def __init__(self) -> None:
        self.calls = 0
        self.tools_seen: list[dict[str, object]] = []

    async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None) -> dict[str, object]:
        self.calls += 1
        self.tools_seen = list(tools or [])
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": '{"path":"hello.py"}'}},
                ],
            }
        return {"content": "The file prints hello."}


class AgentRuntimeTests(unittest.TestCase):
    def test_registry_exposes_registered_agent(self) -> None:
        registry = AgentRegistry()
        registry.register(EchoAgent())

        self.assertTrue(registry.has("echo"))
        self.assertEqual(registry.list_agents()[0].capabilities, ("echo",))

    def test_orchestrator_runs_explicit_worker(self) -> None:
        registry = AgentRegistry()
        registry.register(EchoAgent())
        orchestrator = Orchestrator(registry)

        result = asyncio.run(
            orchestrator.run(
                AgentTask(input_text="hello", agent_type="echo"),
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.agent_type, "echo")
        self.assertEqual(result.output, "hello")
        self.assertEqual(result.routing_reason, "explicit_agent_type")
        self.assertEqual(len(result.steps), 1)

    def test_coding_agent_rejects_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = CodingAgent(client=UnusedClient(), workspace_root=tmp)

            with self.assertRaises(ValueError):
                agent._resolve_path("../outside")

            source = Path(tmp) / "hello.py"
            source.write_text("print('hello')\n", encoding="utf-8")
            result = agent._read_file(Path(tmp).resolve(), {"path": "hello.py"})
            self.assertIn("print('hello')", result["content"])

    def test_coding_agent_runs_read_only_tool_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "hello.py"
            source.write_text("print('hello')\n", encoding="utf-8")
            client = InspectingClient()
            agent = CodingAgent(client=client, workspace_root=tmp, max_rounds=2)

            result = asyncio.run(
                agent.run(
                    AgentTask(input_text="Inspect hello.py", agent_type="coding"),
                    AgentContext(task=AgentTask(input_text="Inspect hello.py", agent_type="coding")),
                )
            )

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.output, "The file prints hello.")
            self.assertEqual(result.tool_traces[0]["tool_name"], "read_file")
            self.assertTrue(all(item["function"]["name"] != "write_file" for item in client.tools_seen))

    def test_orchestrator_executes_explicit_multi_step_plan(self) -> None:
        registry = AgentRegistry()
        registry.register(EchoAgent())
        registry.register(SuffixAgent())
        orchestrator = Orchestrator(registry)

        result = asyncio.run(
            orchestrator.run(
                AgentTask(
                    input_text="root task",
                    parameters={
                        "plan": [
                            {"agent_type": "echo", "task": "first", "parameters": {}},
                            {"agent_type": "suffix", "task": "second", "parameters": {}},
                        ]
                    },
                )
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output, "first:second")
        self.assertEqual(result.metadata["step_count"], 2)
        self.assertEqual(orchestrator.state_store.get(result.task_id).status, "completed")


if __name__ == "__main__":
    unittest.main()
