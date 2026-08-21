from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from app.agents import (
    AgentContext,
    AgentRegistry,
    AgentResult,
    AgentTask,
    CodingAgent,
    Orchestrator,
    SandboxExecutor,
    SandboxPolicy,
    SQLiteTaskStateStore,
)


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


class SandboxClient:
    def __init__(self, tool_name: str, arguments: str) -> None:
        self.calls = 0
        self.tool_name = tool_name
        self.arguments = arguments

    async def chat(self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None) -> dict[str, object]:
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [{"function": {"name": self.tool_name, "arguments": self.arguments}}],
            }
        return {"content": "Sandbox result recorded."}


class RecordingSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, list[str], bool]] = []

    async def run(self, workspace: Path, command: list[str], *, writable: bool = False) -> dict[str, object]:
        self.calls.append((workspace, command, writable))
        return {
            "command": command,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "sandboxed": True,
        }


class AgentRuntimeTests(unittest.TestCase):
    def test_sqlite_task_state_survives_store_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = AgentTask(input_text="persist me", agent_type="echo")
            first_store = SQLiteTaskStateStore(Path(tmp) / "tasks.sqlite3", max_tasks=2)
            first_store.create(task)
            first_store.update(
                task.task_id,
                status="completed",
                agent_type="echo",
                output="saved",
                steps=[AgentResult(agent_name="echo", status="completed", output="saved")],
            )
            first_store.close()

            second_store = SQLiteTaskStateStore(Path(tmp) / "tasks.sqlite3", max_tasks=2)
            restored = second_store.get(task.task_id)
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.status, "completed")
            self.assertEqual(restored.steps[0].output, "saved")
            second_store.close()

    def test_sqlite_task_state_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteTaskStateStore(Path(tmp) / "tasks.sqlite3", max_tasks=1)
            first = AgentTask(input_text="first")
            second = AgentTask(input_text="second")
            store.create(first)
            store.create(second)
            self.assertIsNone(store.get(first.task_id))
            self.assertEqual([item.task_id for item in store.list()], [second.task_id])
            store.close()

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

    def test_sandbox_builds_restricted_docker_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = SandboxExecutor(
                policy=SandboxPolicy(image="sandbox:test", docker_binary="docker"),
            )
            command = executor.build_command(Path(tmp), ["python", "-m", "unittest"])

            self.assertIn("--network=none", command)
            self.assertIn("--read-only", command)
            self.assertIn("--cap-drop=ALL", command)
            self.assertIn("--security-opt", command)
            self.assertIn("no-new-privileges=true", command)
            self.assertIn("--memory-swap", command)
            self.assertEqual(command[-4:], ["sandbox:test", "python", "-m", "unittest"])

    def test_coding_agent_routes_validation_to_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = RecordingSandbox()
            client = SandboxClient("run_tests", '{"suite":"python_unittest"}')
            agent = CodingAgent(client=client, workspace_root=tmp, sandbox=sandbox)  # type: ignore[arg-type]

            result = asyncio.run(
                agent.run(
                    AgentTask(input_text="Run the tests", agent_type="coding"),
                    AgentContext(task=AgentTask(input_text="Run the tests", agent_type="coding")),
                )
            )

            self.assertEqual(result.status, "completed")
            self.assertEqual(len(sandbox.calls), 1)
            self.assertEqual(sandbox.calls[0][1], ["python", "-m", "unittest", "discover", "-s", "tests", "-v"])
            self.assertFalse(sandbox.calls[0][2])

    def test_approved_writes_are_isolated_and_return_a_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "hello.py"
            source.write_text("print('hello')\n", encoding="utf-8")
            client = SandboxClient("write_file", "{\"path\":\"hello.py\",\"content\":\"print('changed')\\n\"}")
            agent = CodingAgent(client=client, workspace_root=tmp, sandbox=RecordingSandbox())  # type: ignore[arg-type]

            result = asyncio.run(
                agent.run(
                    AgentTask(
                        input_text="Update hello.py",
                        agent_type="coding",
                        parameters={"write_approved": True},
                    ),
                    AgentContext(task=AgentTask(input_text="Update hello.py", agent_type="coding")),
                )
            )

            self.assertEqual(result.status, "waiting_approval")
            self.assertEqual(source.read_text(encoding="utf-8"), "print('hello')\n")
            self.assertTrue(result.metadata["changes_proposed"])
            self.assertFalse(result.metadata["changes_applied"])
            self.assertIn("print('changed')", result.artifacts[0]["content"])

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
