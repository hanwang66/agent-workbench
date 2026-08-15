from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from ..ollama_client import OllamaClient
from .base import AgentContext, AgentResult, AgentTask


CODING_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List repository files under a relative directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "max_depth": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search plain text in repository files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_git_diff",
            "description": "Inspect the current git diff without changing files.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run a fixed repository validation command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "suite": {
                        "type": "string",
                        "enum": ["python_unittest", "git_diff_check"],
                    }
                },
                "required": ["suite"],
            },
        },
    },
]

WRITE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a UTF-8 text file after explicit human approval.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}


class CodingAgent:
    name = "coding"
    description = "Inspect a repository, propose code changes, and run safe validation tools."
    capabilities = ("code_search", "file_inspection", "git_diff", "test_execution")

    def __init__(
        self,
        client: OllamaClient,
        workspace_root: str = ".",
        max_rounds: int = 6,
        max_tool_calls: int = 12,
        max_file_bytes: int = 128 * 1024,
        command_timeout_seconds: int = 30,
    ) -> None:
        self._client = client
        self._workspace_root = Path(workspace_root).resolve()
        self._max_rounds = max(1, max_rounds)
        self._max_tool_calls = max(1, max_tool_calls)
        self._max_file_bytes = max(1, max_file_bytes)
        self._command_timeout_seconds = max(1, command_timeout_seconds)

    async def run(self, task: AgentTask, context: AgentContext) -> AgentResult:
        workspace = self._resolve_path(str(task.parameters.get("workspace") or "."))
        if not workspace.is_dir():
            raise ValueError(f"Coding workspace is not a directory: {workspace}")

        write_approved = task.parameters.get("write_approved") is True
        tools = list(CODING_TOOLS)
        if write_approved:
            tools.append(WRITE_TOOL)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a careful coding agent. Inspect the repository before making claims. "
                    "Use repository tools for evidence, never invent file contents, and summarize "
                    "tests and remaining risks. Only write files when the write_file tool is available."
                ),
            },
            {
                "role": "user",
                "content": f"Workspace: {workspace}\nTask:\n{task.input_text}",
            },
        ]
        if context.previous_results:
            previous = "\n\n".join(
                f"{result.agent_name}: {result.output[:4000]}" for result in context.previous_results
            )
            messages.append({"role": "user", "content": f"Previous worker results:\n{previous}"})
        traces: list[dict[str, Any]] = []
        tool_calls_count = 0
        last_content = ""

        for round_index in range(1, self._max_rounds + 1):
            reply = await self._client.chat(messages=messages, tools=tools)
            content = str(reply.get("content") or "").strip()
            if content:
                last_content = content

            raw_tool_calls = reply.get("tool_calls")
            if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
                if last_content:
                    return AgentResult(
                        agent_name=self.name,
                        status="completed",
                        output=last_content,
                        metadata={
                            "workspace": str(workspace),
                            "write_approved": write_approved,
                            "rounds": round_index,
                        },
                        tool_traces=traces,
                    )
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": str(reply.get("content") or ""),
                    "tool_calls": raw_tool_calls,
                }
            )

            for call in raw_tool_calls:
                payload = call.get("function") if isinstance(call, dict) else {}
                payload = payload if isinstance(payload, dict) else {}
                tool_name = str(payload.get("name") or "")
                arguments = self._parse_arguments(payload.get("arguments"))

                if tool_calls_count >= self._max_tool_calls:
                    trace = self._trace(round_index, tool_name, "blocked", "tool call limit reached")
                    traces.append(trace)
                    tool_result = {"error": trace["detail"]}
                elif tool_name == "write_file" and not write_approved:
                    trace = self._trace(round_index, tool_name, "blocked", "human approval required")
                    traces.append(trace)
                    tool_result = {"error": trace["detail"]}
                else:
                    tool_calls_count += 1
                    try:
                        tool_result = await self._execute_tool(tool_name, arguments, workspace)
                        trace = self._trace(round_index, tool_name, "executed", "ok")
                    except Exception as exc:
                        tool_result = {"error": str(exc)}
                        trace = self._trace(round_index, tool_name, "error", str(exc))
                    traces.append(trace)

                messages.append(
                    {
                        "role": "tool",
                        "name": tool_name or "unknown",
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        if last_content:
            return AgentResult(
                agent_name=self.name,
                status="completed",
                output=last_content,
                metadata={"workspace": str(workspace), "write_approved": write_approved},
                tool_traces=traces,
            )
        return AgentResult(
            agent_name=self.name,
            status="failed",
            error="Coding agent reached its round limit without a final response",
            metadata={"workspace": str(workspace), "write_approved": write_approved},
            tool_traces=traces,
        )

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                value = json.loads(raw)
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _trace(round_index: int, tool_name: str, status: str, detail: str) -> dict[str, Any]:
        return {
            "round_index": round_index,
            "tool_name": tool_name or "<unknown>",
            "status": status,
            "detail": detail,
        }

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self._workspace_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ValueError("Path escapes the configured coding workspace") from exc
        return resolved

    async def _execute_tool(self, name: str, arguments: dict[str, Any], workspace: Path) -> dict[str, Any]:
        if name == "list_files":
            return self._list_files(workspace, arguments)
        if name == "search_text":
            return self._search_text(workspace, arguments)
        if name == "read_file":
            return self._read_file(workspace, arguments)
        if name == "get_git_diff":
            return await asyncio.to_thread(self._run_command, workspace, ["git", "diff", "--no-ext-diff"])
        if name == "run_tests":
            suite = str(arguments.get("suite") or "")
            if suite == "python_unittest":
                command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
            elif suite == "git_diff_check":
                command = ["git", "diff", "--check"]
            else:
                raise ValueError("Unsupported test suite")
            return await asyncio.to_thread(self._run_command, workspace, command)
        if name == "write_file":
            return self._write_file(workspace, arguments)
        raise ValueError(f"Tool '{name}' is not available to CodingAgent")

    def _list_files(self, workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        start = self._resolve_path(str(arguments.get("path") or "."))
        if not start.is_dir():
            raise ValueError(f"Not a directory: {start}")
        max_depth = max(0, min(int(arguments.get("max_depth") or 2), 6))
        entries: list[str] = []
        for current, dirs, files in os.walk(start, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.relative_to(start).parts)
            dirs[:] = sorted(
                item for item in dirs if item not in {".git", ".venv", "__pycache__", "node_modules", "data"}
            )
            if depth > max_depth:
                dirs[:] = []
                continue
            for filename in sorted(files):
                path = current_path / filename
                entries.append(str(path.relative_to(workspace)))
                if len(entries) >= 200:
                    return {"files": entries, "truncated": True}
        return {"files": entries, "truncated": False}

    def _search_text(self, workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = str(arguments.get("pattern") or "")
        if not pattern:
            raise ValueError("Search pattern must not be empty")
        start = self._resolve_path(str(arguments.get("path") or "."))
        max_results = max(1, min(int(arguments.get("max_results") or 50), 100))
        results: list[dict[str, Any]] = []
        for current, dirs, files in os.walk(start, followlinks=False):
            dirs[:] = sorted(
                item for item in dirs if item not in {".git", ".venv", "__pycache__", "node_modules", "data"}
            )
            for filename in sorted(files):
                path = Path(current) / filename
                try:
                    path.resolve().relative_to(self._workspace_root)
                except ValueError:
                    # Do not follow a symlinked file outside the configured
                    # workspace while searching repository contents.
                    continue
                try:
                    content = self._read_text(path)
                except (OSError, UnicodeDecodeError, ValueError):
                    continue
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if pattern.lower() in line.lower():
                        results.append(
                            {"path": str(path.relative_to(workspace)), "line": line_number, "text": line[:500]}
                        )
                        if len(results) >= max_results:
                            return {"matches": results, "truncated": True}
        return {"matches": results, "truncated": False}

    def _read_file(self, workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(str(arguments.get("path") or ""))
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        return {"path": str(path.relative_to(workspace)), "content": self._read_text(path)}

    def _read_text(self, path: Path) -> str:
        if path.stat().st_size > self._max_file_bytes:
            raise ValueError(f"File exceeds the {self._max_file_bytes}-byte read limit")
        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ValueError("Binary files are not supported")
        return raw.decode("utf-8")

    def _write_file(self, workspace: Path, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(str(arguments.get("path") or ""))
        content = str(arguments.get("content") or "")
        encoded = content.encode("utf-8")
        if len(encoded) > self._max_file_bytes:
            raise ValueError(f"File exceeds the {self._max_file_bytes}-byte write limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": str(path.relative_to(workspace)), "bytes_written": len(encoded)}

    def _run_command(self, workspace: Path, command: list[str]) -> dict[str, Any]:
        completed = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=self._command_timeout_seconds,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
        }
