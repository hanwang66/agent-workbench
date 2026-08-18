from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """Resource and privilege limits applied to one coding-agent command."""

    image: str = "agent-workbench-sandbox:py312"
    docker_binary: str = "docker"
    enabled: bool = True
    cpus: float = 1.0
    memory: str = "512m"
    pids_limit: int = 128
    output_limit_bytes: int = 64 * 1024
    tmpfs_size: str = "64m"


class SandboxExecutor:
    """Run fixed coding-agent commands in a short-lived restricted container.

    The executor deliberately accepts an argv list and never invokes a shell.
    Command selection remains the responsibility of CodingAgent's allowlist.
    """

    def __init__(self, policy: SandboxPolicy | None = None, timeout_seconds: int = 30) -> None:
        self._policy = policy or SandboxPolicy()
        self._timeout_seconds = max(1, timeout_seconds)

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    def build_command(self, workspace: Path, command: Sequence[str], *, writable: bool = False) -> list[str]:
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("Sandbox command must be a non-empty argv list")
        resolved_workspace = workspace.resolve()
        if not resolved_workspace.is_dir():
            raise ValueError(f"Sandbox workspace is not a directory: {resolved_workspace}")
        if "," in str(resolved_workspace):
            raise ValueError("Sandbox workspace paths containing commas are not supported")

        access = "rw" if writable else "readonly"
        docker_command = [
            self._policy.docker_binary,
            "run",
            "--rm",
            "--init",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={self._policy.tmpfs_size}",
            "--mount",
            f"type=bind,source={resolved_workspace},target=/workspace,{access}",
            "--workdir",
            "/workspace",
            "--cap-drop=ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            str(max(1, self._policy.pids_limit)),
            "--cpus",
            str(max(0.1, self._policy.cpus)),
            "--memory",
            self._policy.memory,
            "--memory-swap",
            self._policy.memory,
            "--ulimit",
            "nofile=256:256",
        ]

        uid = getattr(os, "getuid", lambda: None)()
        gid = getattr(os, "getgid", lambda: None)()
        if uid is not None and gid is not None:
            docker_command.extend(["--user", f"{uid}:{gid}"])

        docker_command.extend([self._policy.image, *command])
        return docker_command

    async def run(self, workspace: Path, command: Sequence[str], *, writable: bool = False) -> dict[str, object]:
        if not self._policy.enabled:
            return await self._run_host(workspace, command)

        docker_command = self.build_command(workspace, command, writable=writable)
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Sandbox runtime '{self._policy.docker_binary}' was not found; "
                "install Docker or set CODING_SANDBOX_ENABLED=false for trusted local development"
            ) from exc

        return await self._collect_process_result(process, command)

    async def _run_host(self, workspace: Path, command: Sequence[str]) -> dict[str, object]:
        """Explicit development fallback; never enabled by default."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Command was not found: {command[0]}") from exc

        return await self._collect_process_result(process, command)

    async def _collect_process_result(
        self,
        process: asyncio.subprocess.Process,
        command: Sequence[str],
    ) -> dict[str, object]:
        stdout_task = asyncio.create_task(self._read_limited(process.stdout))
        stderr_task = asyncio.create_task(self._read_limited(process.stderr))
        output_task = asyncio.gather(stdout_task, stderr_task)
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(output_task), timeout=self._timeout_seconds)
            timed_out = False
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            stdout, stderr = await output_task
            timed_out = True

        await process.wait()

        return {
            "command": list(command),
            "returncode": 124 if timed_out else process.returncode,
            "stdout": stdout["text"],
            "stderr": stderr["text"],
            "stdout_truncated": stdout["truncated"],
            "stderr_truncated": stderr["truncated"],
            "timed_out": timed_out,
            "sandboxed": self._policy.enabled,
        }

    async def _read_limited(self, stream: asyncio.StreamReader | None) -> dict[str, object]:
        if stream is None:
            return {"text": "", "truncated": False}

        limit = max(1024, self._policy.output_limit_bytes)
        chunks: list[bytes] = []
        total = 0
        truncated = False
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                break
            if total < limit:
                remaining = limit - total
                chunks.append(chunk[:remaining])
            if total + len(chunk) > limit:
                truncated = True
            total += len(chunk)

        return {
            "text": b"".join(chunks).decode("utf-8", errors="replace"),
            "truncated": truncated,
        }


__all__ = ["SandboxExecutor", "SandboxPolicy"]
