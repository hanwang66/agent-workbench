from __future__ import annotations

from dataclasses import dataclass

from .base import Agent


class AgentNotFoundError(LookupError):
    """Raised when a requested worker is not registered."""


class DuplicateAgentError(ValueError):
    """Raised when two workers use the same registry name."""


@dataclass(frozen=True, slots=True)
class AgentInfo:
    name: str
    description: str
    capabilities: tuple[str, ...]


class AgentRegistry:
    """A small explicit registry so worker capabilities stay inspectable."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent, *, replace: bool = False) -> None:
        name = agent.name.strip().lower()
        if not name:
            raise ValueError("Agent name must not be empty")
        if name in self._agents and not replace:
            raise DuplicateAgentError(f"Agent '{name}' is already registered")
        self._agents[name] = agent

    def get(self, name: str) -> Agent:
        normalized = name.strip().lower()
        try:
            return self._agents[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._agents)) or "none"
            raise AgentNotFoundError(
                f"Agent '{normalized or '<empty>'}' is not registered. Available agents: {available}"
            ) from exc

    def has(self, name: str) -> bool:
        return name.strip().lower() in self._agents

    def list_agents(self) -> list[AgentInfo]:
        return [
            AgentInfo(
                name=agent.name,
                description=agent.description,
                capabilities=tuple(agent.capabilities),
            )
            for agent in sorted(self._agents.values(), key=lambda item: item.name)
        ]
