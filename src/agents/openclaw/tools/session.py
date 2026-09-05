"""Session orchestration tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agents.openclaw.tools.base import Tool

if TYPE_CHECKING:
    from agents.openclaw._subagent import SubagentManager


@dataclass(frozen=True, slots=True)
class SessionYieldResult:
    """Tool result that can ask the runner to end the current model turn."""

    content: str
    should_yield: bool

    def __str__(self) -> str:
        return self.content


class SessionsYieldTool(Tool):
    """End the current turn so runtime follow-up events can arrive next."""

    def __init__(self, manager: "SubagentManager") -> None:
        self._manager = manager
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "sessions_yield"

    @property
    def description(self) -> str:
        return (
            "End the current model turn and wait for runtime events, primarily "
            "subagent completion events, to arrive as the next message. Use after "
            "spawning required background work when a final answer depends on it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> SessionYieldResult:
        del kwargs
        if not self._manager.has_active(self._session_key):
            return SessionYieldResult(
                content=(
                    "No active subagents are pending for this session. Continue the "
                    "turn normally instead of waiting."
                ),
                should_yield=False,
            )
        return SessionYieldResult(
            content="Yielded this turn. Continue when the next runtime event arrives.",
            should_yield=True,
        )
