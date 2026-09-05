from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TraceAction:
    """A single replayable action in an agent trace (v4 format).

    Each action is one executable operation:
    - ``llm_call``: an LLM inference (input: messages_in; output: raw_response)
    - ``tool_exec``: a tool execution (input: tool_name+args; output: result)

    Multiple actions can share the same ``iteration`` value (e.g., one LLM call
    followed by parallel tool executions).
    """

    action_type: str  # "llm_call" | "tool_exec"
    action_id: str  # unique within trace, e.g. "llm_0", "tool_0_bash"
    agent_id: str = ""
    program_id: str = ""
    instance_id: str = ""
    iteration: int = 0
    ts_start: float = 0.0
    ts_end: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    type: str = "action"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "action_type": self.action_type,
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "program_id": self.program_id,
            "instance_id": self.instance_id,
            "iteration": self.iteration,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "data": self.data,
        }


