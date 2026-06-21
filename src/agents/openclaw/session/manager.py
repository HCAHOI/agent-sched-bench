import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.openclaw.utils.helpers import (
    ensure_dir,
    find_legal_message_start,
    safe_filename,
)


@dataclass
class Session:
    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input."""
        unconsolidated = self.messages[self.last_consolidated :]
        sliced = unconsolidated[-max_messages:]

        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {
                "role": message["role"],
                "content": message.get("content", ""),
            }
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


def _default_sessions_dir() -> Path:
    """Last-resort sessions dir for direct SessionManager construction.

    Returns a non-workspace, user-level path (``OPENCLAW_SESSION_DIR`` >
    ``$XDG_STATE_HOME/openclaw/sessions`` > ``~/.local/state/openclaw/sessions``).
    Eval/CLI paths never hit this — they pass an explicit per-run ``storage_dir``
    derived from ``runtime_dir``. This fallback exists only so direct
    ``SessionManager(workspace)`` construction does not contaminate the workspace.
    """
    explicit = os.environ.get("OPENCLAW_SESSION_DIR")
    if explicit:
        return Path(explicit).expanduser()
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home).expanduser() / "openclaw" / "sessions"
    return Path.home() / ".local" / "state" / "openclaw" / "sessions"


class SessionManager:
    """Store conversation sessions as JSONL files."""

    def __init__(self, workspace: Path, *, storage_dir: Path | None = None):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(storage_dir or _default_sessions_dir())
        self._cache: dict[str, Session] = {}
        self._event_callback = None

    def _get_session_path(self, key: str) -> Path:
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """Get an existing session or create a new one."""
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        path = self._get_session_path(key)
        if not path.exists():
            return None

        messages = []
        metadata = {}
        created_at = None
        last_consolidated = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                if data.get("_type") == "metadata":
                    metadata = data.get("metadata", {})
                    created_at = (
                        datetime.fromisoformat(data["created_at"])
                        if data.get("created_at")
                        else None
                    )
                    last_consolidated = data.get("last_consolidated", 0)
                else:
                    messages.append(data)

        return Session(
            key=key,
            messages=messages,
            created_at=created_at or datetime.now(),
            metadata=metadata,
            last_consolidated=last_consolidated,
        )

    def save(self, session: Session) -> None:
        path = self._get_session_path(session.key)

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[session.key] = session
