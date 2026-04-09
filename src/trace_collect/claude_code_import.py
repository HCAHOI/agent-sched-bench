"""Import a Claude Code session JSONL into a v5 trace for the Gantt viewer.

Reads a Claude Code (Anthropic CLI) session file — the JSONL transcript
written by ``claude-code`` to ``~/.claude/projects/<slug>/*.jsonl`` —
and emits a v5-shape JSONL the existing Gantt tooling can consume with
zero renderer modification.

Direct JSONL writes (no ``TraceLogger``), hand-constructed metadata
header, one pass to harvest metadata + one pass to stream records. No
collection, no simulation — pure post-hoc conversion for visualization.

Rich Claude Code fields are preserved as additive backfill under
``data.*`` on actions and ``metadata.run_config.*`` on the header, per
the v5 extension convention. See
``docs/plans/claude-code-gantt-import.md`` for the full field mapping
table and operator runbook.

Key backfill sources:
- ``assistant.message.usage.cache_*``  → cache-token accounting
- ``assistant.message.content[type="thinking"]`` → extended thinking
- top-level ``cwd``, ``gitBranch``, ``version`` → metadata.run_config
- top-level ``toolUseResult`` sidecar on user records →
  ``totalDurationMs`` (the pre-computed tool duration jackpot),
  ``agentId`` / ``agentType`` (subagent provenance), ``totalTokens`` /
  ``usage`` (subagent token accounting).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


SCAFFOLD_LABEL = "claude-code"
V5_FORMAT_VERSION = 5
_TOOL_RESULT_MAX_CHARS = 8000  # storage cap; viewer truncates display separately

# Canonical 36-char lowercase-hex UUID shape. Used to decide whether the
# session filename stem is itself the session UUID (native CC layout at
# ``~/.claude/projects/<slug>/<uuid>.jsonl``) or a generic label like
# ``trace`` (collector layout that copies the session file out of a
# container as ``trace.jsonl``). When the stem is not UUID-shaped we
# fall back to the ``sessionId`` harvested from the records themselves.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

# Record types that are internal Claude Code state and produce no v5 output.
# ``last-prompt`` arrived on CC CLI 2.1.96 — it's a trailing snapshot of
# the most recent user prompt with no timestamp or content blocks, purely
# a session bookmark. Explicit discard beats implicit fall-through.
_DISCARDABLE_TYPES = frozenset(
    {
        "file-history-snapshot",
        "permission-mode",
        "system",
        "custom-title",
        "agent-name",
        "queue-operation",
        "attachment",  # could become a CONTEXT event in a follow-up
        "last-prompt",  # CC >= 2.1.96 session bookmark; no trace-relevant info
    }
)


def _iso_to_unix(iso_str: str | None) -> float | None:
    """Convert an ISO 8601 timestamp (e.g. ``2026-04-03T08:33:47.356Z``) → Unix float.

    Returns ``None`` for ``None`` or unparseable input. Python 3.11+'s
    ``datetime.fromisoformat`` handles the ``Z`` suffix natively since
    3.11; the ``replace`` shim is a belt-and-braces fallback for older
    string shapes this codebase may encounter.
    """
    if not iso_str:
        return None
    try:
        normalized = iso_str.replace("Z", "+00:00") if iso_str.endswith("Z") else iso_str
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _harvest_session_metadata(session_path: Path) -> dict[str, Any]:
    """Walk the session file to extract the fields needed for the v5 header.

    Reads sequentially and returns the first non-empty values for: model,
    cwd, gitBranch, version, session uuid. Stops early once all fields
    are populated.
    """
    harvested: dict[str, Any] = {
        "model": None,
        "cwd": None,
        "git_branch": None,
        "cli_version": None,
        "session_id": None,
    }
    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if harvested["cwd"] is None and record.get("cwd"):
                harvested["cwd"] = record["cwd"]
            if harvested["git_branch"] is None and record.get("gitBranch"):
                harvested["git_branch"] = record["gitBranch"]
            if harvested["cli_version"] is None and record.get("version"):
                harvested["cli_version"] = record["version"]
            if harvested["session_id"] is None and record.get("sessionId"):
                harvested["session_id"] = record["sessionId"]
            if harvested["model"] is None and record.get("type") == "assistant":
                model = (record.get("message") or {}).get("model")
                if model:
                    harvested["model"] = model

            if all(v is not None for v in harvested.values()):
                break

    return harvested




def _split_assistant_content(
    content: list[dict[str, Any]] | str | None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Split an assistant message.content[] into (text, thinking, tool_uses).

    Claude Code encodes assistant output as a list of typed blocks
    (``text``, ``thinking``, ``tool_use``). We concatenate consecutive
    blocks of the same type so the Gantt tooltip shows a coherent
    preview, and collect tool_use blocks verbatim for later adaptation
    into OpenAI tool_calls shape.
    """
    if content is None or isinstance(content, str):
        text = content or ""
        return text, "", []

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            tool_uses.append(block)

    return "\n".join(text_parts), "\n".join(thinking_parts), tool_uses


def _build_openai_raw_response(
    *,
    model: str | None,
    message_id: str | None,
    text: str,
    tool_uses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Synthesize an OpenAI-shape chat-completion response from Anthropic content.

    The Gantt tooltip extractor (``_extract_detail_from_action`` in
    ``demo/gantt_viewer/backend/payload.py``) reads
    ``data.raw_response.choices[0].message`` expecting the OpenAI schema
    (``content`` + ``tool_calls`` array).
    We adapt Anthropic's native shape into that skeleton so the
    existing tooltip path renders Claude traces without any renderer
    changes.
    """
    openai_tool_calls = [
        {
            "id": tu.get("id", ""),
            "type": "function",
            "function": {
                "name": tu.get("name", ""),
                "arguments": json.dumps(tu.get("input") or {}, ensure_ascii=False),
            },
        }
        for tu in tool_uses
    ]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text or None,
    }
    if openai_tool_calls:
        message["tool_calls"] = openai_tool_calls

    return {
        "id": message_id or "",
        "model": model or "",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def _extract_tool_result_text(
    tool_result_content: list[dict[str, Any]] | str | None,
) -> str:
    """Flatten a tool_result block's content array into a single string."""
    if tool_result_content is None:
        return ""
    if isinstance(tool_result_content, str):
        return tool_result_content
    parts: list[str] = []
    for block in tool_result_content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        else:
            parts.append(str(block))
    return "\n".join(parts)




# Bash stderr preview cap. 2000 chars ≈ 25 80-col lines, fits in a Gantt
# tooltip without dominating it; matches the storage philosophy of
# _TOOL_RESULT_MAX_CHARS (8000 chars for full tool_result text) scaled
# down because stderr is usually a one-shot error blurb, not full output.
_BASH_STDERR_PREVIEW_MAX = 2000


def _backfill_per_tool(
    tool_name: str,
    result_dict: dict[str, Any],
) -> tuple[dict[str, Any], bool | None]:
    """Extract per-tool backfill keys from a ``toolUseResult`` dict.

    ``toolUseResult`` fields are **Task/Agent-specific**: baseline tools each
    carry their own per-tool dict shape with no overlap. This helper translates
    per-tool data into additive backfill keys without touching the universal
    fields the rest of the converter consumes.

    Verified shapes (CC CLI 2.1.96):

    - ``Bash``: ``{interrupted, isImage, noOutputExpected, stderr, stdout}``
      ``interrupted=true`` is the **only** reliable failure signal — the
      paired ``tool_result.is_error`` is typically ``False`` even when
      the bash command was killed. Without this override, interrupted
      bash calls render green on the Gantt timeline (silently wrong).
    - ``Read``: ``{file: {filePath, numLines, totalLines, ...}, type}``
    - ``Edit``: ``{filePath, newString, oldString, originalFile,
      replaceAll, structuredPatch, userModified}``. ``structuredPatch``
      is a pre-parsed diff (list of hunks) — far more useful in a
      tooltip than replaying ``newString``.

    Returns:
        ``(backfill, success_override)``. ``backfill`` is merged into
        the tool_exec ``data`` dict; ``success_override`` (when not
        ``None``) participates in the success-determination chain at a
        lower precedence than ``is_error`` from the tool_result block
        but higher than the legacy ``status`` field.
    """
    backfill: dict[str, Any] = {}
    success_override: bool | None = None

    if tool_name == "Bash":
        if result_dict.get("interrupted"):
            success_override = False
        stderr = result_dict.get("stderr")
        if isinstance(stderr, str) and stderr:
            backfill["stderr_preview"] = stderr[:_BASH_STDERR_PREVIEW_MAX]
    elif tool_name == "Edit":
        patch = result_dict.get("structuredPatch")
        if patch is not None:
            backfill["structured_patch"] = patch
        if result_dict.get("userModified"):
            backfill["user_modified"] = True
    elif tool_name == "Read":
        file_meta = result_dict.get("file")
        if isinstance(file_meta, dict):
            keep = {
                k: file_meta[k]
                for k in ("filePath", "numLines", "totalLines", "startLine")
                if k in file_meta
            }
            if keep:
                backfill["file_meta"] = keep

    return backfill, success_override





def _convert_session_records(
    *,
    session_path: Path,
    agent_id: str,
    session_start_ts: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream v5 records (actions + derived events) for one session file.

    The caller is responsible for emitting the trace_metadata header
    and the summary record — this generator produces only the ``action``
    records in the order they appear in the session. Iteration numbers
    are monotonically increased per call (one counter per ``agent_id``
    lane, since each invocation runs on exactly one lane).
    """
    pending_tool_uses: dict[str, dict[str, Any]] = {}
    iteration = 0
    last_lane_ts: float | None = session_start_ts

    n_tool_actions = 0
    n_llm_actions = 0
    total_llm_ms = 0.0
    total_tool_ms = 0.0
    total_tokens = 0
    first_ts: float | None = None
    last_ts: float | None = None

    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("skipping malformed JSONL line in %s", session_path)
                continue

            rtype = record.get("type")
            if rtype in _DISCARDABLE_TYPES:
                continue

            if rtype == "assistant":
                ts_end = _iso_to_unix(record.get("timestamp"))
                if ts_end is None:
                    logger.debug(
                        "assistant record missing timestamp in %s; skipping",
                        session_path,
                    )
                    continue
                ts_start = last_lane_ts if last_lane_ts is not None else ts_end
                last_lane_ts = ts_end
                if first_ts is None:
                    first_ts = ts_start
                last_ts = ts_end

                message = record.get("message") or {}
                content = message.get("content")
                text, thinking, tool_uses = _split_assistant_content(content)

                raw_response = _build_openai_raw_response(
                    model=message.get("model"),
                    message_id=message.get("id"),
                    text=text,
                    tool_uses=tool_uses,
                )

                usage = message.get("usage") or {}
                prompt_tokens = usage.get("input_tokens", 0) or 0
                completion_tokens = usage.get("output_tokens", 0) or 0
                cache_creation_obj = usage.get("cache_creation") or {}

                llm_latency_ms = max(0.0, (ts_end - ts_start) * 1000)
                total_llm_ms += llm_latency_ms
                total_tokens += prompt_tokens + completion_tokens

                llm_content_preview = text
                if len(llm_content_preview) > 1000:
                    llm_content_preview = (
                        llm_content_preview[:1000] + "\n[...truncated at 1000 chars]"
                    )

                data: dict[str, Any] = {
                    "raw_response": raw_response,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "llm_latency_ms": round(llm_latency_ms, 2),
                    "llm_content": llm_content_preview,
                    # Backfill: Claude-specific fields preserved under data.*
                    "message_id": message.get("id", ""),
                    # FIX-5: Anthropic request id (top-level on assistant
                    # records since CC CLI 2.1.96). Empty string when
                    # absent so the schema is stable across versions.
                    "request_id": record.get("requestId", "") or "",
                    "cache_read_tokens": usage.get("cache_read_input_tokens", 0) or 0,
                    "cache_creation_tokens": usage.get(
                        "cache_creation_input_tokens", 0
                    )
                    or 0,
                    "cache_ephemeral_5m_tokens": cache_creation_obj.get(
                        "ephemeral_5m_input_tokens", 0
                    )
                    or 0,
                    "cache_ephemeral_1h_tokens": cache_creation_obj.get(
                        "ephemeral_1h_input_tokens", 0
                    )
                    or 0,
                }
                if thinking:
                    data["thinking"] = thinking
                if usage.get("service_tier"):
                    data["service_tier"] = usage["service_tier"]

                yield {
                    "type": "action",
                    "action_type": "llm_call",
                    "action_id": f"llm_{iteration}",
                    "agent_id": agent_id,
                    "iteration": iteration,
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                    "data": data,
                }
                n_llm_actions += 1

                # Register pending tool_uses for later pairing. Tool execution
                # nominally begins right after the LLM call finishes.
                for tu in tool_uses:
                    tu_id = tu.get("id") or ""
                    if not tu_id:
                        continue
                    pending_tool_uses[tu_id] = {
                        "ts_start": ts_end,
                        "tool_name": tu.get("name", "unknown"),
                        "tool_args": json.dumps(
                            tu.get("input") or {}, ensure_ascii=False
                        ),
                        "iteration": iteration,
                    }

                iteration += 1

            elif rtype == "user":
                message = record.get("message") or {}
                content = message.get("content")
                user_ts = _iso_to_unix(record.get("timestamp"))

                # Advance the lane clock to the user record's timestamp
                # BEFORE branching on content shape. A plain user text
                # message (content is a string) still carries wall-clock
                # meaning: it's when the user submitted the request, so
                # the next assistant's llm_latency_ms should be measured
                # from here, not from some earlier assistant's ts_end.
                if user_ts is not None:
                    last_lane_ts = max(last_lane_ts or 0.0, user_ts)

                if not isinstance(content, list):
                    continue  # plain user text, no tool_result blocks to process
                # toolUseResult is usually a dict with the rich sidecar
                # fields (totalDurationMs, agentId, usage, ...) but some
                # CC tool types (e.g. simple file reads) stuff a plain
                # string in there. Defensive type-check so the converter
                # doesn't crash on those records; we just lose the
                # sidecar backfill for that single tool_result.
                tool_use_result_raw = record.get("toolUseResult")
                tool_use_result: dict[str, Any] = (
                    tool_use_result_raw
                    if isinstance(tool_use_result_raw, dict)
                    else {}
                )

                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue

                    tu_id = block.get("tool_use_id") or ""
                    pending = pending_tool_uses.pop(tu_id, None)
                    unpaired = pending is None

                    if pending is not None:
                        tool_ts_start = pending["ts_start"]
                        tool_iteration = pending["iteration"]
                        tool_name = pending["tool_name"]
                        tool_args = pending["tool_args"]
                    else:
                        tool_ts_start = user_ts if user_ts is not None else (
                            last_lane_ts or 0.0
                        )
                        tool_iteration = max(0, iteration - 1)
                        tool_name = "unknown_tool"
                        tool_args = "{}"

                    # Prefer toolUseResult.totalDurationMs (pre-computed by CC).
                    # Fall back to user_ts - tool_ts_start when the sidecar is absent.
                    sidecar_duration = tool_use_result.get("totalDurationMs")
                    if sidecar_duration is not None:
                        duration_ms = float(sidecar_duration)
                    elif user_ts is not None:
                        duration_ms = max(0.0, (user_ts - tool_ts_start) * 1000)
                    else:
                        duration_ms = 0.0
                    tool_ts_end = tool_ts_start + (duration_ms / 1000.0)
                    total_tool_ms += duration_ms

                    # Per-tool backfill (FIX-4): Bash.interrupted is the
                    # only reliable failure signal for interrupted bash
                    # commands; Edit.structuredPatch is a ready-to-render
                    # diff; Read.file carries file metadata. None of
                    # these live under the universal Task/Agent shape.
                    per_tool_backfill, success_override = _backfill_per_tool(
                        tool_name, tool_use_result
                    )

                    is_error = block.get("is_error")
                    if is_error is not None:
                        # is_error from the tool_result block is the
                        # most authoritative direct signal — it wins
                        # over per-tool overrides and the status field.
                        success = not bool(is_error)
                    elif success_override is not None:
                        success = success_override
                    elif tool_use_result.get("status"):
                        success = tool_use_result["status"] == "completed"
                    else:
                        success = True

                    result_text = _extract_tool_result_text(block.get("content"))
                    if len(result_text) > _TOOL_RESULT_MAX_CHARS:
                        result_text = (
                            result_text[:_TOOL_RESULT_MAX_CHARS]
                            + f"\n[...truncated at {_TOOL_RESULT_MAX_CHARS} chars]"
                        )

                    tool_data: dict[str, Any] = {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_result": result_text,
                        "duration_ms": round(duration_ms, 2),
                        "success": success,
                    }
                    if unpaired:
                        tool_data["note"] = (
                            f"unpaired tool_result: no matching tool_use for id "
                            f"{tu_id[:12]}..."
                        )
                    # Backfill from toolUseResult sidecar (Task/Agent shape)
                    if tool_use_result.get("agentId") or tool_use_result.get("agentType"):
                        tool_data["subagent_meta"] = {
                            "agent_id": tool_use_result.get("agentId"),
                            "agent_type": tool_use_result.get("agentType"),
                        }
                    if (
                        tool_use_result.get("totalTokens") is not None
                        or tool_use_result.get("usage") is not None
                    ):
                        tool_data["subagent_tokens"] = {
                            "total": tool_use_result.get("totalTokens"),
                            "usage": tool_use_result.get("usage"),
                        }
                    if tool_use_result.get("totalToolUseCount") is not None:
                        tool_data["subagent_tool_use_count"] = tool_use_result[
                            "totalToolUseCount"
                        ]
                    # Per-tool backfill (Bash/Edit/Read shapes) — additive only
                    tool_data.update(per_tool_backfill)

                    if first_ts is None:
                        first_ts = tool_ts_start
                    last_ts = tool_ts_end

                    # action_id must be unique within the trace — include
                    # a short tool_use_id suffix so parallel tool calls in
                    # the same iteration (e.g. two Read blocks in one
                    # assistant turn) don't collide on the same ID.
                    tu_id_suffix = (tu_id or "noid")[-8:]
                    yield {
                        "type": "action",
                        "action_type": "tool_exec",
                        "action_id": (
                            f"tool_{tool_iteration}_{tool_name}_{tu_id_suffix}"
                        ),
                        "agent_id": agent_id,
                        "iteration": tool_iteration,
                        "ts_start": tool_ts_start,
                        "ts_end": tool_ts_end,
                        "data": tool_data,
                    }
                    n_tool_actions += 1

                    last_lane_ts = max(last_lane_ts or 0.0, tool_ts_end)

    # Drain any unpaired tool_use blocks left in pending_tool_uses —
    # these are orphans: an assistant registered a tool_use but no
    # matching tool_result ever arrived in the file (crash, partial
    # write, or the session ended before the tool came back). Per
    # CLAUDE.md §5 "preserve all intermediate outputs", we emit a
    # zero-duration tool_exec for each orphan so the fact of the
    # invocation survives on the Gantt timeline with a descriptive note.
    for orphan_id, pending in pending_tool_uses.items():
        orphan_ts = pending["ts_start"]
        orphan_id_suffix = orphan_id[-8:] if orphan_id else "noid"
        yield {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": (
                f"tool_{pending['iteration']}_{pending['tool_name']}"
                f"_{orphan_id_suffix}_orphan"
            ),
            "agent_id": agent_id,
            "iteration": pending["iteration"],
            "ts_start": orphan_ts,
            "ts_end": orphan_ts,
            "data": {
                "tool_name": pending["tool_name"],
                "tool_args": pending["tool_args"],
                "tool_result": "",
                "duration_ms": 0.0,
                "success": False,
                "note": (
                    f"orphan tool_use: no tool_result received for id "
                    f"{orphan_id[:12]}... (partial write or interrupted session)"
                ),
            },
        }
        n_tool_actions += 1

    # Attach summary as a pseudo-attribute via a trailing yield. We use a
    # special sentinel dict the caller unpacks; keeps the generator's
    # single-responsibility for emission.
    yield {
        "__summary__": True,
        "agent_id": agent_id,
        "n_iterations": n_llm_actions,
        "n_tool_actions": n_tool_actions,
        "total_llm_ms": round(total_llm_ms, 2),
        "total_tool_ms": round(total_tool_ms, 2),
        "total_tokens": total_tokens,
        "elapsed_s": round((last_ts or 0.0) - (first_ts or 0.0), 3),
    }





def import_claude_code_session(
    *,
    session_path: Path,
    output_dir: Path,
    include_sidechains: bool = True,
    run_id: str | None = None,
) -> Path:
    """Convert a Claude Code session into a v5 JSONL for the Gantt viewer.

    Args:
        session_path: Path to the Claude Code session JSONL (typically
            ``~/.claude/projects/<slug>/<session-uuid>.jsonl``).
        output_dir: Where to write the converted v5 trace. The final
            file lands at
            ``<output_dir>/claude-code-import/<session-uuid>/<session-uuid>.jsonl``.
        include_sidechains: If True, walk ``<session-dir>/<session-uuid>/
            subagents/agent-*.jsonl`` and fold each subagent's records
            into the same output with a distinct ``agent_id``. Each
            subagent becomes its own Gantt lane.
        run_id: Optional run identifier for the output directory. Defaults
            to the session uuid derived from the filename.

    Returns:
        Path to the produced v5 JSONL trace.
    """
    session_path = Path(session_path).expanduser().resolve()
    if not session_path.exists():
        raise FileNotFoundError(f"Claude Code session not found: {session_path}")

    # Resolve the canonical session UUID. Native CC files live at
    # ``~/.claude/projects/<slug>/<uuid>.jsonl`` so the filename stem IS
    # the UUID. Collector-produced files land at ``.../attempt_N/trace.jsonl``
    # — the real UUID lives only inside the records (as ``sessionId``).
    # Trusting the stem unconditionally causes cross-task collisions in
    # the output directory (every ``trace.jsonl`` writes to
    # ``claude-code-import/trace/trace.jsonl`` and overwrites the last).
    # Harvest first so both the stem check and the fallback see the
    # same data; use the harvested sessionId only when the stem is not
    # UUID-shaped so existing native-layout callers are unaffected.
    harvested = _harvest_session_metadata(session_path)
    if _UUID_RE.match(session_path.stem):
        session_uuid = session_path.stem
    elif harvested["session_id"]:
        session_uuid = harvested["session_id"]
    else:
        session_uuid = session_path.stem

    run_dir_name = run_id or session_uuid
    run_dir = (
        Path(output_dir).expanduser().resolve()
        / "claude-code-import"
        / run_dir_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    target_path = run_dir / f"{session_uuid}.jsonl"

    # Build the v5 metadata header — mandatory trace_format_version=5
    # plus a metadata.run_config extension blob carrying the CC-specific
    # runtime context (cwd / git_branch / cli_version).
    run_config: dict[str, Any] = {}
    if harvested["cwd"]:
        run_config["cwd"] = harvested["cwd"]
    if harvested["git_branch"]:
        run_config["git_branch"] = harvested["git_branch"]
    if harvested["cli_version"]:
        run_config["cli_version"] = harvested["cli_version"]

    metadata: dict[str, Any] = {
        "type": "trace_metadata",
        "trace_format_version": V5_FORMAT_VERSION,
        "scaffold": SCAFFOLD_LABEL,
        "mode": "import",
        "instance_id": session_uuid,
        "source_trace": str(session_path),
    }
    if harvested["model"]:
        metadata["model"] = harvested["model"]
    if run_config:
        metadata["run_config"] = run_config

    # Stream the conversion. We collect actions first (so the summary
    # can be materialized from aggregates), then write the full JSONL
    # in one pass at the end. Memory footprint is bounded by the
    # session size — fine for CC sessions (up to tens of MB).
    all_actions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    main_agent_id = session_uuid
    for item in _convert_session_records(
        session_path=session_path, agent_id=main_agent_id
    ):
        if item.get("__summary__"):
            summaries.append(item)
        else:
            all_actions.append(item)

    # Sidechains: walk <session-dir>/<session-uuid>/subagents/agent-*.jsonl
    if include_sidechains:
        sidechain_dir = session_path.parent / session_uuid / "subagents"
        if sidechain_dir.exists() and sidechain_dir.is_dir():
            for agent_file in sorted(sidechain_dir.glob("agent-*.jsonl")):
                sub_agent_id = agent_file.stem  # e.g. "agent-a3db9c581cbafb09d"
                for item in _convert_session_records(
                    session_path=agent_file, agent_id=sub_agent_id
                ):
                    if item.get("__summary__"):
                        summaries.append(item)
                    else:
                        all_actions.append(item)

    # Backfill max_iterations = main-lane assistant count for the
    # metadata header so the Gantt header widget shows a useful value.
    main_summary = next(
        (s for s in summaries if s["agent_id"] == main_agent_id), None
    )
    if main_summary:
        metadata["max_iterations"] = main_summary["n_iterations"]

    # Write the v5 JSONL atomically: metadata header, sorted actions,
    # then one summary per agent lane.
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as dst:
        dst.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for action in all_actions:
            dst.write(json.dumps(action, ensure_ascii=False) + "\n")
        for summary_data in summaries:
            record = {
                "type": "summary",
                "agent_id": summary_data["agent_id"],
                "n_iterations": summary_data["n_iterations"],
                "n_tool_actions": summary_data["n_tool_actions"],
                "total_llm_ms": summary_data["total_llm_ms"],
                "total_tool_ms": summary_data["total_tool_ms"],
                "total_tokens": summary_data["total_tokens"],
                "elapsed_s": summary_data["elapsed_s"],
                "model": harvested["model"] or "",
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Imported Claude Code session %s → %s (%d actions, %d lanes)",
        session_uuid,
        target_path,
        len(all_actions),
        len(summaries),
    )
    return target_path
