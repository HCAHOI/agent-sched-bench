"""De-tokenize block_topk-selected blocks back to their source text.

block_position shows WHERE each selected block sits on the KV token axis; this
companion recovers WHAT TEXT those tokens are. The recordings persist neither the
token ids nor the rendered prompt string, but every llm_call in ``trace.jsonl``
carries its full ``messages_in`` list, and re-rendering it with the model's
tokenizer reproduces the exact prompt the recorder tokenized (same path as
``serving.recording.backend_hf.tokenize_chat_with_segments``). We then map each
selected block's token range -> char range (via the tokenizer offset mapping) ->
the literal text.

Alignment is self-checked against ``segments.json``: a call is only decoded when
the re-tokenized total length and every segment boundary match the recorded
values. Calls that fail the check are emitted with ``align_ok=false`` and no
text rather than risk silently mis-attributed text.

Tokenizer-only: loads ``AutoTokenizer`` (no model weights, no GPU, no torch).
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.recoding_figures.recording_loader import (  # noqa: E402
    IterationRecord,
    find_attempt_dirs,
    load_iteration_records,
)
from scripts.recoding_figures.plot_head_span_grid import (  # noqa: E402
    _parse_layer_arg,
    block_position_rows,
)
from scripts.recoding_figures.plot_sparse_segment_grid import (  # noqa: E402
    _json_ready,
    _load_json_required,
    _safe_name,
    _write_csv,
)

# Faithful copy of serving.recording.backend_hf._ALLOWED_MSG_KEYS. We replicate
# the recorder's message normalization here (rather than importing backend_hf,
# which would pull torch) so the re-rendered prompt is byte-identical to what was
# tokenized at record time — including tool_call argument coercion, which shifts
# assistant-turn tokens if skipped.
_ALLOWED_MSG_KEYS = frozenset(
    {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"}
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="attempt/task/run dirs (need recordings/ + trace.jsonl)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Tokenizer model id. Default: read recordings/meta.json model name.",
    )
    parser.add_argument("--layers", type=str, default=None, help="e.g. 0,9,18,27,38,47")
    parser.add_argument(
        "--tools-json",
        type=Path,
        default=None,
        help=(
            "JSON file with the OpenAI-format tool definitions openclaw passed "
            "via tools= (reproduced from agents.openclaw tools.get_definitions). "
            "Required for exact alignment when the recorded prompt rendered tool "
            "schemas into the system segment."
        ),
    )
    parser.add_argument("--include-orphans", action="store_true")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument(
        "--top-k", type=int, default=20, help="blocks per call in the markdown digest"
    )
    parser.add_argument(
        "--split-by-task", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    tools = json.loads(args.tools_json.read_text(encoding="utf-8")) if args.tools_json else None
    summary = build_detokenized_selected_blocks(
        inputs=args.inputs,
        output_dir=args.output_dir,
        model=args.model,
        layers=_parse_layer_arg(args.layers),
        include_orphans=args.include_orphans,
        max_iters=args.max_iters,
        top_k=args.top_k,
        split_by_task=args.split_by_task,
        tools=tools,
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


def build_detokenized_selected_blocks(
    *,
    inputs: Sequence[Path],
    output_dir: Path,
    model: str | None = None,
    layers: Sequence[int] | None = None,
    include_orphans: bool = False,
    max_iters: int | None = None,
    top_k: int = 20,
    split_by_task: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = load_iteration_records(
        inputs, include_orphans=include_orphans, max_iters=max_iters
    )
    attempt_dirs = find_attempt_dirs(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    if split_by_task:
        by_task: dict[str, list[IterationRecord]] = {}
        for record in records:
            by_task.setdefault(record.task, []).append(record)
        groups = [
            (task, task_records, output_dir / _safe_name(task))
            for task, task_records in sorted(by_task.items())
        ]
    else:
        groups = [("all_tasks", records, output_dir)]

    group_summaries = []
    for label, group_records, group_dir in groups:
        group_summaries.append(
            _detokenize_group(label, group_records, group_dir, model, layers, top_k, tools)
        )

    run_summary = {
        "inputs": [str(path) for path in inputs],
        "attempt_dirs": [str(path) for path in attempt_dirs],
        "artifact": "detokenized_block_topk_selected_blocks",
        "split_by_task": split_by_task,
        "groups": group_summaries,
    }
    (output_dir / "selected_blocks_detok_run_summary.json").write_text(
        json.dumps(_json_ready(run_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_summary


def _detokenize_group(
    label: str,
    records: Sequence[IterationRecord],
    group_dir: Path,
    model: str | None,
    layers: Sequence[int] | None,
    top_k: int,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    trajectory_rows, _layer_rows, layers_used, _meta = block_position_rows(
        records, layers=layers
    )
    if not trajectory_rows:
        raise ValueError(f"{label}: no block-position observations were found")

    model_id = model or _model_from_meta(records)
    tokenizer = _load_tokenizer(model_id)

    # block_position keys rows by observed_call_idx alone, so a group must be a
    # single attempt — otherwise two attempts' call_idx would collide and we
    # could match a call against the wrong attempt's trace. Fail loud.
    attempt_dirs = {r.attempt_dir for r in records}
    if len(attempt_dirs) != 1:
        raise ValueError(
            f"{records[0].task}: expected exactly one attempt dir per task group, "
            f"got {sorted(str(p) for p in attempt_dirs)}; "
            "block_position/detok assume a single attempt per task"
        )
    attempt_dir = records[0].attempt_dir

    iter_dir_by_call = {int(r.call_idx): r.iter_dir for r in records}
    rows_by_call: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        rows_by_call[int(row["observed_call_idx"])].append(row)

    llm_calls = _load_llm_calls(attempt_dir / "trace.jsonl")
    action_id_by_call = _call_action_ids(records[0].recordings_dir)

    out_rows: list[dict[str, Any]] = []
    call_status: list[dict[str, Any]] = []
    for call_idx in sorted(rows_by_call):
        iter_dir = iter_dir_by_call.get(call_idx)
        seg_payload = _load_json_required(iter_dir / "segments.json")
        input_tokens = int(seg_payload.get("input_tokens", 0))
        segments = list(seg_payload.get("segments", []))
        messages, match_kind = _match_messages(
            llm_calls, action_id_by_call, call_idx, input_tokens
        )

        full_text, offsets, render_err = _render_and_offsets(tokenizer, messages, tools)
        align_ok = False
        n_mismatch = -1
        n_tok = len(offsets)
        if render_err is None:
            align_ok, n_mismatch = _alignment_ok(offsets, segments, input_tokens)

        n_decoded = 0
        for row in rows_by_call[call_idx]:
            char_start = char_end = None
            decoded_text = None
            decoded_partial = False
            if align_ok:
                ts = int(row["token_start"])
                te = min(int(row["token_end"]), n_tok)
                if 0 <= ts < n_tok and te > ts:
                    char_start = int(offsets[ts][0])
                    char_end = int(offsets[te - 1][1])
                    decoded_text = full_text[char_start:char_end]
                    # block straddles the prompt/generation frontier: only the
                    # in-prompt tokens are decodable here (generated tokens are
                    # the model output, not in the prompt offset grid).
                    decoded_partial = te < int(row["token_end"])
                    n_decoded += 1
            out_rows.append(
                {
                    **{k: row.get(k) for k in (
                        "task", "observed_call_idx", "block_id", "token_start",
                        "token_end", "seg_lo", "seg_hi", "straddles", "segment_id",
                        "role", "tool_name", "selection_freq", "n_hits",
                        "n_valid_layer_steps", "mean_attn", "n_contributors",
                    )},
                    "char_start": char_start,
                    "char_end": char_end,
                    "align_ok": align_ok,
                    "decoded_partial": decoded_partial,
                    "decoded_text": decoded_text,
                }
            )
        call_status.append(
            {
                "call_idx": call_idx,
                "input_tokens": input_tokens,
                "retokenized_tokens": n_tok,
                "align_ok": align_ok,
                "segment_boundary_mismatches": n_mismatch,
                "match_kind": match_kind,
                "render_error": render_err,
                "n_selected_blocks": len(rows_by_call[call_idx]),
                "n_decoded_blocks": n_decoded,
            }
        )

    group_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(group_dir / "selected_blocks_detok.csv", out_rows)
    (group_dir / "selected_blocks_top_text.md").write_text(
        _digest_markdown(label, model_id, out_rows, call_status, top_k),
        encoding="utf-8",
    )
    n_calls_ok = sum(1 for s in call_status if s["align_ok"])
    group_summary = {
        "label": label,
        "model": model_id,
        "layers_used": layers_used,
        "n_records": len(records),
        "n_calls": len(call_status),
        "n_calls_aligned": n_calls_ok,
        "n_selected_block_rows": len(out_rows),
        "n_decoded_rows": sum(1 for r in out_rows if r["decoded_text"] is not None),
        "n_llm_calls_in_trace": len(llm_calls),
        "calls": call_status,
    }
    (group_dir / "selected_blocks_detok_summary.json").write_text(
        json.dumps(_json_ready(group_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {k: v for k, v in group_summary.items() if k != "calls"} | {
        "output_dir": str(group_dir),
    }


def _model_from_meta(records: Sequence[IterationRecord]) -> str:
    for record in records:
        meta_path = record.recordings_dir / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model = meta.get("model")
            name = model.get("name") if isinstance(model, dict) else model
            if name:
                return str(name)
    raise ValueError("could not read model name from recordings/meta.json; pass --model")


def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        # FP8 / quantized repos occasionally omit tokenizer files; fall back to
        # the base instruct repo (identical tokenizer).
        base = model_id.replace("-FP8", "").replace("-fp8", "")
        if base == model_id:
            raise
        tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    # Offset mapping (the token->char map this tool depends on) is fast-only.
    if not getattr(tok, "is_fast", False):
        raise ValueError(
            f"{model_id}: a fast tokenizer is required for return_offsets_mapping"
        )
    return tok


def _load_llm_calls(trace_path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not trace_path.is_file():
        return calls
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "action" and record.get("action_type") == "llm_call":
                data = record.get("data") or {}
                calls.append(
                    {
                        "action_id": record.get("action_id"),
                        "messages_in": data.get("messages_in"),
                        "prompt_tokens": data.get("prompt_tokens"),
                    }
                )
    return calls


def _call_action_ids(recordings_dir: Path) -> dict[int, str]:
    """Map recording call_idx -> trace action_id from recordings/meta.json.

    This is the authoritative recording<->trace link (meta iters carry
    ``trace_action_id`` = the trace action's ``action_id``), so it disambiguates
    calls that share a prompt-token count (e.g. malformed-retry re-issues).
    """
    out: dict[int, str] = {}
    meta_path = recordings_dir / "meta.json"
    if not meta_path.is_file():
        return out
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for item in meta.get("iters", []):
        call_idx = item.get("call_idx")
        action_id = item.get("trace_action_id")
        if call_idx is not None and action_id is not None:
            out[int(call_idx)] = str(action_id)
    return out


def _match_messages(
    llm_calls: Sequence[dict[str, Any]],
    action_id_by_call: dict[int, str],
    call_idx: int,
    input_tokens: int,
) -> tuple[list[dict[str, Any]] | None, str]:
    """Map a recording call_idx to its trace messages_in.

    Authoritative path: recording call_idx -> meta trace_action_id -> the trace
    action with that action_id (unique, retry-safe). Fallbacks (older recordings
    without trace_action_id): positional when prompt_tokens agrees, then a UNIQUE
    prompt_tokens match; an ambiguous prompt_tokens collision refuses (returns
    None) so the gate marks the call unaligned rather than risk wrong text.
    """
    action_id = action_id_by_call.get(call_idx)
    if action_id is not None:
        for cand in llm_calls:
            if cand.get("action_id") == action_id and cand.get("messages_in"):
                return cand["messages_in"], "trace_action_id"
    if 0 <= call_idx < len(llm_calls):
        cand = llm_calls[call_idx]
        if cand.get("messages_in") and cand.get("prompt_tokens") == input_tokens:
            return cand["messages_in"], "positional"
    by_tokens = [
        cand
        for cand in llm_calls
        if cand.get("messages_in") and cand.get("prompt_tokens") == input_tokens
    ]
    if len(by_tokens) == 1:
        return by_tokens[0]["messages_in"], "by_prompt_tokens"
    if len(by_tokens) > 1:
        return None, "ambiguous_prompt_tokens"
    if 0 <= call_idx < len(llm_calls):
        return llm_calls[call_idx].get("messages_in"), "positional_fallback"
    return None, "no_match"


def _loads_lenient(value: str) -> Any:
    """json_repair.loads — same coercion the recorder uses (backend_hf:167).

    json_repair (``json-repair>=0.30``) is a hard project dependency; we require
    it rather than falling back to ``json.loads`` because a different coercion on
    malformed tool-call arguments would silently shift assistant-turn tokens. A
    missing dependency raises, which the caller turns into a render error and an
    ``align_ok=False`` (no text) — never silently divergent text.
    """
    import json_repair

    return json_repair.loads(value)


def _normalize_tool_arguments(value: Any) -> dict[str, Any]:
    """Copy of backend_hf._normalize_tool_arguments: coerce args to a dict."""
    if isinstance(value, str):
        value = _loads_lenient(value)
    if not isinstance(value, dict):
        return {}
    return value


def _sanitize_empty_content(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of LLMProvider._sanitize_empty_content (providers/base.py)."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and not content:
            clean = dict(msg)
            clean["content"] = (
                None
                if (msg.get("role") == "assistant" and msg.get("tool_calls"))
                else "(empty)"
            )
            result.append(clean)
            continue
        if isinstance(content, list):
            new_items: list[Any] = []
            changed = False
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") in ("text", "input_text", "output_text")
                    and not item.get("text")
                ):
                    changed = True
                    continue
                if isinstance(item, dict) and "_meta" in item:
                    new_items.append({k: v for k, v in item.items() if k != "_meta"})
                    changed = True
                else:
                    new_items.append(item)
            if changed:
                clean = dict(msg)
                if new_items:
                    clean["content"] = new_items
                elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                    clean["content"] = None
                else:
                    clean["content"] = "(empty)"
                result.append(clean)
                continue
        if isinstance(content, dict):
            clean = dict(msg)
            clean["content"] = [content]
            result.append(clean)
            continue
        result.append(dict(msg))
    return result


def _normalize_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Faithful copy of backend_hf._normalize_messages (preserve ids off).

    Empty-content sanitize -> allowed-key restriction (+assistant content=None)
    -> tool_call argument coercion. Replicated rather than imported because
    backend_hf pulls torch.
    """
    sanitized = []
    for msg in _sanitize_empty_content(messages):
        clean = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
        if clean.get("role") == "assistant" and "content" not in clean:
            clean["content"] = None
        sanitized.append(clean)
    normalized: list[dict[str, Any]] = []
    for message in sanitized:
        copied = dict(message)
        tool_calls = []
        for tool_call in copied.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            tc = dict(tool_call)
            function = dict(tc.get("function") or {})
            function["arguments"] = _normalize_tool_arguments(
                function.get("arguments", tc.get("arguments", {}))
            )
            tc["function"] = function
            tool_calls.append(tc)
        if tool_calls:
            copied["tool_calls"] = tool_calls
        normalized.append(copied)
    return normalized


def _render_and_offsets(
    tokenizer,
    messages: Sequence[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
) -> tuple[str, list[tuple[int, int]], str | None]:
    """Reproduce the record-time rendered prompt + per-token char offsets.

    Mirrors backend_hf._apply_chat_template (passes the same ``tools=`` OpenClaw
    sent, which the Qwen template renders into the system segment) and
    _tokenize_with_offsets (add_special_tokens=False on the rendered string).
    Returns (full_text, offsets, error).
    """
    if not messages:
        return "", [], "no_messages"
    try:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        full_text = tokenizer.apply_chat_template(_normalize_messages(messages), **kwargs)
        encoded = tokenizer(
            full_text, add_special_tokens=False, return_offsets_mapping=True
        )
        offsets = [tuple(int(x) for x in pair) for pair in encoded["offset_mapping"]]
        return full_text, offsets, None
    except Exception as exc:  # noqa: BLE001 - record and gate, never emit bad text
        return "", [], f"{type(exc).__name__}: {exc}"


def _alignment_ok(
    offsets: Sequence[tuple[int, int]],
    segments: Sequence[dict[str, Any]],
    input_tokens: int,
) -> tuple[bool, int]:
    """True iff re-tokenization matches the recorded grid at every boundary.

    Checks total length == input_tokens AND that every prompt segment's recorded
    token_start/token_end is reproduced from its char_start/char_end via the
    offset mapping (same binary-search rule as backend_hf._token_boundary_for_char).
    Because segment boundaries are cumulative, equal total length plus every
    boundary landing on its recorded token index leaves no room for interior
    drift within a segment (a different interior split would shift a later
    boundary and fail the check). This is the bit-exact gate that licenses
    emitting text; on any mismatch the caller emits no text for the call.
    """
    if len(offsets) != input_tokens:
        return False, -1
    starts = [int(o[0]) for o in offsets]
    mismatches = 0
    for seg in segments:
        # Only prompt segments carry char offsets; the trailing generation
        # (output) segment has token positions beyond input_tokens and no
        # char span, so it is not part of the prompt offset grid.
        if seg.get("char_start") is None or seg.get("char_end") is None:
            continue
        if _token_index_for_char(starts, int(seg["char_start"])) != int(seg["token_start"]):
            mismatches += 1
        if _token_index_for_char(starts, int(seg["char_end"])) != int(seg["token_end"]):
            mismatches += 1
    return mismatches == 0, mismatches


def _token_index_for_char(starts: Sequence[int], char: int) -> int:
    # Equivalent to backend_hf._token_boundary_for_char (first token whose start
    # >= char) because left-to-right tokenization yields non-decreasing starts.
    return bisect.bisect_left(starts, char)


def _digest_markdown(
    label: str,
    model_id: str,
    out_rows: Sequence[dict[str, Any]],
    call_status: Sequence[dict[str, Any]],
    top_k: int,
) -> str:
    by_call: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in out_rows:
        by_call[int(row["observed_call_idx"])].append(row)
    status_by_call = {int(s["call_idx"]): s for s in call_status}

    lines = [
        f"# Selected blocks -> source text: `{label}`",
        "",
        f"- Model / tokenizer: `{model_id}`",
        f"- Calls aligned: `{sum(1 for s in call_status if s['align_ok'])}/{len(call_status)}` "
        "(only aligned calls show text; others failed the segment-boundary self-check)",
        f"- Top {top_k} most-selected blocks per call (by selection_freq).",
        "",
    ]
    for call_idx in sorted(by_call):
        status = status_by_call.get(call_idx, {})
        rows = [r for r in by_call[call_idx] if r["decoded_text"] is not None]
        rows.sort(key=lambda r: -(r.get("selection_freq") or 0.0))
        lines.append(
            f"## call {call_idx}  (align_ok={str(status.get('align_ok')).lower()}, "
            f"input_tokens={status.get('input_tokens')}, "
            f"selected_blocks={status.get('n_selected_blocks')})"
        )
        if not rows:
            lines.append("")
            lines.append("_no aligned/decoded blocks for this call_")
            lines.append("")
            continue
        lines.append("")
        lines.append("| freq | role | tokens | text |")
        lines.append("|---|---|---|---|")
        for row in rows[:top_k]:
            freq = row.get("selection_freq") or 0.0
            role = row.get("role") or "?"
            tok = f"{row['token_start']}-{row['token_end']}"
            text = _md_cell(row["decoded_text"])
            lines.append(f"| {freq:.3f} | {role} | {tok} | {text} |")
        lines.append("")
    return "\n".join(lines)


def _md_cell(text: str, *, limit: int = 300) -> str:
    snippet = text if len(text) <= limit else text[:limit] + "…"
    return snippet.replace("\\", "\\\\").replace("\n", "⏎").replace("|", "\\|").replace("\r", "")


if __name__ == "__main__":
    main()
