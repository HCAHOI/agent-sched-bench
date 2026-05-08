"""Attention-map probe with message-segmented annotations.

For specified layers, captures HEAD-AVERAGED attention maps per LLM call in a
trace, downsamples to a manageable resolution for storage, and tracks per-
message token boundaries so the plotter can mark segments (system / user /
assistant_message / assistant_call / tool_result / meta / gen_prompt) on the
heatmap.

Memory: requires `attn_implementation="eager"` because SDPA/FlashAttn don't
return the [seq, seq] score matrix. At seq=4096 with Qwen2.5-1.5B (28 layers,
14 heads), peak GPU memory during forward is ~16 GB (3 GB weights + 13 GB
attentions). Reduce --max-input-tokens for smaller models / GPUs.

Per (layer, iter) we save:
  - downsampled attention map [D, D] (D = --downsample, default 256)
  - segment token offsets (start, end, role) in ORIGINAL token coords
  - segment-to-segment aggregate matrix [n_segments, n_segments] = average
    attention mass from queries in segment_i to all keys in segment_j (using
    FULL resolution before downsampling)

Output: JSON with shape:
  {
    "model": str, "downsample": int, "target_layers": [int, ...],
    "iterations": [
      {
        "iter": int, "input_tokens": int, "forward_ms": float,
        "segments": [{"start": int, "end": int, "role": str}, ...],
        "layers": [
          {
            "layer": int,
            "attn_map_downsampled": [[float, ...], ...],  # [D, D]
            "seg_to_seg": [[float, ...], ...],            # [n_seg, n_seg]
          }, ...
        ]
      }, ...
    ]
  }
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

_MARKER_PREFIX = "<<<<AGENT_SCHED_BENCH_SEGMENT_"


def _flatten_to_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) for c in content)
        clean: dict[str, Any] = {"role": role, "content": str(content)}
        for key in ("tool_calls", "tool_call_id", "name", "reasoning_content"):
            if key in m:
                clean[key] = m[key]
        if role not in ("system", "user", "assistant", "tool"):
            clean = {"role": "assistant", "content": f"[{role}] {content}"}
        out.append(clean)
    return out


def build_chat_with_segments(
    tok, messages: list[dict[str, Any]], max_tokens: int
) -> tuple[torch.Tensor, list[dict[str, Any]]] | tuple[None, None]:
    """Tokenize chat-templated messages, returning input_ids and per-message
    segment offsets (start, end, role) in token coordinates.

    If total tokens > max_tokens, return (None, None) so caller can skip.
    """
    if _is_qwen_tool_template(tok):
        qwen_messages = _normalize_qwen_tool_arguments(messages)
        final_text, char_segments = _render_qwen_chat_with_segments(qwen_messages)
        template_text = tok.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        if final_text != template_text:
            raise ValueError("Qwen probe renderer did not match tokenizer chat template")
    elif _is_glm_tool_template(tok):
        glm_messages = _normalize_glm_tool_arguments(messages)
        final_text, char_segments = _render_glm_chat_with_segments(glm_messages)
        template_text = tok.apply_chat_template(
            glm_messages, tokenize=False, add_generation_prompt=True
        )
        if final_text != template_text:
            raise ValueError("GLM probe renderer did not match tokenizer chat template")
    else:
        final_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        marked_messages, markers = _mark_message_content(messages)
        marked_text = tok.apply_chat_template(
            marked_messages, tokenize=False, add_generation_prompt=True
        )
        stripped_text, marker_positions = _strip_markers(marked_text, markers)
        if stripped_text != final_text:
            raise ValueError("marked chat template did not round-trip to final prompt")
        if not messages:
            char_segments = []
        else:
            content_ends = [
                marker_positions[_end_marker(i)] for i in range(len(messages))
            ]
            char_boundaries = [0, *content_ends]
            roles = [_segment_role_for_message(m) for m in messages]
            if content_ends[-1] < len(final_text):
                char_boundaries.append(len(final_text))
                roles.append("gen_prompt")
            char_segments = [
                {"start": char_boundaries[i], "end": char_boundaries[i + 1], "role": role}
                for i, role in enumerate(roles)
            ]

    final, offsets = _tokenize_with_offsets(tok, final_text)
    n_tokens = final.input_ids.shape[1]
    if n_tokens > max_tokens:
        return None, None

    segments: list[dict[str, Any]] = []
    for segment in char_segments:
        start = _token_boundary_for_char(tok, final_text, offsets, segment["start"])
        end = _token_boundary_for_char(tok, final_text, offsets, segment["end"])
        if start >= end:
            continue
        segments.append(
            {
                "start": start,
                "end": end,
                "role": segment["role"],
            }
        )
    return final.input_ids, segments


def _is_qwen_tool_template(tok) -> bool:
    template = getattr(tok, "chat_template", "") or ""
    return "<tool_call>" in template and "<|im_start|>" in template


def _is_glm_tool_template(tok) -> bool:
    template = getattr(tok, "chat_template", "") or ""
    return (
        "[gMASK]<sop>" in template
        and "<|assistant|>" in template
        and "<tool_call>" in template
        and "<arg_key>" in template
    )


def _segment_role_for_message(message: dict[str, Any]) -> str:
    role = message.get("role", "user")
    if role == "assistant":
        return "assistant_message"
    if role == "tool":
        return "tool_result"
    return str(role)


def _append_segment(
    parts: list[str], segments: list[dict[str, Any]], role: str, text: str
) -> None:
    if not text:
        return
    start = sum(len(part) for part in parts)
    parts.append(text)
    segments.append({"start": start, "end": start + len(text), "role": role})


def _render_qwen_chat_with_segments(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Render Qwen chat-template text while preserving fine-grained boundaries."""
    parts: list[str] = []
    segments: list[dict[str, Any]] = []

    def append_segment(role: str, text: str) -> None:
        _append_segment(parts, segments, role, text)

    if messages and messages[0].get("role") == "system":
        append_segment("meta", "<|im_start|>system\n")
        append_segment("system", _message_content(messages[0]))
        append_segment("meta", "<|im_end|>\n")
        start_index = 1
    else:
        append_segment("meta", "<|im_start|>system\n")
        append_segment(
            "system",
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        )
        append_segment("meta", "<|im_end|>\n")
        start_index = 0

    i = start_index
    while i < len(messages):
        message = messages[i]
        role = message.get("role", "user")
        if role == "tool":
            tool_run: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                tool_run.append(messages[i])
                i += 1
            for offset, tool_message in enumerate(tool_run):
                if offset == 0:
                    append_segment("meta", "<|im_start|>user")
                append_segment(
                    "tool_result",
                    f"\n<tool_response>\n{_message_content(tool_message)}\n</tool_response>",
                )
                if offset == len(tool_run) - 1:
                    append_segment("meta", "<|im_end|>\n")
            continue

        if role == "assistant" and message.get("tool_calls"):
            append_segment("meta", "<|im_start|>assistant")
            content = _message_content(message)
            if content:
                append_segment("assistant_message", f"\n{content}")
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", tool_call)
                append_segment(
                    "assistant_call",
                    '\n<tool_call>\n{"name": "'
                    + str(function.get("name", ""))
                    + '", "arguments": '
                    + json.dumps(function.get("arguments"), ensure_ascii=False)
                    + "}\n</tool_call>"
                )
            append_segment("meta", "<|im_end|>\n")
        elif role == "assistant":
            append_segment("meta", "<|im_start|>assistant\n")
            append_segment("assistant_message", _message_content(message))
            append_segment("meta", "<|im_end|>\n")
        else:
            append_segment("meta", f"<|im_start|>{role}\n")
            append_segment(role, _message_content(message))
            append_segment("meta", "<|im_end|>\n")
        i += 1

    append_segment("gen_prompt", "<|im_start|>assistant\n")
    return "".join(parts), segments


def _render_glm_chat_with_segments(
    messages: list[dict[str, Any]],
    enable_thinking: bool | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Render GLM-4.5 chat-template text with fine-grained segment roles."""
    parts: list[str] = []
    segments: list[dict[str, Any]] = []

    def append_segment(role: str, text: str) -> None:
        _append_segment(parts, segments, role, text)

    append_segment("meta", "[gMASK]<sop>")
    last_user_index = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user_index = index

    for index, message in enumerate(messages):
        role = message.get("role", "user")
        if role == "user":
            content = _message_content(message)
            append_segment("meta", "<|user|>\n")
            append_segment("user", content)
            if (
                enable_thinking is False
                and not _visible_text(content).endswith("/nothink")
            ):
                append_segment("meta", "/nothink")
        elif role == "assistant":
            append_segment("meta", "<|assistant|>")
            reasoning_content, content = _glm_reasoning_and_content(message)
            if index > last_user_index and reasoning_content:
                append_segment(
                    "assistant_message",
                    "\n<think>" + reasoning_content.strip() + "</think>",
                )
            else:
                append_segment("assistant_message", "\n<think></think>")
            if content.strip():
                append_segment("assistant_message", "\n" + content.strip())
            for tool_call in message.get("tool_calls") or []:
                append_segment("assistant_call", _format_glm_tool_call(tool_call))
        elif role == "tool":
            if index == 0 or messages[index - 1].get("role") != "tool":
                append_segment("meta", "<|observation|>")
            content = message.get("content", "")
            if isinstance(content, str):
                append_segment(
                    "tool_result",
                    "\n<tool_response>\n" + content + "\n</tool_response>",
                )
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "output" in item:
                        rendered = str(item["output"])
                    else:
                        rendered = str(item)
                    append_segment(
                        "tool_result",
                        "\n<tool_response>\n" + rendered + "\n</tool_response>",
                    )
            else:
                append_segment(
                    "tool_result",
                    "\n<tool_response>\n" + str(content) + "\n</tool_response>",
                )
        elif role == "system":
            append_segment("meta", "<|system|>\n")
            append_segment("system", _message_content(message))
        else:
            append_segment("meta", f"<|{role}|>\n")
            append_segment(role, _message_content(message))

    gen_prompt = "<|assistant|>"
    if enable_thinking is False:
        gen_prompt += "\n<think></think>"
    append_segment("gen_prompt", gen_prompt)
    return "".join(parts), segments


def _visible_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        visible: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                visible.append(str(item.get("text", "")))
            elif isinstance(item, str):
                visible.append(item)
        return "".join(visible)
    return str(content)


def _glm_reasoning_and_content(message: dict[str, Any]) -> tuple[str, str]:
    content = _visible_text(message.get("content", ""))
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        return reasoning, content
    if "</think>" in content:
        before, after = content.split("</think>", maxsplit=1)
        reasoning = before.rstrip("\n").split("<think>")[-1].lstrip("\n")
        return reasoning, after.lstrip("\n")
    return "", content


def _format_glm_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function", tool_call)
    text = "\n<tool_call>" + str(function.get("name", ""))
    arguments = _parse_tool_arguments_mapping(function.get("arguments", {}), "GLM")
    for key, value in arguments.items():
        text += (
            f"\n<arg_key>{key}</arg_key>"
            f"\n<arg_value>{_glm_arg_value(value)}</arg_value>"
        )
    text += "\n</tool_call>"
    return text


def _glm_arg_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _normalize_qwen_tool_arguments(
    messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not message.get("tool_calls"):
            normalized.append(message)
            continue
        copied = dict(message)
        copied["tool_calls"] = [
            _normalize_qwen_tool_call(tool_call)
            for tool_call in message.get("tool_calls") or []
        ]
        normalized.append(copied)
    return normalized


def _normalize_qwen_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    copied = dict(tool_call)
    function = copied.get("function")
    if isinstance(function, dict):
        function = dict(function)
        function["arguments"] = _parse_tool_arguments_mapping(
            function.get("arguments", {}), "Qwen"
        )
        copied["function"] = function
    else:
        copied["arguments"] = _parse_tool_arguments_mapping(
            copied.get("arguments", {}), "Qwen"
        )
    return copied


def _normalize_glm_tool_arguments(
    messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not message.get("tool_calls"):
            normalized.append(message)
            continue
        copied = dict(message)
        copied["tool_calls"] = [
            _normalize_glm_tool_call(tool_call)
            for tool_call in message.get("tool_calls") or []
        ]
        normalized.append(copied)
    return normalized


def _normalize_glm_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    copied = dict(tool_call)
    function = copied.get("function")
    if isinstance(function, dict):
        function = dict(function)
        function["arguments"] = _parse_tool_arguments_mapping(
            function.get("arguments", {}), "GLM"
        )
        copied["function"] = function
    else:
        copied["arguments"] = _parse_tool_arguments_mapping(
            copied.get("arguments", {}), "GLM"
        )
    return copied


def _parse_tool_arguments_mapping(arguments: Any, family: str) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{family} tool_call arguments must be a JSON object") from exc
        arguments = parsed
    if not isinstance(arguments, dict):
        raise ValueError(f"{family} tool_call arguments must be a mapping")
    return arguments


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item.get("text", item)) for item in content)
    if content is None:
        return ""
    return str(content)


def _start_marker(index: int) -> str:
    return f"{_MARKER_PREFIX}{index:04d}_START>>>>"


def _end_marker(index: int) -> str:
    return f"{_MARKER_PREFIX}{index:04d}_END>>>>"


def _mark_message_content(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    markers: list[str] = []
    marked: list[dict[str, str]] = []
    for i, message in enumerate(messages):
        content = message["content"]
        if _MARKER_PREFIX in content:
            raise ValueError("message content contains reserved probe marker prefix")
        start = _start_marker(i)
        end = _end_marker(i)
        markers.extend([start, end])
        marked.append(
            {
                "role": message["role"],
                "content": f"{start}{content}{end}",
            }
        )
    return marked, markers


def _strip_markers(
    text: str, markers: list[str]
) -> tuple[str, dict[str, int]]:
    positions: dict[str, int] = {}
    stripped: list[str] = []
    i = 0
    while i < len(text):
        matched = next((marker for marker in markers if text.startswith(marker, i)), None)
        if matched is not None:
            if matched in positions:
                raise ValueError(f"duplicate marker in rendered prompt: {matched}")
            positions[matched] = len(stripped)
            i += len(matched)
            continue
        stripped.append(text[i])
        i += 1

    missing = [marker for marker in markers if marker not in positions]
    if missing:
        raise ValueError(f"markers missing from rendered prompt: {missing[:3]}")
    return "".join(stripped), positions


def _token_boundary_for_char(
    tok: Any, text: str, offsets: list[list[int]] | None, char_pos: int
) -> int:
    """Return first token whose start offset is at or after char_pos."""
    if char_pos <= 0:
        return 0
    if offsets is None:
        return _token_count(tok(text[:char_pos], add_special_tokens=False))
    for i, (start, _end) in enumerate(offsets):
        if start >= char_pos:
            return i
    return len(offsets)


def _tokenize_with_offsets(
    tok: Any, text: str
) -> tuple[Any, list[list[int]] | None]:
    try:
        final = tok(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        final = tok(text, return_tensors="pt", add_special_tokens=False)
        return final, None
    offsets = final.pop("offset_mapping")[0].tolist()
    return final, offsets


def _token_count(encoded: Any) -> int:
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.numel()) if input_ids.ndim == 1 else int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def downsample_attn(attn_2d: torch.Tensor, D: int) -> torch.Tensor:
    """Downsample [seq, seq] map to [D, D] via adaptive avg pooling.
    Uses 4D adaptive_avg_pool2d under the hood."""
    x = attn_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    return F.adaptive_avg_pool2d(x, (D, D)).squeeze(0).squeeze(0)


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    bad = int((~torch.isfinite(tensor)).sum().item())
    if bad:
        raise FloatingPointError(f"{name} contains {bad} non-finite values")


def segment_aggregate(
    attn_2d: torch.Tensor, segments: list[dict[str, Any]]
) -> list[list[float]]:
    """Average per-query attention mass from segment_i to segment_j.

    Each cell is block.sum(dim=-1).mean(): for queries in segment_i, the mean
    total probability assigned to all keys in segment_j.
    """
    n_seg = len(segments)
    out = [[0.0] * n_seg for _ in range(n_seg)]
    for i in range(n_seg):
        qs, qe = segments[i]["start"], segments[i]["end"]
        if qs >= qe:
            continue
        q_block = attn_2d[qs:qe]  # [q_count, seq]
        for j in range(n_seg):
            ks, ke = segments[j]["start"], segments[j]["end"]
            if ks >= ke:
                continue
            block = q_block[:, ks:ke]
            out[i][j] = float(block.sum(dim=-1).mean().item())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--layers", type=str, default="1,4,8",
                   help="Comma-separated layer indices to capture")
    p.add_argument("--downsample", type=int, default=256,
                   help="Downsample target resolution for attention maps")
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()
    if args.downsample <= 0:
        raise SystemExit("--downsample must be positive")

    target_layers = sorted(int(x) for x in args.layers.split(","))
    print(f"target layers: {target_layers}", file=sys.stderr)

    print(f"loading {args.model} (eager attn)", file=sys.stderr)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    n_layers_total = model.config.num_hidden_layers
    bad = [li for li in target_layers if li < 0 or li >= n_layers_total]
    if bad:
        raise SystemExit(f"layer indices out of range (model has {n_layers_total}): {bad}")

    iterations_out: list[dict[str, Any]] = []
    n_done = 0
    n_skipped = 0
    with open(args.trace, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "action" or r.get("action_type") != "llm_call":
                continue
            if n_done >= args.max_iters:
                break
            messages = r["data"].get("messages_in") or []
            if not messages:
                continue

            chat = _flatten_to_chat(messages)
            try:
                input_ids, segments = build_chat_with_segments(tok, chat, args.max_input_tokens)
            except Exception as exc:
                print(f"iter {r.get('iteration')}: chat-template error: {exc}", file=sys.stderr)
                continue
            if input_ids is None:
                n_skipped += 1
                print(f"iter {r.get('iteration')}: skipping (>{args.max_input_tokens} tokens)", file=sys.stderr)
                continue

            input_ids = input_ids.to("cuda:0")
            n_tokens = input_ids.shape[1]

            t0 = time.time()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, output_attentions=True, use_cache=False)
            torch.cuda.synchronize()
            forward_ms = (time.time() - t0) * 1000

            # outputs.attentions: tuple of [batch, heads, seq, seq] — one per layer
            layer_records: list[dict[str, Any]] = []
            for li in target_layers:
                attn = outputs.attentions[li]  # [1, heads, seq, seq]
                head_avg = attn[0].mean(dim=0).to(torch.float32)  # [seq, seq]
                require_finite_tensor("head-averaged attention", head_avg)
                # FULL-resolution segment aggregation BEFORE downsample
                seg_to_seg = segment_aggregate(head_avg, segments)
                # Downsample for the heatmap visualization
                downsampled = downsample_attn(head_avg, args.downsample).cpu()
                require_finite_tensor("downsampled attention map", downsampled)
                layer_records.append({
                    "layer": li,
                    "attn_map_downsampled": downsampled.tolist(),
                    "seg_to_seg": seg_to_seg,
                })

            # Free large attention tensors before next iter
            del outputs
            torch.cuda.empty_cache()

            iterations_out.append({
                "iter": int(r.get("iteration", n_done)),
                "input_tokens": int(n_tokens),
                "forward_ms": forward_ms,
                "segments": segments,
                "layers": layer_records,
            })
            n_done += 1
            print(f"iter {r.get('iteration')} ({n_tokens} toks, {forward_ms:.0f}ms, {len(segments)} segs)", file=sys.stderr)

    payload = {
        "model": args.model,
        "downsample": args.downsample,
        "target_layers": target_layers,
        "n_iterations": len(iterations_out),
        "n_skipped": n_skipped,
        "trace": str(args.trace),
        "iterations": iterations_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
