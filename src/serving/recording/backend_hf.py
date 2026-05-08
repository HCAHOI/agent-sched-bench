"""HuggingFace backend for optional internal recording."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import secrets
import string
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import json_repair

from agents.openclaw.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from serving.recording.hooks import LayerCapturer
from serving.recording.recording import RecordingConfig, segment_role


_ALLOWED_MSG_KEYS = frozenset(
    {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"}
)
_ALNUM = string.ascii_letters + string.digits


def _short_tool_id() -> str:
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _message_has_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content)
    if isinstance(content, list):
        return bool(content)
    return True


def _token_count(encoded: Any) -> int:
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(input_ids, "ndim"):
        return int(input_ids.numel()) if input_ids.ndim == 1 else int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _token_boundary_for_char(tokenizer: Any, text: str, offsets: Any, char_pos: int) -> int:
    if char_pos <= 0:
        return 0
    if offsets is None:
        return _token_count(tokenizer(text[:char_pos], add_special_tokens=False))
    for idx, pair in enumerate(offsets):
        start = int(pair[0])
        if start >= char_pos:
            return idx
    return len(offsets)


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[Any, list[list[int]] | None]:
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded, None
    offsets = encoded.pop("offset_mapping")[0].tolist()
    return encoded, offsets


def _normalize_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json_repair.loads(value)
    if not isinstance(value, dict):
        return {}
    return value


def _normalize_tool_name(value: Any) -> str:
    name = str(value or "").strip()
    while name.startswith("<function="):
        name = name[len("<function=") :].strip()
    while name.endswith(">"):
        name = name[:-1].strip()
    return name


def _parse_tool_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    if (
        value[0] not in "{[\"'-0123456789"
        and value not in {"true", "false", "null"}
    ):
        return value
    try:
        return json_repair.loads(value)
    except Exception:
        return value


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = LLMProvider._sanitize_request_messages(
        LLMProvider._sanitize_empty_content(messages),
        _ALLOWED_MSG_KEYS,
    )
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


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_chat_with_segments(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]], str]:
    messages = _normalize_messages(messages)
    full_text = _apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=True,
    )

    char_segments: list[dict[str, Any]] = []
    previous_end = 0
    for idx, message in enumerate(messages):
        prefix = _apply_chat_template(
            tokenizer,
            messages[: idx + 1],
            tools=tools,
            add_generation_prompt=False,
        )
        if not full_text.startswith(prefix):
            raise ValueError("chat template prefixes do not align with full prompt")
        end = len(prefix)
        if previous_end < end:
            char_segments.append(
                {
                    "role": segment_role(message),
                    "message_index": idx,
                    "char_start": previous_end,
                    "char_end": end,
                    "has_content": _message_has_content(message),
                    "has_tool_calls": bool(message.get("tool_calls")),
                }
            )
        previous_end = end
    if previous_end < len(full_text):
        char_segments.append(
            {
                "role": "gen_prompt",
                "message_index": None,
                "char_start": previous_end,
                "char_end": len(full_text),
                "has_content": False,
                "has_tool_calls": False,
            }
        )

    encoded, offsets = _tokenize_with_offsets(tokenizer, full_text)
    segments: list[dict[str, Any]] = []
    for segment in char_segments:
        start = _token_boundary_for_char(
            tokenizer, full_text, offsets, int(segment["char_start"])
        )
        end = _token_boundary_for_char(
            tokenizer, full_text, offsets, int(segment["char_end"])
        )
        if start >= end:
            continue
        segments.append(
            {
                **segment,
                "token_start": start,
                "token_end": end,
            }
        )
    return encoded, segments, full_text


def _strip_tool_blocks(text: str, blocks: list[tuple[int, int]]) -> str:
    if not blocks:
        return text
    parts: list[str] = []
    last = 0
    for start, end in blocks:
        parts.append(text[last:start])
        last = end
    parts.append(text[last:])
    return "".join(parts).strip() or None


def _parse_qwen_xml_tool_calls(text: str) -> tuple[str | None, list[ToolCallRequest]]:
    import re

    calls: list[ToolCallRequest] = []
    blocks: list[tuple[int, int]] = []
    outer_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    function_pattern = re.compile(
        r"<function=([^>\n]+)>\s*(.*?)\s*</function>",
        re.DOTALL,
    )
    parameter_pattern = re.compile(
        r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )
    for match in outer_pattern.finditer(text):
        function_match = function_pattern.search(match.group(1))
        if function_match is None:
            continue
        args = {
            param_match.group(1).strip(): _parse_tool_value(param_match.group(2))
            for param_match in parameter_pattern.finditer(function_match.group(2))
        }
        calls.append(
            ToolCallRequest(
                id=_short_tool_id(),
                name=_normalize_tool_name(function_match.group(1)),
                arguments=args,
            )
        )
        blocks.append((match.start(), match.end()))
    return _strip_tool_blocks(text, blocks), calls


def _parse_qwen_json_tool_calls(text: str) -> tuple[str | None, list[ToolCallRequest]]:
    import re

    calls: list[ToolCallRequest] = []
    blocks: list[tuple[int, int]] = []
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        try:
            payload = json_repair.loads(match.group(1))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        calls.append(
            ToolCallRequest(
                id=_short_tool_id(),
                name=_normalize_tool_name(payload.get("name")),
                arguments=_normalize_tool_arguments(payload.get("arguments", {})),
            )
        )
        blocks.append((match.start(), match.end()))
    return _strip_tool_blocks(text, blocks), calls


def _parse_glm_tool_calls(text: str) -> tuple[str | None, list[ToolCallRequest]]:
    import re

    calls: list[ToolCallRequest] = []
    blocks: list[tuple[int, int]] = []
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        body = match.group(1).strip()
        if not body:
            continue
        name, _, rest = body.partition("\n")
        args: dict[str, Any] = {}
        for arg_match in re.finditer(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            rest,
            re.DOTALL,
        ):
            key = arg_match.group(1).strip()
            raw_value = arg_match.group(2).strip()
            try:
                args[key] = _parse_tool_value(raw_value)
            except Exception:
                args[key] = raw_value
        calls.append(
            ToolCallRequest(
                id=_short_tool_id(),
                name=_normalize_tool_name(name),
                arguments=args,
            )
        )
        blocks.append((match.start(), match.end()))
    return _strip_tool_blocks(text, blocks), calls


def parse_text_tool_calls(text: str) -> tuple[str | None, list[ToolCallRequest]]:
    content, calls = _parse_qwen_xml_tool_calls(text)
    if calls:
        return content, calls
    content, calls = _parse_qwen_json_tool_calls(text)
    if calls:
        return content, calls
    return _parse_glm_tool_calls(text)


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _hf_max_memory(torch_module: Any) -> dict[int | str, str] | None:
    gpu_gib = _positive_int_env("HF_RECORDING_MAX_GPU_MEMORY_GIB")
    if gpu_gib is None:
        return None
    device_count = int(torch_module.cuda.device_count())
    if device_count <= 0:
        raise ValueError("HF_RECORDING_MAX_GPU_MEMORY_GIB requires CUDA devices")
    cpu_gib = _positive_int_env("HF_RECORDING_MAX_CPU_MEMORY_GIB") or 128
    max_memory: dict[int | str, str] = {
        device_idx: f"{gpu_gib}GiB" for device_idx in range(device_count)
    }
    max_memory["cpu"] = f"{cpu_gib}GiB"
    return max_memory


class HFRecordingProvider(LLMProvider):
    """OpenClaw provider that records HF model internals while generating."""

    def __init__(
        self,
        *,
        default_model: str,
        config: RecordingConfig | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base)
        self.default_model = default_model
        self.config = config or RecordingConfig()
        self.generation = GenerationSettings(temperature=0.1, max_tokens=8192)
        self._call_idx = 0

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.model_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            default_model,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            default_model,
            torch_dtype=dtype,
            device_map=self.config.device_map,
            max_memory=_hf_max_memory(torch),
            attn_implementation="sdpa",
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()
        self._torch = torch
        self._captures_router_logits = self._model_has_router_logits()
        self.capturer = LayerCapturer(
            self.model,
            config=self.config,
            model_summary=self._model_summary(),
        )

    def get_default_model(self) -> str:
        return self.default_model

    def start_attempt(self, recordings_dir: Path) -> None:
        self._call_idx = 0
        self.capturer.start_attempt(recordings_dir)

    def finish_attempt(self) -> None:
        self.capturer.finish_attempt()

    def _model_summary(self) -> dict[str, Any]:
        cfg = self.model.config
        return {
            "name": self.default_model,
            "architectures": list(getattr(cfg, "architectures", []) or []),
            "model_type": getattr(cfg, "model_type", None),
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
            "num_attention_heads": getattr(cfg, "num_attention_heads", None),
            "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "num_experts": getattr(cfg, "num_experts", None)
            or getattr(cfg, "num_local_experts", None),
            "num_experts_per_tok": getattr(cfg, "num_experts_per_tok", None)
            or getattr(cfg, "num_experts_per_token", None),
            "record_router_logits": self._captures_router_logits,
            "recording_config": asdict(self.config),
        }

    def _model_has_router_logits(self) -> bool:
        cfg = self.model.config
        return any(
            getattr(cfg, name, None) is not None
            for name in (
                "num_experts",
                "num_local_experts",
                "num_experts_per_tok",
                "num_experts_per_token",
            )
        )

    def _input_device(self) -> Any:
        for parameter in self.model.parameters():
            if str(parameter.device) != "meta":
                return parameter.device
        return self._torch.device("cpu")

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        del model
        if reasoning_effort is not None:
            return LLMResponse(
                content=(
                    "Error: HF recording provider does not support "
                    "reasoning_effort."
                ),
                finish_reason="error",
                extra={"error_type": "unsupported_reasoning_effort"},
            )
        if isinstance(tool_choice, dict) or tool_choice not in {None, "auto", "none"}:
            return LLMResponse(
                content=(
                    'Error: HF recording provider does not support forced tool_choice; '
                    'supported values are ["none", "auto"].'
                ),
                finish_reason="error",
                extra={"error_type": "unsupported_tool_choice"},
            )
        template_tools = None if tool_choice == "none" else tools
        call_idx = self._call_idx
        self._call_idx += 1
        started_at = time.time()

        encoded, segments, _prompt_text = tokenize_chat_with_segments(
            self.tokenizer,
            messages,
            tools=template_tools,
        )
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        input_token_count = int(input_ids.shape[-1])

        def run_generate() -> tuple[str, list[int]]:
            input_tensor = input_ids.to(self._input_device())
            attention_mask = self._torch.ones_like(input_tensor)
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_tensor,
                "attention_mask": attention_mask,
                "max_new_tokens": max(1, int(max_tokens)),
                "do_sample": temperature > 0,
                "return_dict_in_generate": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = float(temperature)
            with self._torch.no_grad(), self.capturer.recording_session(
                call_idx=call_idx,
                segments=segments,
                input_token_count=input_token_count,
            ):
                sequences = self.model.generate(**generation_kwargs)
                output_ids = sequences[0, input_token_count:].detach().cpu().tolist()
                if self._captures_router_logits:
                    with self.capturer.suspend_attention():
                        router_outputs = self.model(
                            input_ids=sequences,
                            attention_mask=self._torch.ones_like(sequences),
                            use_cache=False,
                            return_dict=True,
                            output_attentions=False,
                            output_router_logits=True,
                        )
                    self.capturer.record_router_logits(
                        router_outputs,
                        total_tokens=int(sequences.shape[-1]),
                    )
                self.capturer.flush(output_token_ids=output_ids)
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return text, output_ids

        self._clear_cuda_cache()
        try:
            text, output_ids = await asyncio.to_thread(run_generate)
        finally:
            self._clear_cuda_cache()
        content, tool_calls = parse_text_tool_calls(text)
        elapsed_ms = (time.time() - started_at) * 1000.0
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={
                "prompt_tokens": input_token_count,
                "completion_tokens": len(output_ids),
                "total_tokens": input_token_count + len(output_ids),
            },
            extra={
                "llm_wall_ts_end": time.time(),
                "llm_call_time_ms": elapsed_ms,
                "llm_timing_source": "hf_recording_generate_wall_ms",
            },
        )


class HFRecordingServer:
    """Minimal OpenAI-compatible chat endpoint for task-container agents."""

    def __init__(
        self,
        provider: HFRecordingProvider,
        *,
        bind_host: str = "0.0.0.0",
        public_host: str | None = None,
    ) -> None:
        self.provider = provider
        self._bind_host = bind_host
        self._public_host = public_host or os.environ.get("HF_RECORDING_PUBLIC_HOST")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def api_base(self) -> str:
        if self._server is None:
            raise RuntimeError("server is not running")
        host, port = self._server.server_address
        if host in {"0.0.0.0", "::"}:
            host = self._public_host or "127.0.0.1"
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "HFRecordingServer":
        provider = self.provider

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path not in {"/v1/chat/completions", "/chat/completions"}:
                    self.send_error(404)
                    return
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = asyncio.run(
                    provider.chat(
                        messages=payload.get("messages") or [],
                        tools=payload.get("tools"),
                        model=payload.get("model"),
                        max_tokens=int(payload.get("max_tokens") or 4096),
                        temperature=float(payload.get("temperature") or 0.0),
                        reasoning_effort=payload.get("reasoning_effort"),
                        tool_choice=payload.get("tool_choice"),
                    )
                )
                body = self._response_body(response)
                raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, _format: str, *args: Any) -> None:
                return

            @staticmethod
            def _response_body(response: LLMResponse) -> dict[str, Any]:
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                }
                if response.tool_calls:
                    message["tool_calls"] = [
                        tool_call.to_openai_tool_call()
                        for tool_call in response.tool_calls
                    ]
                return {
                    "id": "hf-recording",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": provider.default_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": response.finish_reason,
                        }
                    ],
                    "usage": response.usage,
                }

        self._server = ThreadingHTTPServer((self._bind_host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
