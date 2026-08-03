"""Codex Responses provider using a local ChatGPT subscription login."""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json_repair
from openai import AsyncOpenAI

from llm_call.provider_base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)

CODEX_DEFAULT_API_BASE = "https://chatgpt.com/backend-api/codex"


@dataclass(frozen=True)
class CodexCredentials:
    access_token: str
    account_id: str | None


class CodexProvider(LLMProvider):
    """Call Codex through credentials created by ``codex login``."""

    def __init__(
        self,
        api_key: str | None,
        api_base: str | None,
        default_model: str,
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        service_tier: str | None = None,
        timeout: float | None = None,
    ) -> None:
        unsupported = [
            name
            for name, value in {
                "max_tokens": max_tokens if max_tokens != 4096 else None,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }.items()
            if value is not None
        ]
        if unsupported:
            raise ValueError(
                "Codex Responses does not support generation setting(s): "
                + ", ".join(unsupported)
            )
        if service_tier not in {None, "fast"}:
            raise ValueError(
                "Codex Responses service_tier must be 'fast' or omitted"
            )
        super().__init__(api_key, api_base or CODEX_DEFAULT_API_BASE)
        self.default_model = default_model
        self.service_tier = "priority" if service_tier == "fast" else None
        self.timeout = timeout
        self.generation = GenerationSettings(
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        if max_tokens != 4096:
            return _error_response(
                ValueError("Codex Responses does not support max_tokens")
            )
        if temperature != 0.0:
            return _error_response(
                ValueError("Codex Responses does not support temperature")
            )
        return await self._request(
            messages,
            tools=tools,
            model=model,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            on_content_delta=None,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        if max_tokens != 4096:
            return _error_response(
                ValueError("Codex Responses does not support max_tokens")
            )
        if temperature != 0.0:
            return _error_response(
                ValueError("Codex Responses does not support temperature")
            )
        return await self._request(
            messages,
            tools=tools,
            model=model,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            on_content_delta=on_content_delta,
        )

    async def _request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        model: str | None,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        on_content_delta: Callable[[str], Awaitable[None]] | None,
    ) -> LLMResponse:
        callback_emitted = False
        try:
            credentials = load_codex_credentials(self.api_key)
            instructions, input_items = _responses_input(
                self._sanitize_empty_content(messages)
            )
            kwargs: dict[str, Any] = {
                "model": model or self.default_model,
                "instructions": instructions,
                "input": input_items,
                "parallel_tool_calls": False,
                "store": False,
                "stream": True,
                "include": [],
            }
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            if self.service_tier is not None:
                kwargs["service_tier"] = self.service_tier
            if tools:
                kwargs["tools"] = [_responses_tool(tool) for tool in tools]
                kwargs["tool_choice"] = _responses_tool_choice(tool_choice)

            headers = (
                {"ChatGPT-Account-ID": credentials.account_id}
                if credentials.account_id
                else None
            )
            client_kwargs: dict[str, Any] = {
                "api_key": credentials.access_token,
                "base_url": (
                    (self.api_base or CODEX_DEFAULT_API_BASE).rstrip("/") + "/"
                ),
                "default_headers": headers,
            }
            if self.timeout is not None:
                client_kwargs["timeout"] = self.timeout
            client = AsyncOpenAI(**client_kwargs)


            async def emit(delta: str) -> None:
                nonlocal callback_emitted
                callback_emitted = True
                if on_content_delta is not None:
                    await on_content_delta(delta)

            try:
                stream = await client.responses.create(**kwargs)
                return await _consume_stream(
                    stream,
                    emit if on_content_delta is not None else None,
                )
            finally:
                await client.close()
        except Exception as exc:
            if callback_emitted:
                return _error_response(exc, after_stream_output=True)
            return _error_response(exc)

    def get_default_model(self) -> str:
        return self.default_model


def load_codex_credentials(access_token: str | None = None) -> CodexCredentials:
    """Load a Codex OAuth token from an override, environment, or auth.json."""

    token = (access_token or os.environ.get("CODEX_ACCESS_TOKEN", "")).strip()
    account_id = os.environ.get("CODEX_ACCOUNT_ID", "").strip() or None
    auth_path = _codex_auth_path()

    if not token:
        if not auth_path.is_file():
            raise ValueError(
                f"Codex auth file not found at {auth_path}. Run `codex login` first."
            )
        data = _read_auth_json(auth_path)
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            raise ValueError(
                "Codex auth file has no ChatGPT tokens. "
                "Run `codex login` with ChatGPT auth."
            )
        raw_token = tokens.get("access_token")
        if not isinstance(raw_token, str) or not raw_token.strip():
            raise ValueError("Codex access token is missing. Run `codex login`.")
        token = raw_token.strip()
        raw_account_id = tokens.get("account_id")
        if account_id is None and isinstance(raw_account_id, str):
            account_id = raw_account_id.strip() or None
    elif account_id is None and auth_path.is_file():
        tokens = _read_auth_json(auth_path).get("tokens")
        raw_account_id = tokens.get("account_id") if isinstance(tokens, dict) else None
        if isinstance(raw_account_id, str):
            account_id = raw_account_id.strip() or None

    expiry = _jwt_expiry(token)
    if expiry is not None and expiry <= time.time() + 60:
        raise ValueError("Codex access token is expired. Run `codex login`.")
    return CodexCredentials(token, account_id)


def _codex_auth_path() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser() / "auth.json"


def _read_auth_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Codex auth file is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Codex auth file must contain an object: {path}")
    return value


def _jwt_expiry(token: str) -> int | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    except (ValueError, json.JSONDecodeError):
        return None
    expiry = data.get("exp")
    return int(expiry) if isinstance(expiry, (int, float)) else None


def _responses_input(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role in {"system", "developer"}:
            text = _text_content(content)
            if text:
                instructions.append(text)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Codex tool message is missing tool_call_id")
            if not isinstance(content, str):
                raise ValueError("Codex tool message content must be text")
            items.append(
                {"type": "function_call_output", "call_id": call_id, "output": content}
            )
            continue
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            content_items = _content_items(role, content)
            if content_items:
                items.append({"type": "message", "role": role, "content": content_items})
            items.extend(_function_call_items(message["tool_calls"]))
            continue
        content_items = _content_items(role, content)
        if content_items:
            items.append(
                {"type": "message", "role": role or "user", "content": content_items}
            )
    return "\n\n".join(instructions), items


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text", "")).strip()
        for part in content
        if isinstance(part, dict)
        and part.get("type") in {"text", "input_text", "output_text"}
        and str(part.get("text", "")).strip()
    )


def _content_items(role: str, content: Any) -> list[dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": text_type, "text": content}] if content else []
    if not isinstance(content, list):
        raise ValueError("Codex message content must be text or content parts")
    items: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Codex content part must be an object")
        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                items.append({"type": text_type, "text": text})
        elif part_type == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str) or not url:
                raise ValueError("Codex image content part has no URL")
            items.append({"type": "input_image", "image_url": url})
        else:
            raise ValueError(f"Unsupported Codex content part type: {part_type}")
    return items


def _function_call_items(tool_calls: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ValueError("Codex assistant tool call has no function")
        function = call["function"]
        call_id = call.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Codex assistant tool call has no id")
        if not isinstance(name, str) or not name:
            raise ValueError("Codex assistant tool call has no function name")
        if not isinstance(arguments, str):
            raise ValueError("Codex assistant tool call arguments must be text")
        items.append(
            {
                "type": "function_call",
                "name": name,
                "arguments": arguments,
                "call_id": call_id,
            }
        )
    return items


def _responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, dict):
        raise ValueError("Codex tool has no function schema")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Codex function tool has no name")
    converted: dict[str, Any] = {
        "type": "function",
        "name": name,
        "parameters": function.get("parameters", {"type": "object"}),
    }
    description = function.get("description")
    if isinstance(description, str) and description:
        converted["description"] = description
    return converted


def _responses_tool_choice(
    tool_choice: str | dict[str, Any] | None,
) -> str | dict[str, str]:
    if not isinstance(tool_choice, dict):
        return tool_choice or "auto"
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError("Codex function tool_choice has no name")
    return {"type": "function", "name": name}


async def _consume_stream(
    stream: Any,
    on_content_delta: Callable[[str], Awaitable[None]] | None,
) -> LLMResponse:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    usage: dict[str, int] = {}
    response_metadata: dict[str, Any] = {"provider": "codex"}
    saw_text_delta = False

    saw_completed = False
    async for event in stream:
        event_type = getattr(event, "type", "")
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = getattr(event, "delta", None)
            if isinstance(delta, str) and delta:
                saw_text_delta = True
                content_parts.append(delta)
                if on_content_delta is not None:
                    await on_content_delta(delta)
        elif event_type == "response.reasoning_summary_text.delta":
            delta = getattr(event, "delta", None)
            if isinstance(delta, str) and delta:
                reasoning_parts.append(delta)
        elif event_type == "response.output_item.done":
            item = getattr(event, "item", None)
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                tool_calls.append(_tool_call_from_item(item))
            elif item_type == "message" and not saw_text_delta:
                text = _message_item_text(item)
                if text:
                    saw_text_delta = True
                    content_parts.append(text)
                    if on_content_delta is not None:
                        await on_content_delta(text)
        elif event_type in {"response.completed", "response.incomplete"}:
            saw_completed = event_type == "response.completed"
            response = getattr(event, "response", None)
            usage = _usage_from_response(response) or usage
            response_metadata.update(_response_metadata(response))
            if event_type == "response.incomplete":
                details = getattr(response, "incomplete_details", None)
                reason = getattr(details, "reason", None)
                return LLMResponse(
                    content=f"Codex Responses incomplete: {reason or 'unknown'}",
                    finish_reason="error",
                    usage=usage,
                    extra={"codex_metadata": response_metadata},
                )
        elif event_type == "response.failed":
            response = getattr(event, "response", None)
            usage = _usage_from_response(response) or usage
            response_metadata.update(_response_metadata(response))
            error = getattr(response, "error", None)
            message = getattr(error, "message", None)
            response_metadata["error"] = message or "unknown"
            retry_disabled = saw_text_delta and on_content_delta is not None
            return LLMResponse(
                content=(
                    "Codex stream failed after emitting output; retry disabled"
                    if retry_disabled
                    else f"Codex Responses failed: {message or 'unknown'}"
                ),
                finish_reason="error",
                usage=usage,
                extra={"codex_metadata": response_metadata},
            )
        elif event_type == "error":
            message = getattr(event, "message", None) or "unknown"
            response_metadata["error"] = message
            retry_disabled = saw_text_delta and on_content_delta is not None
            return LLMResponse(
                content=(
                    "Codex stream failed after emitting output; retry disabled"
                    if retry_disabled
                    else f"Codex Responses error: {message}"
                ),
                finish_reason="error",
                usage=usage,
                extra={"codex_metadata": response_metadata},
            )

    if not saw_completed:
        response_metadata["error"] = "stream ended before response.completed"
        return LLMResponse(
            content="Codex stream ended before response.completed",
            finish_reason="error",
            usage=usage,
            extra={"codex_metadata": response_metadata},
        )
    return LLMResponse(
        content="".join(content_parts) or None,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=usage,
        reasoning_content="".join(reasoning_parts) or None,
        extra={"codex_metadata": response_metadata},
    )


def _tool_call_from_item(item: Any) -> ToolCallRequest:
    name = getattr(item, "name", None)
    raw_arguments = getattr(item, "arguments", None)
    call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Codex function call has no name")
    if not isinstance(raw_arguments, str):
        raise ValueError("Codex function call arguments are not text")
    arguments = json_repair.loads(raw_arguments) if raw_arguments else {}
    if not isinstance(arguments, dict):
        raise ValueError("Codex function call arguments must be an object")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("Codex function call has no call_id")
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _message_item_text(item: Any) -> str:
    parts: list[str] = []
    for content in getattr(item, "content", None) or []:
        content_type = getattr(content, "type", None)
        if content_type == "output_text":
            text = getattr(content, "text", None)
        elif content_type == "refusal":
            text = getattr(content, "refusal", None)
        else:
            text = None
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _usage_from_response(response: Any) -> dict[str, int]:
    raw = getattr(response, "usage", None)
    if raw is None:
        return {}
    prompt = int(getattr(raw, "input_tokens", 0) or 0)
    completion = int(getattr(raw, "output_tokens", 0) or 0)
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(getattr(raw, "total_tokens", 0) or prompt + completion),
    }
    details = getattr(raw, "input_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0)
    if cached:
        result["cached_tokens"] = cached
    return result


def _response_metadata(response: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "service_tier": getattr(response, "service_tier", None),
            "status": getattr(response, "status", None),
        }.items()
        if value is not None
    }


def _error_response(
    exc: Exception, *, after_stream_output: bool = False
) -> LLMResponse:
    status = getattr(exc, "status_code", None)
    extra: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "provider": "codex",
    }
    if status is not None:
        extra["http_status"] = status
    if after_stream_output:
        extra["error"] = str(exc)
        message = "Codex stream failed after emitting output; retry disabled"
    else:
        message = f"Error calling LLM: {exc}"
    return LLMResponse(content=message, finish_reason="error", extra=extra)
