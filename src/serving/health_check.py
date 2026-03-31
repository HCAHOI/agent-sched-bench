from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import httpx


def safe_version(package: str) -> str | None:
    """Return the installed package version when available."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_chat_payload(
    model: str,
    messages: list[dict[str, str]],
    program_id: str | None = None,
) -> dict[str, Any]:
    """Construct a minimal OpenAI-compatible chat payload."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0.0,
    }
    if program_id is not None:
        payload["program_id"] = program_id
    return payload


async def wait_for_models_endpoint(
    client: httpx.AsyncClient,
    api_base: str,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    """Poll `/models` until the raw vLLM server is responsive."""
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"{api_base}/models")
            if response.status_code == 200:
                return response.json()
            last_error = f"unexpected status {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        await asyncio.sleep(poll_interval_s)
    raise TimeoutError(f"Timed out waiting for /models: {last_error}")


async def require_metrics_endpoint(client: httpx.AsyncClient, metrics_url: str) -> str:
    """Require the Prometheus metrics endpoint to respond successfully."""
    response = await client.get(metrics_url)
    response.raise_for_status()
    return response.text


async def run_chat_smoke(
    client: httpx.AsyncClient,
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    program_id: str | None = None,
) -> dict[str, Any]:
    """Send a minimal chat-completions request to the server."""
    payload = build_chat_payload(model=model, messages=messages, program_id=program_id)
    response = await client.post(
        f"{api_base}/chat/completions",
        json=payload,
    )
    response.raise_for_status()
    return response.json()


async def verify_server(args: argparse.Namespace) -> dict[str, Any]:
    """Run the raw vLLM readiness checks and return a report."""
    timeout = httpx.Timeout(args.timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        models_payload = await wait_for_models_endpoint(
            client=client,
            api_base=args.api_base,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
        )
        resolved_model = args.model
        if resolved_model == "auto":
            models = models_payload.get("data") or []
            if not models:
                raise ValueError("no models available to resolve auto model id")
            resolved_model = models[0]["id"]
        pre_metrics_payload = await require_metrics_endpoint(client, args.metrics_url)
        chat_payloads = []
        request_messages: list[dict[str, str]] = [{"role": "user", "content": args.prompt}]
        for index in range(args.repeat):
            chat_payloads.append(
                await run_chat_smoke(
                    client=client,
                    api_base=args.api_base,
                    model=resolved_model,
                    messages=request_messages,
                    program_id=args.program_id,
                )
            )
            if index < args.repeat - 1:
                assistant_message = (
                    (((chat_payloads[-1].get("choices") or [{}])[0].get("message") or {}).get("content"))
                    or ""
                )
                request_messages = [
                    *request_messages,
                    {"role": "assistant", "content": assistant_message},
                    {"role": "user", "content": args.followup_prompt},
                ]
        post_metrics_payload = await require_metrics_endpoint(client, args.metrics_url)

    pre_prefix_cache_hit_rates = parse_prefix_cache_hit_rates(pre_metrics_payload)
    post_prefix_cache_hit_rates = parse_prefix_cache_hit_rates(post_metrics_payload)

    return {
        "timestamp": int(time.time()),
        "api_base": args.api_base,
        "metrics_url": args.metrics_url,
        "model_path": args.model_path,
        "requested_model": args.model,
        "resolved_model": resolved_model,
        "program_id": args.program_id,
        "repeat": args.repeat,
        "followup_prompt": args.followup_prompt,
        "require_prefix_cache_hit": args.require_prefix_cache_hit,
        "vllm_spec": args.vllm_spec,
        "installed_versions": {
            "vllm": safe_version("vllm"),
            "httpx": safe_version("httpx"),
        },
        "models_response": models_payload,
        "metrics_available": "vllm:" in post_metrics_payload,
        "metrics_sample": post_metrics_payload.splitlines()[:10],
        "pre_prefix_cache_hit_rates": pre_prefix_cache_hit_rates,
        "post_prefix_cache_hit_rates": post_prefix_cache_hit_rates,
        "chat_responses": chat_payloads,
    }


def validate_report(report: dict[str, Any]) -> list[str]:
    """Validate the readiness report against serving-checkpoint acceptance signals."""
    errors: list[str] = []
    if not report["models_response"].get("data"):
        errors.append("/v1/models returned an empty model list")
    if not report["metrics_available"]:
        errors.append("/metrics did not expose any vllm-prefixed metrics")
    if report.get("require_prefix_cache_hit"):
        pre_rates = report.get("pre_prefix_cache_hit_rates") or {}
        post_rates = report.get("post_prefix_cache_hit_rates") or {}
        deltas = [
            post_rates.get(metric, 0.0) - pre_rates.get(metric, 0.0)
            for metric in set(pre_rates) | set(post_rates)
        ]
        if max(deltas, default=0.0) <= 0.0:
            errors.append("prefix cache hit rate did not increase during this run")

    for index, chat_response in enumerate(report["chat_responses"]):
        choices = chat_response.get("choices") or []
        if not choices:
            errors.append(f"chat completion #{index} returned no choices")
            continue
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            errors.append(f"chat completion #{index} returned empty content")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a raw vLLM OpenAI-compatible server and emit a report."
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="Reply with the word READY.")
    parser.add_argument(
        "--followup-prompt",
        default="Continue the same conversation and reply with READY-AGAIN.",
    )
    parser.add_argument("--program-id")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vllm-spec", required=True)
    parser.add_argument("--require-prefix-cache-hit", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def parse_prefix_cache_hit_rates(metrics_payload: str) -> dict[str, float]:
    """Extract prefix cache hit-rate gauges from a Prometheus metrics payload."""
    metrics: dict[str, float] = {}
    for line in metrics_payload.splitlines():
        if line.startswith("vllm:gpu_prefix_cache_hit_rate"):
            metrics["gpu_prefix_cache_hit_rate"] = float(line.split()[-1])
        if line.startswith("vllm:cpu_prefix_cache_hit_rate"):
            metrics["cpu_prefix_cache_hit_rate"] = float(line.split()[-1])
    return metrics


def main() -> None:
    args = parse_args()
    report = asyncio.run(verify_server(args))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.fail_on_mismatch:
        errors = validate_report(report)
        if errors:
            raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
