from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agents.base import AgentBase
from agents.code_agent import CodeAgent
from agents.data_agent import DataAgent
from agents.research_agent import ResearchAgent


AgentFactory = Callable[[str, str, str], AgentBase]


@dataclass(slots=True)
class RunnerTaskResult:
    """Result bundle for one task execution under the harness."""

    summary: dict[str, Any]
    trace: list[dict[str, Any]]


def build_agent_factory(agent_name: str) -> AgentFactory:
    """Resolve a configured agent type into a constructor."""
    mapping: dict[str, type[AgentBase]] = {
        "code": CodeAgent,
        "data": DataAgent,
        "research": ResearchAgent,
    }
    if agent_name not in mapping:
        raise ValueError(f"Unsupported agent name: {agent_name}")

    agent_cls = mapping[agent_name]

    def factory(agent_id: str, api_base: str, model: str) -> AgentBase:
        return agent_cls(agent_id=agent_id, api_base=api_base, model=model)

    return factory


class BenchmarkRunner:
    """Run a batch of agent tasks with bounded concurrency."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        api_base: str,
        model: str,
        concurrency: int,
        tasks: list[dict[str, Any]],
        *,
        arrival_mode: str = "closed_loop",
        task_timeout_s: float | None = None,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self.agent_factory = agent_factory
        self.api_base = api_base
        self.model = model
        self.concurrency = concurrency
        self.tasks = tasks
        self.arrival_mode = arrival_mode
        self.task_timeout_s = task_timeout_s
        self._stop_requested = False

    def request_stop(self) -> None:
        """Signal that no new tasks should start."""
        self._stop_requested = True

    async def _run_single_task(self, task: dict[str, Any], idx: int) -> RunnerTaskResult:
        agent = self.agent_factory(f"agent-{idx:04d}", self.api_base, self.model)
        try:
            run_coro = agent.run(task)
            if self.task_timeout_s is not None:
                success = await asyncio.wait_for(run_coro, timeout=self.task_timeout_s)
            else:
                success = await run_coro
            agent.task_success = success
            summary = agent.summary()
            summary["timed_out"] = False
            return RunnerTaskResult(summary=summary, trace=agent.get_trace())
        except (TimeoutError, asyncio.TimeoutError):
            agent.task_success = False
            summary = agent.summary()
            summary["timed_out"] = True
            summary["error"] = "timeout"
            return RunnerTaskResult(summary=summary, trace=agent.get_trace())
        except Exception as exc:  # pragma: no cover - defensive harness path
            agent.task_success = False
            summary = agent.summary()
            summary["timed_out"] = False
            summary["error"] = str(exc)
            summary["exception_type"] = type(exc).__name__
            return RunnerTaskResult(summary=summary, trace=agent.get_trace())

    async def run(self) -> list[RunnerTaskResult]:
        """Execute all queued tasks with closed-loop concurrency control."""
        if self.arrival_mode != "closed_loop":
            raise ValueError(f"Unsupported arrival_mode: {self.arrival_mode}")

        queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
        for index, task in enumerate(self.tasks):
            queue.put_nowait((index, task))

        results: list[RunnerTaskResult | None] = [None] * len(self.tasks)

        async def worker() -> None:
            while not queue.empty() and not self._stop_requested:
                idx, task = await queue.get()
                try:
                    results[idx] = await self._run_single_task(task, idx)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
        await queue.join()
        self._stop_requested = True
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        return [result for result in results if result is not None]


def install_signal_handlers(runner: BenchmarkRunner) -> None:
    """Install best-effort signal hooks so Ctrl-C stops starting new tasks."""
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        runner.request_stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, _request_stop)
        except NotImplementedError:
            signal.signal(signum, lambda *_args: runner.request_stop())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent benchmark runner.")
    parser.add_argument("--agent", required=True, choices=["code", "data", "research"])
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--task-timeout-s", type=float)
    parser.add_argument("--output")
    return parser.parse_args()


async def _run_cli(args: argparse.Namespace) -> list[RunnerTaskResult]:
    tasks = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))
    runner = BenchmarkRunner(
        agent_factory=build_agent_factory(args.agent),
        api_base=args.api_base,
        model=args.model,
        concurrency=args.concurrency,
        tasks=tasks,
        task_timeout_s=args.task_timeout_s,
    )
    install_signal_handlers(runner)
    return await runner.run()


def main() -> None:
    args = parse_args()
    started = time.time()
    results = asyncio.run(_run_cli(args))
    payload = {
        "started_at": started,
        "finished_at": time.time(),
        "results": [
            {"summary": result.summary, "trace": result.trace}
            for result in results
        ],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
