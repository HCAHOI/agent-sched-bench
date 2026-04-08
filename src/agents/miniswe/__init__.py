"""Miniswe scaffold package init.

Side effect on import: registers the miniswe scaffold prepare adapter
into `trace_collect.scaffold_registry`. The lazy-import gate in
`scaffold_registry._ensure_loaded` triggers this on the first
`get_prepare("miniswe")` call.
"""

from agents.miniswe.agent import (
    MiniSWECodeAgent as MiniSWECodeAgent,
    ContextManagedAgent as ContextManagedAgent,
)


def _register_miniswe_prepare() -> None:
    """Build and register the miniswe scaffold prepare adapter."""
    from trace_collect.scaffold_registry import (
        PreparedWorkspace,
        SimulatePrepareConfig,
        register_scaffold_prepare,
    )

    async def _miniswe_prepare(
        task: dict, config: SimulatePrepareConfig
    ) -> PreparedWorkspace:
        # MiniSWECodeAgent is already imported at the top of this
        # module — the cycle break (Pre-mortem B item 2 of
        # trace-sim-vastai-pipeline plan) lives in scaffold_registry's
        # _ensure_loaded, which only invokes this module on the first
        # get_prepare("miniswe") call. The closure here just uses the
        # already-loaded class.
        agent = MiniSWECodeAgent(
            agent_id=config.agent_id,
            api_base=config.api_base,
            model=config.model,
            api_key=config.api_key,
            command_timeout_s=config.command_timeout_s,
            task_timeout_s=config.task_timeout_s,
            repos_root=str(config.repos_root) if config.repos_root else None,
            max_context_tokens=config.max_context_tokens,
        )
        await agent.prepare(task)

        # Cleanup closure captures the agent instance so the existing
        # mini-swe cleanup contract is preserved verbatim (Pre-mortem B
        # item 1 of trace-sim-vastai-pipeline plan: byte-identical
        # mini-swe regression requires preserving cleanup semantics).
        def _cleanup() -> None:
            import shutil

            if agent._workdir:
                shutil.rmtree(agent._workdir, ignore_errors=True)

        assert agent._workdir is not None, (
            "MiniSWECodeAgent.prepare() succeeded but _workdir is None"
        )
        return PreparedWorkspace(
            repo_dir=agent._workdir / "repo",
            cleanup=_cleanup,
        )

    register_scaffold_prepare("miniswe", _miniswe_prepare)


_register_miniswe_prepare()
