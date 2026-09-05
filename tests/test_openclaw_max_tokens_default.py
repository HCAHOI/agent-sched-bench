from __future__ import annotations

import inspect


def test_unified_provider_default_max_tokens_is_4096() -> None:
    from llm_call.openclaw import UnifiedProvider

    sig = inspect.signature(UnifiedProvider.__init__)
    assert sig.parameters["max_tokens"].default == 4096


def test_agent_defaults_max_tokens_is_4096() -> None:
    from agents.openclaw.config.schema import AgentDefaults

    assert AgentDefaults().max_tokens == 4096


def test_cli_max_tokens_default_is_4096() -> None:
    from agents.openclaw._cli import build_parser

    parser = build_parser()
    for action in parser._actions:
        if "--max-tokens" in action.option_strings:
            assert action.default == 4096
            return
    raise AssertionError("--max-tokens action not found in parser")

