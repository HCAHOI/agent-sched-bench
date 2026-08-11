import json

import pytest

import scripts.evaluation.generate_pennylane_pytest_xdist_spec as generation


def test_generation_prompt_contains_only_frozen_documentation() -> None:
    prompt = generation.generation_prompt()

    assert "pytest-8.3.5+pytest-xdist-3.8.0" in prompt
    assert "--numprocesses" in prompt
    assert "PennyLaneAI" not in prompt
    assert "sampled_peak_rss_mb" not in prompt


def test_attempt_reservation_is_atomic_and_persistent(tmp_path, monkeypatch) -> None:
    output = tmp_path / "attempt"
    monkeypatch.setattr(generation, "_OUTPUT", output)

    generation._reserve_attempt("head")

    assert json.loads((output / "attempt.json").read_text())["status"] == "started"
    with pytest.raises(FileExistsError):
        generation._reserve_attempt("other-head")


def test_generation_status_checks_identity_budgets_and_event(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(generation, "_OUTPUT", tmp_path)
    response = {
        "schema": "tool-spec-v1",
        "tool": "pytest",
        "documented_version": generation._VERSION,
        "invocations": [{"tokens": ["pytest"], "operation": "pytest"}],
        "operations": [{"name": "pytest", "arguments": [], "positionals": None}],
        "relations": [],
    }
    response_text = json.dumps(response)
    response_path = tmp_path / "pytest_xdist.response.json"
    events_path = tmp_path / "pytest_xdist.events.jsonl"
    response_path.write_text(response_text)
    events_path.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": response_text},
            }
        )
        + "\n"
    )
    valid_cost = {
        "usage": {
            "input_tokens": generation._MAX_INPUT_TOKENS,
            "cached_input_tokens": 0,
            "output_tokens": 1,
        }
    }

    assert generation._generation_status(response, valid_cost) == "valid"
    over_token_cost = json.loads(json.dumps(valid_cost))
    over_token_cost["usage"]["input_tokens"] += 1
    assert (
        generation._generation_status(response, over_token_cost)
        == "unsupported_structural_failure"
    )

    oversized_text = response_text + " " * (generation._MAX_RESPONSE_BYTES + 1)
    response_path.write_text(oversized_text)
    events_path.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": oversized_text},
            }
        )
        + "\n"
    )
    assert (
        generation._generation_status(response, valid_cost)
        == "unsupported_structural_failure"
    )

    response_path.write_text(response_text)
    (tmp_path / "pytest_xdist.events.jsonl").write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "different"},
            }
        )
        + "\n"
    )
    assert (
        generation._generation_status(response, valid_cost)
        == "unsupported_structural_failure"
    )

    events_path.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": response_text},
            }
        )
        + "\n"
    )
    wrong_tool = {**response, "tool": "wrong"}
    wrong_version = {**response, "documented_version": "wrong"}
    assert (
        generation._generation_status(wrong_tool, valid_cost)
        == "unsupported_structural_failure"
    )
    assert (
        generation._generation_status(wrong_version, valid_cost)
        == "unsupported_structural_failure"
    )
