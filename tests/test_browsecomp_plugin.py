from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from agents.benchmarks import REGISTRY, get_benchmark_class
from agents.benchmarks.base import BenchmarkConfig
from agents.benchmarks.browsecomp import BrowseCompBenchmark, decrypt, derive_key
from trace_collect.prompt_loader import load_prompt_template, render_prompt


_SCHEMA = ["problem", "answer", "problem_topic", "canary"]


def _encrypt(plaintext: str, password: str) -> str:
    raw = plaintext.encode("utf-8")
    key = derive_key(password, len(raw))
    encrypted = bytes(value ^ key_byte for value, key_byte in zip(raw, key))
    return base64.b64encode(encrypted).decode("ascii")


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_SCHEMA, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _encrypted_csv_bytes(
    *,
    question: str = "Which planet is known as the red planet?",
    answer: str = "Mars",
    topic: str = "astronomy",
    canary: str = "row-canary",
) -> bytes:
    return _csv_bytes(
        [
            {
                "problem": _encrypt(question, canary),
                "answer": _encrypt(answer, canary),
                "problem_topic": topic,
                "canary": canary,
            }
        ]
    )


def _make_config(
    tmp_path: Path,
    *,
    raw_bytes: bytes | None = None,
    **extra_overrides,
) -> BenchmarkConfig:
    payload = raw_bytes if raw_bytes is not None else _encrypted_csv_bytes()
    extras = {
        "task_source_kind": "browsecomp_official_csv",
        "csv_url": "https://example.invalid/browsecomp.csv",
        "csv_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_schema": list(_SCHEMA),
        "expected_row_count": 1,
        "encrypted": True,
        "id_field": "_row_index",
        "question_field": "problem",
        "answer_field": "answer",
        "topic_field": "problem_topic",
        "canary_field": "canary",
        "scorer_template": "grader",
        "scorer_provider": "openrouter",
        "scorer_model": "openai/gpt-4.1",
        "scorer_temperature": 0.0,
        "web_search_provider": "brave",
        "web_fetch_provider": "jina",
        "max_search_calls": 20,
        "max_fetch_calls": 20,
        **extra_overrides,
    }
    return BenchmarkConfig(
        slug="browsecomp",
        display_name="BrowseComp",
        trace_root=tmp_path / "traces",
        data_root=tmp_path / "data",
        default_max_iterations=100,
        selection_n=32,
        selection_seed=42,
        default_prompt_template="default",
        extras=extras,
    )


def test_browsecomp_registered_for_deep_research_only(tmp_path: Path) -> None:
    assert REGISTRY["browsecomp"] is BrowseCompBenchmark
    plugin = get_benchmark_class("browsecomp")(_make_config(tmp_path))

    assert plugin.execution_environment == "host"
    assert plugin.runtime_mode_for("deep-research") == "host_controller"
    assert plugin.image_name_for({"image_name": "should-not-be-used"}) is None
    with pytest.raises(NotImplementedError, match="deep-research"):
        plugin.runtime_mode_for("openclaw")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_schema": ["answer", "problem", "problem_topic", "canary"]}, "schema"),
        ({"expected_row_count": 0}, "row_count"),
        ({"max_search_calls": 0}, "max_search_calls"),
        ({"max_fetch_calls": 0}, "max_fetch_calls"),
        ({"scorer_temperature": 0.2}, "scorer_temperature"),
    ],
)
def test_browsecomp_config_validation_fails_closed(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BrowseCompBenchmark(_make_config(tmp_path, **overrides))


def test_browsecomp_config_requires_all_source_and_runtime_extras(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="csv_url"):
        BrowseCompBenchmark(
            BenchmarkConfig(
                slug="browsecomp",
                display_name="BrowseComp",
                trace_root=tmp_path / "traces",
                default_max_iterations=100,
                selection_n=1,
                selection_seed=1,
                extras={},
            )
        )


def test_browsecomp_csv_validation_fails_closed_on_sha_schema_and_row_count(
    tmp_path: Path,
) -> None:
    valid_bytes = _encrypted_csv_bytes()
    plugin = BrowseCompBenchmark(_make_config(tmp_path, raw_bytes=valid_bytes))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        plugin._parse_validated_rows(b"not the pinned csv")

    wrong_schema_bytes = b"problem,answer,canary\nq,a,c\n"
    wrong_schema = BrowseCompBenchmark(
        _make_config(tmp_path, raw_bytes=wrong_schema_bytes)
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        wrong_schema._parse_validated_rows(wrong_schema_bytes)

    wrong_count = BrowseCompBenchmark(
        _make_config(tmp_path, raw_bytes=valid_bytes, expected_row_count=2)
    )
    with pytest.raises(ValueError, match="row-count mismatch"):
        wrong_count._parse_validated_rows(valid_bytes)


def test_decrypt_round_trips_browsecomp_canary_ciphertext() -> None:
    ciphertext = _encrypt("Unicode answer: Café Mars", "canary-123")

    assert decrypt(ciphertext, "canary-123") == "Unicode answer: Café Mars"


def test_load_tasks_decrypts_and_normalizes_official_csv_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_bytes = _encrypted_csv_bytes(
        question="What city hosts the Eiffel Tower?",
        answer="Paris",
        topic="landmarks",
        canary="canary-city",
    )
    monkeypatch.setattr(
        BrowseCompBenchmark,
        "_fetch_csv_bytes",
        lambda self: raw_bytes,
    )
    plugin = BrowseCompBenchmark(_make_config(tmp_path, raw_bytes=raw_bytes))

    [task] = plugin.load_tasks()

    assert task == {
        "instance_id": "browsecomp-0",
        "problem_statement": "What city hosts the Eiffel Tower?",
        "reference_answer": "Paris",
        "problem_topic": "landmarks",
        "task_source_kind": "browsecomp_official_csv",
        "task_source_id": "0",
        "task_source_path": "https://example.invalid/browsecomp.csv",
        "task_source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "task_source_schema": list(_SCHEMA),
        "task_source_row_count": 1,
        "repo": None,
        "image_name": None,
        "docker_image": None,
    }


def test_load_tasks_ignores_local_json_cache_that_would_bypass_pinned_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "_row_index": 999,
                    "problem": "LOCAL UNENCRYPTED QUESTION",
                    "answer": "LOCAL SECRET ANSWER",
                    "problem_topic": "local-topic",
                    "canary": "not-used",
                }
            ]
        ),
        encoding="utf-8",
    )
    raw_bytes = _encrypted_csv_bytes(question="Pinned CSV question", answer="Pinned answer")
    monkeypatch.setenv("AGENT_SCHED_BENCH_USE_LOCAL_TASK_CACHE", "1")
    monkeypatch.setattr(BrowseCompBenchmark, "_fetch_csv_bytes", lambda self: raw_bytes)
    plugin = BrowseCompBenchmark(_make_config(tmp_path, raw_bytes=raw_bytes))

    [task] = plugin.load_tasks()

    assert task["problem_statement"] == "Pinned CSV question"
    assert task["reference_answer"] == "Pinned answer"
    assert task["task_source_kind"] == "browsecomp_official_csv"


def test_browsecomp_agent_prompt_renders_question_without_reference_or_topic() -> None:
    task = {
        "problem_statement": "QUESTION_SENTINEL public problem",
        "reference_answer": "REFERENCE_SENTINEL private answer",
        "problem_topic": "TOPIC_SENTINEL private topic",
    }

    rendered = render_prompt(
        load_prompt_template("default", "browsecomp"),
        task["problem_statement"],
    )

    assert "QUESTION_SENTINEL public problem" in rendered
    assert "REFERENCE_SENTINEL" not in rendered
    assert "TOPIC_SENTINEL" not in rendered
