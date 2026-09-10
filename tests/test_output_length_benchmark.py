from __future__ import annotations

import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import httpx
import pytest
import torch
from torch import nn

from scripts.evaluation.output_length_benchmark import (
    evaluate_predictions,
    export_dataset,
    label_dataset,
)
from scripts.evaluation.output_length_predictors import (
    _build_parser,
    _fit_logt_mle,
    _point_examples,
    _run_egtp,
    _tie_examples,
    run_egtp_static,
    run_outlets_static,
    run_ssjf_reg,
    run_tie,
)


def _write_trace(
    path: Path,
    session_id: str,
    lengths: tuple[int, ...],
    *,
    model_key: str = "model",
) -> None:
    rows: list[dict[str, object]] = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "instance_id": session_id,
            model_key: "source-model",
        }
    ]
    for index, length in enumerate(lengths):
        rows.append(
            {
                "type": "action",
                "action_type": "llm_call",
                "action_id": f"llm_{index}",
                "agent_id": session_id,
                "iteration": index,
                "ts_start": float(index),
                "ts_end": float(index + 1),
                "data": {
                    "messages_in": [{"role": "user", "content": f"step {index}"}],
                    "completion_tokens": length,
                    "raw_response": {
                        "choices": [{"finish_reason": "stop"}],
                    },
                },
            }
        )
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _export_two_sessions(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    _write_trace(source / "one" / "trace.jsonl", "one", (10, 20))
    _write_trace(source / "two" / "trace.jsonl", "two", (30, 40))
    dataset = tmp_path / "dataset"
    export_dataset(
        [source],
        dataset,
        request_options_path=None,
        seed=42,
        train_fraction=0.5,
        validation_fraction=0.0,
    )
    return dataset


def _export_three_sessions(tmp_path: Path) -> Path:
    source = tmp_path / "source-three"
    for index, session in enumerate(("one", "two", "three"), start=1):
        _write_trace(
            source / session / "trace.jsonl",
            session,
            (index * 10, index * 10 + 5),
        )
    dataset = tmp_path / "dataset-three"
    export_dataset(
        [source],
        dataset,
        request_options_path=None,
        seed=42,
        train_fraction=0.34,
        validation_fraction=0.34,
    )
    return dataset


def _write_natural_labels(
    dataset: Path,
    output: Path,
    *,
    draws: int = 1,
    temperature: float = 0.7,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    output.mkdir()
    if rows is None:
        source = [
            json.loads(line)
            for line in (dataset / "source_labels.jsonl").read_text().splitlines()
        ]
        rows = [
            {
                **label,
                "draw_id": draw_id,
                "actual_tokens": label["actual_tokens"] + draw_id,
            }
            for label in source
            for draw_id in range(draws)
        ]
    rows = [
        {
            **row,
            "schema_version": 1,
            "label_source": "target_natural_generation",
            "finish_reason": "stop",
        }
        for row in rows
    ]
    labels = output / "labels.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))
    protocol = {
        "schema_version": 1,
        "dataset_dir": str(dataset.resolve()),
        "dataset_protocol": json.loads((dataset / "dataset.json").read_text()),
        "api_base": "http://localhost/v1",
        "model": "target-model",
        "api_key_env": "",
        "max_tokens": 4096,
        "temperature": temperature,
        "top_p": None,
        "seed": 42,
        "draws": draws,
        "splits": ["train", "validation", "test"],
        "natural_termination_required": True,
    }
    (output / "protocol.json").write_text(json.dumps(protocol))
    return labels


def test_export_splits_whole_sessions(tmp_path: Path) -> None:
    dataset = _export_two_sessions(tmp_path)
    rows = [
        json.loads(line)
        for line in (dataset / "prefixes.jsonl").read_text().splitlines()
    ]
    split_by_session: dict[str, set[str]] = {}
    for row in rows:
        split_by_session.setdefault(row["session_id"], set()).add(row["split"])
    assert {session: len(splits) for session, splits in split_by_session.items()} == {
        "one": 1,
        "two": 1,
    }
    assert {row["split"] for row in rows} == {"train", "test"}
    assert len((dataset / "source_labels.jsonl").read_text().splitlines()) == 4
    protocol = json.loads((dataset / "dataset.json").read_text())
    assert protocol["prediction_contract"]["required_fields"] == [
        "sample_id",
        "predicted_tokens",
    ]


def test_export_accepts_replay_source_model_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_trace(source / "trace.jsonl", "one", (10,), model_key="source_model")
    dataset = tmp_path / "dataset"
    export_dataset(
        [source],
        dataset,
        request_options_path=None,
        seed=42,
        train_fraction=0.0,
        validation_fraction=0.0,
    )
    row = json.loads((dataset / "prefixes.jsonl").read_text().splitlines()[0])
    assert row["source_model"] == "source-model"


def test_export_samples_flat_trace_and_counts_invalid_source_action(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    trace_path = source / "bench__one.trace.jsonl"
    _write_trace(trace_path, "one", (10, 20, 30, 40, 50, 60, 70))
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    rows[4]["data"]["completion_tokens"] = 0
    rows[0]["benchmark"] = "bench"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = tmp_path / "dataset"
    protocol = export_dataset(
        [source],
        dataset,
        request_options_path=None,
        seed=42,
        train_fraction=0.0,
        validation_fraction=0.0,
        steps_per_session=5,
        skip_invalid_source_actions=True,
    )

    prefixes = [
        json.loads(line)
        for line in (dataset / "prefixes.jsonl").read_text().splitlines()
    ]
    assert [row["action_id"] for row in prefixes] == [
        "llm_0",
        "llm_2",
        "llm_4",
        "llm_6",
    ]
    assert {row["benchmark"] for row in prefixes} == {"bench"}
    assert protocol["sampling"]["excluded_invalid_source_actions"] == 1
    assert protocol["counts"]["trajectories_by_benchmark"] == {"bench": 1}


def test_export_accepts_flat_trace_in_archive(tmp_path: Path) -> None:
    source = tmp_path / "source" / "bench__one.trace.jsonl"
    _write_trace(source, "one", (10, 20))
    archive = tmp_path / "traces.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(source, arcname="flat/bench__one.trace.jsonl")

    protocol = export_dataset(
        [archive],
        tmp_path / "dataset",
        request_options_path=None,
        seed=42,
        train_fraction=0.0,
        validation_fraction=0.0,
    )

    assert protocol["counts"]["samples"] == 2


def test_label_uses_natural_completion_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _export_two_sessions(tmp_path)
    protocol_path = dataset / "dataset.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["request_options"] = {
        "tools": [{"type": "function", "function": {"name": "shell"}}],
        "tool_choice": "auto",
    }
    protocol_path.write_text(json.dumps(protocol))
    calls: list[dict[str, object]] = []

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, _endpoint: str, *, json: dict[str, object]) -> httpx.Response:
            calls.append(json)
            return httpx.Response(
                200,
                request=httpx.Request("POST", "http://localhost/v1/chat/completions"),
                json={
                    "id": "response",
                    "model": "target-model",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "done"}}
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                },
            )

    monkeypatch.setattr(
        "scripts.evaluation.output_length_benchmark.httpx.Client",
        lambda **_kwargs: FakeClient(),
    )
    output = tmp_path / "labels"
    kwargs = {
        "api_base": "http://localhost/v1",
        "model": "target-model",
        "api_key_env": "",
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": None,
        "seed": 1,
        "draws": 2,
        "splits": ("test",),
        "timeout_s": 10.0,
    }
    result = label_dataset(dataset, output, **kwargs)
    first_call_count = len(calls)
    assert result["completed_labels"] == 4
    assert first_call_count == 4
    assert calls[0]["max_tokens"] == 100
    assert "ignore_eos" not in calls[0]
    assert calls[0]["tool_choice"] == "auto"

    label_dataset(dataset, output, **kwargs)
    assert len(calls) == first_call_count

    protocol["request_options"]["tool_choice"] = "none"
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="protocol does not match"):
        label_dataset(dataset, output, **kwargs)


def test_label_parallelizes_draws_after_one_prefix_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _export_two_sessions(tmp_path)
    active = 0
    maximum_active = 0
    lock = __import__("threading").Lock()

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, _endpoint: str, *, json: dict[str, object]) -> httpx.Response:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            __import__("time").sleep(0.01)
            with lock:
                active -= 1
            return httpx.Response(
                200,
                request=httpx.Request("POST", "http://localhost/v1/chat/completions"),
                json={
                    "id": f"response-{json['seed']}",
                    "model": "target-model",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "ok"}}
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                },
            )

    monkeypatch.setattr(
        "scripts.evaluation.output_length_benchmark.httpx.Client",
        lambda **_kwargs: FakeClient(),
    )
    result = label_dataset(
        dataset,
        tmp_path / "parallel-labels",
        api_base="http://localhost/v1",
        model="target-model",
        api_key_env="",
        max_tokens=100,
        temperature=0.7,
        top_p=0.8,
        seed=42,
        draws=3,
        splits=("test",),
        timeout_s=10,
        concurrency=2,
    )
    assert result["completed_labels"] == 6
    assert maximum_active == 2


def test_evaluate_requires_identical_ids_and_computes_q_error(tmp_path: Path) -> None:
    dataset = _export_two_sessions(tmp_path)
    labels_path = _write_natural_labels(dataset, tmp_path / "natural-labels")
    prefixes = [
        json.loads(line)
        for line in (dataset / "prefixes.jsonl").read_text().splitlines()
    ]
    test_ids = {row["sample_id"] for row in prefixes if row["split"] == "test"}
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    actual_by_id = {
        row["sample_id"]: row["actual_tokens"]
        for row in labels
        if row["sample_id"] in test_ids
    }
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "predicted_tokens": actual * 2}) + "\n"
            for sample_id, actual in actual_by_id.items()
        ),
        encoding="utf-8",
    )
    result = evaluate_predictions(
        dataset,
        labels_path,
        [("double", predictions)],
        split="test",
    )
    metrics = result["methods"]["double"]
    assert metrics["q_error"] == {
        "q50": 2.0,
        "q90": 2.0,
        "q95": 2.0,
        "q99": 2.0,
        "mean": 2.0,
    }
    assert metrics["mean_accuracy"] == 0.5

    predictions.write_text(
        json.dumps(
            {
                "sample_id": next(iter(test_ids)),
                "predicted_tokens": actual_by_id[next(iter(test_ids))],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_predictions(
            dataset,
            labels_path,
            [("incomplete", predictions)],
            split="test",
        )


def test_evaluate_rejects_uneven_draw_coverage(tmp_path: Path) -> None:
    dataset = _export_two_sessions(tmp_path)
    prefixes = [
        json.loads(line)
        for line in (dataset / "prefixes.jsonl").read_text().splitlines()
    ]
    all_ids = [row["sample_id"] for row in prefixes]
    test_ids = [row["sample_id"] for row in prefixes if row["split"] == "test"]
    labels = _write_natural_labels(
        dataset,
        tmp_path / "uneven-labels",
        draws=2,
        rows=[
            {"sample_id": sample_id, "draw_id": 0, "actual_tokens": 10}
            for sample_id in all_ids
        ]
        + [{"sample_id": test_ids[0], "draw_id": 1, "actual_tokens": 11}],
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "predicted_tokens": 10}) + "\n"
            for sample_id in test_ids
        )
    )
    with pytest.raises(ValueError, match="coverage"):
        evaluate_predictions(
            dataset,
            labels,
            [("method", predictions)],
            split="test",
        )
    with pytest.raises(ValueError, match="coverage"):
        _point_examples(dataset, labels)


def test_formal_consumers_reject_recorded_trace_labels(tmp_path: Path) -> None:
    dataset = _export_two_sessions(tmp_path)
    with pytest.raises(ValueError, match="natural-label protocol is missing"):
        evaluate_predictions(
            dataset,
            dataset / "source_labels.jsonl",
            [],
            split="test",
        )


def test_export_rejects_multi_choice_and_second_token_cap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_trace(source / "trace.jsonl", "one", (10,))
    for key in ("n", "max_completion_tokens"):
        options = tmp_path / f"{key}.json"
        options.write_text(json.dumps({key: 2}))
        with pytest.raises(ValueError, match="controlled keys"):
            export_dataset(
                [source],
                tmp_path / f"dataset-{key}",
                request_options_path=options,
                seed=42,
                train_fraction=0.0,
                validation_fraction=0.0,
            )


def test_label_rejects_censored_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _export_two_sessions(tmp_path)

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs) -> httpx.Response:
            return httpx.Response(
                200,
                request=httpx.Request("POST", "http://localhost/v1/chat/completions"),
                json={
                    "choices": [{"finish_reason": "length", "message": {}}],
                    "usage": {"completion_tokens": 10},
                },
            )

    monkeypatch.setattr(
        "scripts.evaluation.output_length_benchmark.httpx.Client",
        lambda **_kwargs: FakeClient(),
    )
    with pytest.raises(RuntimeError, match="hit max_tokens"):
        label_dataset(
            dataset,
            tmp_path / "labels",
            api_base="http://localhost/v1",
            model="target-model",
            api_key_env="",
            max_tokens=10,
            temperature=0.0,
            top_p=None,
            seed=1,
            draws=1,
            splits=("test",),
            timeout_s=10.0,
        )


def test_ssjf_reg_emits_common_prediction_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _export_two_sessions(tmp_path)
    labels_path = _write_natural_labels(dataset, tmp_path / "ssjf-labels")

    class FakeTokenizer:
        truncation_side = "right"
        calls: list[dict[str, object]] = []

        def __call__(self, texts, **kwargs):
            self.calls.append(kwargs)
            size = len(texts)
            return {
                "input_ids": torch.arange(1, 4).repeat(size, 1),
                "attention_mask": torch.ones((size, 3), dtype=torch.long),
            }

    class FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = nn.Embedding(8, 4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    tokenizer = FakeTokenizer()
    encoders: list[FakeEncoder] = []

    def fake_encoder(_name: str) -> FakeEncoder:
        encoder = FakeEncoder()
        encoders.append(encoder)
        return encoder

    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors.AutoTokenizer.from_pretrained",
        lambda _name: tokenizer,
    )
    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors.AutoModel.from_pretrained",
        fake_encoder,
    )
    output = tmp_path / "ssjf"
    protocol = run_ssjf_reg(
        dataset,
        labels_path,
        output,
        encoder="fake",
        epochs=4,
        batch_size=2,
        learning_rate=1e-3,
        seed=42,
        device_name="cpu",
    )
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert protocol["method"] == "ssjf-reg"
    assert protocol["input"].endswith("left_truncate_to_512_tokens")
    assert protocol["prediction_postprocess"] == "clamp_to_at_least_one_token"
    assert len(protocol["train_loss"]) == 4
    assert tokenizer.truncation_side == "left"
    assert all(call["max_length"] == 512 for call in tokenizer.calls)
    assert all(call["truncation"] is True for call in tokenizer.calls)
    assert len(encoders) == 1
    assert not any(parameter.requires_grad for parameter in encoders[0].parameters())
    assert len(predictions) == 2
    assert all(row["predicted_tokens"] > 0 for row in predictions)

    args = _build_parser().parse_args(
        [
            "ssjf-reg",
            "--dataset-dir",
            "dataset",
            "--labels",
            "labels",
            "--output-dir",
            "output",
        ]
    )
    assert (args.epochs, args.batch_size, args.learning_rate) == (6, 16, 1e-5)


def test_egtp_static_adapts_official_predictor_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _export_two_sessions(tmp_path)
    labels_path = _write_natural_labels(dataset, tmp_path / "egtp-labels")
    invoked: list[dict[str, object]] = []

    class FakeChatTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 1
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return f"rendered {messages[0]['content']}"

    def fake_run(
        upstream_dir: Path,
        python: Path,
        data_dir: Path,
        output_dir: Path,
        options: dict[str, object],
    ) -> None:
        import csv

        with (data_dir / "test.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        assert all(
            row["user_prompt_content"].startswith("rendered step ") for row in rows
        )
        invoked.append(
            {"upstream_dir": upstream_dir, "python": python, "options": options}
        )
        output_dir.mkdir(parents=True)
        (output_dir / "test_predictions.csv").write_text(
            "sample_idx,pred_length\n"
            + "".join(f"{index},{index + 11}\n" for index in range(len(rows)))
        )

    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors._run_egtp",
        fake_run,
    )
    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors.AutoTokenizer.from_pretrained",
        lambda _name: FakeChatTokenizer(),
    )
    output = tmp_path / "egtp"
    protocol = run_egtp_static(
        dataset,
        labels_path,
        output,
        tmp_path / "official-egtp",
        upstream_python=Path("/fake/python"),
        model_id="target-model",
        prompt_prefix_k=4,
        num_bins=20,
        lambda_val=0.95,
        epochs=200,
        batch_size=256,
        learning_rate=2e-5,
        extractor_batch_size=64,
        torch_dtype="bfloat16",
        seed=42,
    )
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert protocol["method"] == "egtp-static"
    assert protocol["input"].startswith("full_messages_rendered_by_target_tokenizer")
    assert "first 4 target-model tokens" in protocol["input"]
    assert protocol["unused_split"] == "validation"
    assert invoked[0]["options"]["prompt_prefix_k"] == 4
    assert [row["predicted_tokens"] for row in predictions] == [11.0, 12.0]

    args = _build_parser().parse_args(
        [
            "egtp-static",
            "--dataset-dir",
            "dataset",
            "--labels",
            "labels",
            "--output-dir",
            "output",
            "--upstream-dir",
            "upstream",
            "--model-id",
            "model",
        ]
    )
    assert (args.prompt_prefix_k, args.num_bins, args.epochs) == (4, 20, 200)


def test_egtp_runner_resolves_relative_subprocess_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = tmp_path / "upstream" / "EGTP" / "main.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "out = Path(sys.argv[sys.argv.index('--output_dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'ran').write_text('ok')\n"
    )
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors._verify_upstream",
        lambda *_args: None,
    )
    _run_egtp(
        Path("upstream"),
        Path(sys.executable),
        Path("data"),
        Path("output"),
        {},
    )
    assert (tmp_path / "output" / "ran").read_text() == "ok"


def test_outlets_static_adapts_official_checkpoint_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _export_two_sessions(tmp_path)
    invoked: list[dict[str, object]] = []

    def fake_run(
        upstream_dir: Path,
        python: Path,
        inputs_path: Path,
        output_path: Path,
        **kwargs: object,
    ) -> None:
        inputs = [json.loads(line) for line in inputs_path.read_text().splitlines()]
        assert all(row["prompt"].startswith("[") for row in inputs)
        invoked.append(
            {"upstream_dir": upstream_dir, "python": python, "kwargs": kwargs}
        )
        output_path.write_text(
            "".join(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "predicted_tokens": index + 21,
                    }
                )
                + "\n"
                for index, row in enumerate(inputs)
            )
        )

    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors._run_outlets", fake_run
    )
    output = tmp_path / "outlets"
    protocol = run_outlets_static(
        dataset,
        output,
        tmp_path / "official-outlets",
        upstream_python=Path("/fake/python"),
        checkpoint_path=tmp_path / "checkpoint",
        base_model_path="target-model",
        config_path=tmp_path / "config.json",
        device="cuda:0",
    )
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert protocol["method"] == "outlets-static"
    assert protocol["checkpoint_provenance"].startswith("official_OUTLETS")
    assert invoked[0]["kwargs"]["base_model_path"] == "target-model"
    assert [row["predicted_tokens"] for row in predictions] == [21, 22]

    args = _build_parser().parse_args(
        [
            "outlets-static",
            "--dataset-dir",
            "dataset",
            "--output-dir",
            "output",
            "--upstream-dir",
            "upstream",
            "--checkpoint-path",
            "checkpoint",
            "--base-model-path",
            "model",
            "--config-path",
            "config",
        ]
    )
    assert args.device == "cuda"


def test_outlets_worker_resolves_paths_and_preserves_official_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.evaluation.output_length_predictors import _run_outlets

    package = tmp_path / "upstream" / "outlets"
    package.mkdir(parents=True)
    (package / "inference_length.py").write_text(
        """
class LengthPredictor:
    def __init__(self, **kwargs):
        assert kwargs["normalize_length"] is True
        assert kwargs["adapter"] is True
        assert kwargs["use_target_model"] is True

    def predict(self, prompt):
        return {"predicted_length": len(prompt)}
"""
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.bin").touch()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "inputs.jsonl").write_text(
        json.dumps({"sample_id": "sample", "prompt": "hello"}) + "\n"
    )
    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors._verify_upstream",
        lambda *_args: None,
    )
    monkeypatch.chdir(tmp_path)
    _run_outlets(
        Path("upstream"),
        Path(sys.executable),
        Path("inputs.jsonl"),
        Path("predictions.jsonl"),
        checkpoint_path=Path("checkpoint"),
        base_model_path="model-id",
        config_path=Path("config.json"),
        device="cpu",
    )
    assert json.loads((tmp_path / "predictions.jsonl").read_text()) == {
        "sample_id": "sample",
        "predicted_tokens": 5,
    }


def test_tie_trains_fixed_logt_and_emits_point_and_distribution_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _export_three_sessions(tmp_path)
    source_labels = [
        json.loads(line)
        for line in (dataset / "source_labels.jsonl").read_text().splitlines()
    ]
    labels_path = _write_natural_labels(
        dataset,
        tmp_path / "tie-labels",
        draws=2,
        rows=[
            row
            for label in source_labels
            for row in (
                label,
                {**label, "draw_id": 1, "actual_tokens": label["actual_tokens"] + 2},
            )
        ],
    )
    examples, draw_ids = _tie_examples(dataset, labels_path)
    assert draw_ids == [0, 1]
    assert all(row["sigma"] > 0 for rows in examples.values() for row in rows)

    class FakeTokenizer:
        truncation_side = "left"
        calls: list[dict[str, object]] = []

        def __call__(self, texts, **kwargs):
            self.calls.append(kwargs)
            size = len(texts)
            return {
                "input_ids": torch.arange(1, 4).repeat(size, 1),
                "attention_mask": torch.ones((size, 3), dtype=torch.long),
            }

    class FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = nn.Embedding(8, 4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    tokenizer = FakeTokenizer()
    encoders: list[FakeEncoder] = []

    def fake_encoder(_name: str, **_kwargs: object) -> FakeEncoder:
        encoder = FakeEncoder()
        encoders.append(encoder)
        return encoder

    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors.AutoTokenizer.from_pretrained",
        lambda _name, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors.AutoModel.from_pretrained",
        fake_encoder,
    )
    monkeypatch.setattr(
        "scripts.evaluation.output_length_predictors._verify_upstream",
        lambda *_args: None,
    )
    output = tmp_path / "tie"
    protocol = run_tie(
        dataset,
        labels_path,
        output,
        tmp_path / "official-tie",
        encoder="fake",
        epochs=2,
        encoder_tuning_epochs=1,
        batch_size=2,
        learning_rate=2e-5,
        frozen_learning_rate=5e-5,
        seed=42,
        device_name="cpu",
    )
    point = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    distribution = [
        json.loads(line)
        for line in (output / "distribution_predictions.jsonl").read_text().splitlines()
    ]
    assert protocol["method"] == "tie-fixed-logt"
    assert protocol["draw_ids"] == [0, 1]
    assert protocol["point_prediction"] == "distribution_median_exp_mu"
    assert len(point) == len(distribution) == 2
    assert all(row["predicted_tokens"] > 0 for row in point)
    assert all(row["predicted_logt_sigma"] > 0 for row in distribution)
    assert protocol["target_fit"] == "joint_MLE_via_L-BFGS-B_on_log_token_lengths"
    assert len(encoders) == 1
    assert not any(parameter.requires_grad for parameter in encoders[0].parameters())
    assert tokenizer.truncation_side == "right"
    assert all(call["max_length"] == 512 for call in tokenizer.calls)
    assert all(call["padding"] == "max_length" for call in tokenizer.calls)

    args = _build_parser().parse_args(
        [
            "tie",
            "--dataset-dir",
            "dataset",
            "--labels",
            "labels",
            "--output-dir",
            "output",
            "--upstream-dir",
            "upstream",
            "--encoder",
            "encoder",
        ]
    )
    assert (args.epochs, args.encoder_tuning_epochs, args.batch_size) == (20, 12, 32)


def test_tie_logt_mle_and_zero_variance_boundary() -> None:
    mu, sigma = _fit_logt_mle([0.0, 1.0])
    assert mu == pytest.approx(0.5, abs=1e-6)
    assert sigma == pytest.approx(0.5, abs=1e-6)
    assert _fit_logt_mle([2.0, 2.0]) == (2.0, 1e-6)


def test_tie_rejects_deterministic_multi_draw_labels(tmp_path: Path) -> None:
    dataset = _export_three_sessions(tmp_path)
    labels = _write_natural_labels(
        dataset,
        tmp_path / "deterministic-labels",
        draws=2,
        temperature=0.0,
    )
    with pytest.raises(ValueError, match="stochastic natural-label draws"):
        _tie_examples(dataset, labels)
