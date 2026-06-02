"""Modal post-hoc segment-grid plotting for uploaded HF recordings.

Benchmark-agnostic: any target in `SOURCE_PREFIXES` (swe-rebench or
terminal-bench tasks) is rendered by the same pipeline. Two figure modes share
one downloaded copy of the recordings:

  - `sparse`     -> `plot_sparse_segment_grid.build_sparse_segment_grids`
                    (sparse-filtered retained attention share / ratio)
  - `head_span`  -> `plot_head_span_grid.build_head_span_segment_grids`
                    (within-segment attention mean / std)

It downloads only the files the renderer needs, renders on Modal, then uploads
plotting outputs back to the HF dataset under a per-mode analysis prefix. It
never reruns inference or modifies source recording artifacts.

The renderers themselves are Modal-free and run anywhere the recordings already
live (e.g. the remote GPU server); this module is only the Modal venue.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal


APP_NAME = "asb-segment-grid-plot"
VOLUME_NAME = "asb-terminal-recordings"
HF_SECRET_NAME = "asb-hf-token"
HF_REPO_ID = "HCAHOI/agent-sched-bench-recordings"
HF_REPO_TYPE = "dataset"
OUTPUT_DATE = "20260601"
DEFAULT_TARGET = "swe-rebench"
DEFAULT_MODE = "sparse"
# Per-mode HF analysis-prefix roots. Both modes consume the same downloaded
# recordings; only the output location and the renderer differ.
MODE_OUTPUT_ROOTS = {
    "sparse": "analysis/sparse-grid-polished-iter-axis",
    "head_span": "analysis/head-span-grid",
}
SOURCE_PREFIXES = {
    "swe-rebench": (
        "openclaw-Qwen-Qwen3-Coder-30B-A3B-Instruct-FP8-swe-rebench/"
        "0b01001001__spectree-64/20260531T142639"
    ),
    "fix-git": (
        "openclaw-Qwen-Qwen3-Coder-30B-A3B-Instruct-FP8-terminal-bench/"
        "fix-git/20260531T094138"
    ),
    "causal-inference-r": (
        "openclaw-Qwen-Qwen3-Coder-30B-A3B-Instruct-FP8-terminal-bench/"
        "causal-inference-r/20260531T111625"
    ),
}

VOLUME_ROOT = Path("/data")
NEEDED_SUFFIXES = (
    ".done",
    "attention.npz",
    "routing.npz",
    "segments.json",
    "sparse_attention.npz",
)
NEEDED_EXACT = ("recordings/meta.json",)

LOCAL_FILE = Path(__file__).resolve()
LOCAL_SCRIPTS_DIR = (
    LOCAL_FILE.parents[2] / "scripts"
    if len(LOCAL_FILE.parents) > 2
    else Path("/opt/scripts")
)
SCRIPTS_DIR = LOCAL_SCRIPTS_DIR if LOCAL_SCRIPTS_DIR.exists() else Path("/opt/scripts")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates")
    .pip_install("huggingface_hub", "matplotlib", "numpy")
    .add_local_dir(SCRIPTS_DIR, remote_path="/opt/scripts", copy=True)
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
app = modal.App(APP_NAME)


@dataclass(frozen=True)
class TargetConfig:
    target: str
    mode: str
    source_prefix: str
    output_prefix: str
    source_namespace: str
    task_id: str
    output_date: str
    run_label: str
    hf_cache_dir: Path
    extract_dir: Path
    attempt_dir: Path
    output_dir: Path


def _target_config(target: str, mode: str = DEFAULT_MODE) -> TargetConfig:
    if target not in SOURCE_PREFIXES:
        choices = ", ".join(sorted(SOURCE_PREFIXES))
        raise ValueError(f"unknown target {target!r}; expected one of: {choices}")
    if mode not in MODE_OUTPUT_ROOTS:
        choices = ", ".join(sorted(MODE_OUTPUT_ROOTS))
        raise ValueError(f"unknown mode {mode!r}; expected one of: {choices}")
    source_prefix = SOURCE_PREFIXES[target]
    source_parts = source_prefix.split("/")
    if len(source_parts) < 3:
        raise ValueError(f"source prefix is too short: {source_prefix}")
    source_namespace = source_parts[0]
    task_id = source_parts[-2]
    output_prefix = (
        f"{MODE_OUTPUT_ROOTS[mode]}/{source_namespace}/{task_id}/{OUTPUT_DATE}"
    )
    # Input cache is mode-independent so prepare downloads once and both modes
    # reuse it; only the output dir is namespaced by mode to avoid collisions.
    run_label = (
        f"{_safe_run_component(target)}-{_safe_run_component(task_id)}-"
        f"grid-iter-{OUTPUT_DATE}"
    )
    extract_dir = VOLUME_ROOT / "extracted" / run_label
    return TargetConfig(
        target=target,
        mode=mode,
        source_prefix=source_prefix,
        output_prefix=output_prefix,
        source_namespace=source_namespace,
        task_id=task_id,
        output_date=OUTPUT_DATE,
        run_label=run_label,
        hf_cache_dir=VOLUME_ROOT / "hf-prefix-cache" / run_label,
        extract_dir=extract_dir,
        attempt_dir=extract_dir / task_id / "attempt_1",
        output_dir=VOLUME_ROOT / "outputs" / run_label / mode,
    )


def _safe_run_component(value: str) -> str:
    return value.replace("/", "-").replace("__", "-")


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=[secret],
    cpu=4,
    memory=32768,
    timeout=60 * 60 * 2,
)
def prepare_from_hf(
    target: str = DEFAULT_TARGET,
    mode: str = DEFAULT_MODE,
    force: bool = False,
) -> dict[str, Any]:
    return _prepare_from_hf(_target_config(target, mode), force=force)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=[secret],
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 2,
)
def render_and_upload(
    target: str = DEFAULT_TARGET,
    mode: str = DEFAULT_MODE,
    force: bool = False,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    config = _target_config(target, mode)
    token = os.environ["HF_TOKEN"]
    prepared = _prepare_from_hf(config, force=force)
    if force and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    if config.output_dir.exists():
        raise FileExistsError(f"{config.output_dir} already exists; pass force=True")

    sys.path.insert(0, "/opt")
    import matplotlib

    matplotlib.use("Agg")
    summary = _render_grid(config)
    (config.output_dir / "modal_source.json").write_text(
        json.dumps(
            {
                "target": config.target,
                "mode": config.mode,
                "source_repo_id": HF_REPO_ID,
                "source_prefix": config.source_prefix,
                "output_prefix": config.output_prefix,
                "prepared": prepared,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    api = HfApi(token=token)
    info = api.upload_folder(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        folder_path=str(config.output_dir),
        path_in_repo=config.output_prefix,
        token=token,
        commit_message=f"add iter-axis {config.mode} grid plot for {config.target}",
    )
    volume.commit()
    return {
        "target": config.target,
        "mode": config.mode,
        "output_dir": str(config.output_dir),
        "hf_output_prefix": config.output_prefix,
        "commit_url": info.commit_url,
        "summary": summary,
        **_output_stats(config),
    }


def _render_grid(config: TargetConfig) -> dict[str, Any]:
    """Dispatch to the Modal-free renderer for the configured mode."""
    if config.mode == "sparse":
        from scripts.recoding_figures.plot_sparse_segment_grid import (
            build_sparse_segment_grids,
        )

        return build_sparse_segment_grids(
            inputs=[config.attempt_dir],
            output_dir=config.output_dir,
            include_orphans=False,
            split_by_task=True,
        )
    if config.mode == "head_span":
        from scripts.recoding_figures.plot_head_span_grid import (
            build_head_span_segment_grids,
        )

        return build_head_span_segment_grids(
            inputs=[config.attempt_dir],
            output_dir=config.output_dir,
            include_orphans=False,
            split_by_task=True,
        )
    raise ValueError(f"unknown mode {config.mode!r}")


@app.local_entrypoint()
def main(
    action: str = "all",
    target: str = DEFAULT_TARGET,
    mode: str = DEFAULT_MODE,
    force: bool = False,
) -> None:
    """Run `prepare`, `render`, or `all` for a configured target and mode."""
    if action not in {"prepare", "render", "all"}:
        raise ValueError("action must be one of: prepare, render, all")
    _target_config(target, mode)
    if action in {"prepare", "all"}:
        print(
            json.dumps(
                prepare_from_hf.remote(target=target, mode=mode, force=force), indent=2
            )
        )
    if action in {"render", "all"}:
        result = render_and_upload.remote(target=target, mode=mode, force=force)
        print(json.dumps(result, indent=2))


def _prepare_from_hf(config: TargetConfig, *, force: bool) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    _validate_run_identity(config)
    token = os.environ["HF_TOKEN"]
    marker = config.extract_dir / ".complete"
    if force and config.extract_dir.exists():
        shutil.rmtree(config.extract_dir)
    if marker.exists() and not force:
        _validate_prepare_marker(config, marker)
        return {"status": "already_prepared", **_artifact_stats(config)}

    api = HfApi(token=token)
    files = api.list_repo_files(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        token=token,
    )
    prefix = f"{config.source_prefix}/"
    wanted = [
        path
        for path in sorted(files)
        if path.startswith(prefix) and _is_needed_source_file(path[len(prefix) :])
    ]
    if not wanted:
        raise FileNotFoundError(f"no needed files under {HF_REPO_ID}:{config.source_prefix}")

    config.attempt_dir.mkdir(parents=True, exist_ok=True)
    config.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    copied_bytes = 0
    for repo_path in wanted:
        downloaded = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                filename=repo_path,
                token=token,
                local_dir=str(config.hf_cache_dir),
            )
        )
        relative = Path(repo_path[len(prefix) :])
        if relative.parts[:1] and relative.parts[0].startswith("attempt_"):
            relative = Path(*relative.parts[1:])
        dest = config.attempt_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, dest)
        copied += 1
        copied_bytes += dest.stat().st_size
        if copied % 25 == 0:
            print(f"{config.target}: copied {copied}/{len(wanted)} files", flush=True)

    marker.write_text(
        json.dumps(
            {
                "repo_id": HF_REPO_ID,
                "repo_type": HF_REPO_TYPE,
                "source_prefix": config.source_prefix,
                "task_id": config.task_id,
                "files": copied,
                "bytes": copied_bytes,
                "needed_suffixes": list(NEEDED_SUFFIXES),
                "needed_exact": list(NEEDED_EXACT),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stats = _artifact_stats(config)
    if stats["complete_iters"] == 0:
        raise ValueError("downloaded artifact has no complete recording iterations")
    volume.commit()
    return {"status": "prepared", **stats}


def _validate_run_identity(config: TargetConfig) -> None:
    output_parts = config.output_prefix.split("/")
    if len(output_parts) < 4:
        raise ValueError(f"output prefix is too short: {config.output_prefix}")
    if output_parts[-3] != config.source_namespace:
        raise ValueError(
            "output namespace does not match source namespace: "
            f"{output_parts[-3]} != {config.source_namespace}"
        )
    if output_parts[-2] != config.task_id:
        raise ValueError(
            "output task does not match source task: "
            f"{output_parts[-2]} != {config.task_id}"
        )
    if config.output_date not in config.run_label or config.task_id.split("__")[0] not in config.run_label:
        raise ValueError(f"run label is inconsistent with task/date: {config.run_label}")


def _validate_prepare_marker(config: TargetConfig, marker: Path) -> None:
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "repo_id": HF_REPO_ID,
        "repo_type": HF_REPO_TYPE,
        "source_prefix": config.source_prefix,
        "task_id": config.task_id,
        "needed_suffixes": list(NEEDED_SUFFIXES),
        "needed_exact": list(NEEDED_EXACT),
    }
    mismatches = {
        key: {"expected": expected_value, "actual": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"{marker} was prepared from a different source; "
            f"pass force=True to refresh it. Mismatches: {mismatches}"
        )


def _is_needed_source_file(relative: str) -> bool:
    return relative in NEEDED_EXACT or relative.endswith(NEEDED_SUFFIXES)


def _artifact_stats(config: TargetConfig) -> dict[str, Any]:
    recordings = config.attempt_dir / "recordings"
    iter_dirs = list(recordings.glob("iter_*"))
    return {
        "attempt_dir": str(config.attempt_dir),
        "complete_iters": sum(1 for path in iter_dirs if (path / ".done").is_file()),
        "attention_npz": len(list(recordings.glob("iter_*/attention.npz"))),
        "sparse_attention_npz": len(list(recordings.glob("iter_*/sparse_attention.npz"))),
        "routing_npz": len(list(recordings.glob("iter_*/routing.npz"))),
        "segments_json": len(list(recordings.glob("iter_*/segments.json"))),
        "meta_json": (recordings / "meta.json").is_file(),
    }


def _output_stats(config: TargetConfig) -> dict[str, Any]:
    files = [path for path in config.output_dir.rglob("*") if path.is_file()]
    return {
        "output_files": len(files),
        "output_bytes": sum(path.stat().st_size for path in files),
    }
