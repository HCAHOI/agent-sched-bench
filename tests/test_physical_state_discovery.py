import json
from pathlib import Path
import subprocess

from scripts.evaluation import discover_physical_state_files as subject


def test_discovery_traces_target_in_pinned_image(tmp_path, monkeypatch) -> None:
    starts = []
    commands = []
    stops = []

    def start(image_id, task_id, purpose, executable, *, network):
        starts.append((image_id, task_id, purpose, executable, network))
        return "discovery-container", 4.0

    def run(argv, *, timeout, check=True):
        commands.append((argv, timeout, check))
        return subprocess.CompletedProcess(argv, 0, "passed", ""), 5.0

    def copy(_executable, source, destination):
        Path(destination).write_text(
            'openat(AT_FDCWD, "/testbed/x.py", O_RDONLY) = 3</testbed/x.py>\n',
            encoding="utf-8",
        )
        assert source == f"discovery-container:{subject.STRACE_CONTAINER_PATH}"
        return 2.0

    monkeypatch.setattr(subject, "_start", start)
    monkeypatch.setattr(
        subject,
        "_ensure_strace",
        lambda *_args: {"version": "strace 6", "install_ms": 0.0},
    )
    monkeypatch.setattr(subject, "_run", run)
    monkeypatch.setattr(subject, "_copy", copy)
    monkeypatch.setattr(
        subject, "_stop", lambda container_id, _executable: stops.append(container_id) or 3.0
    )

    result = subject._discover(
        task={
            "task_id": "task-a",
            "target_tool_args": {
                "command": "python -m pytest tests/test_x.py",
                "timeout": 300,
                "working_dir": "/testbed",
            },
        },
        prepared={"prepared_image_id": "sha256:" + "a" * 64},
        staging=tmp_path,
        executable="docker",
    )

    assert starts == [("sha256:" + "a" * 64, "task-a", "strace", "docker", "host")]
    assert commands[0][0] == [
        "docker",
        "exec",
        "-w",
        "/testbed",
        "discovery-container",
        "strace",
        "-f",
        "-qq",
        "-yy",
        "-e",
        "trace=%file",
        "-s",
        "4096",
        "-o",
        subject.STRACE_CONTAINER_PATH,
        "/bin/sh",
        "-c",
        "python -m pytest tests/test_x.py",
    ]
    assert commands[0][1] == 360
    assert stops == ["discovery-container"]
    assert result["resolved_path_count"] == 1


def test_filter_uses_fresh_networkless_pinned_image(tmp_path, monkeypatch) -> None:
    (tmp_path / "open-paths.json").write_text("{}", encoding="utf-8")
    starts = []
    stops = []

    def start(image_id, task_id, purpose, executable, *, network):
        starts.append((image_id, task_id, purpose, executable, network))
        return "filter-container", 4.0

    def copy(_executable, source, destination):
        destination = str(destination)
        if destination.endswith("template.json"):
            Path(destination).write_text(
                json.dumps(
                    {
                        "file_count": 1,
                        "total_bytes": 3,
                        "truncated_by": None,
                    }
                ),
                encoding="utf-8",
            )
        elif destination.endswith("template.tsv"):
            Path(destination).write_text("3\t/testbed/x.py\n", encoding="utf-8")
        return 1.0

    monkeypatch.setattr(subject, "_start", start)
    monkeypatch.setattr(subject, "_copy", copy)
    monkeypatch.setattr(
        subject,
        "_run",
        lambda argv, **_kwargs: (subprocess.CompletedProcess(argv, 0, "", ""), 5.0),
    )
    monkeypatch.setattr(
        subject, "_stop", lambda container_id, _executable: stops.append(container_id) or 3.0
    )

    result = subject._filter_clean_image(
        task_id="task-a",
        prepared={"prepared_image_id": "sha256:" + "b" * 64},
        staging=tmp_path,
        executable="docker",
    )

    assert starts == [
        ("sha256:" + "b" * 64, "task-a", "pre-target-filter", "docker", "none")
    ]
    assert stops == ["filter-container"]
    assert result["file_count"] == 1


def test_discover_task_writes_completion_artifact_last(tmp_path, monkeypatch) -> None:
    task = {
        "task_id": "task-a",
        "target_action_id": "target",
        "target_tool_args": {
            "command": "pytest",
            "timeout": 300,
            "working_dir": "/testbed",
        },
    }
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("{}", encoding="utf-8")
    prepared = {
        "prepared_image": "image:tag",
        "prepared_image_id": "sha256:" + "c" * 64,
    }
    monkeypatch.setattr(subject, "_manifest_task", lambda _task_id: (task, [], {}))
    monkeypatch.setattr(
        subject,
        "read_prepared_artifact_with_digest",
        lambda _task, _path: (prepared, "prepared-digest"),
    )

    def discover(**kwargs):
        staging = kwargs["staging"]
        (staging / "strace.log").write_text("strace", encoding="utf-8")
        (staging / "open-paths.json").write_text("{}", encoding="utf-8")
        return {"target_exit_code": 0}

    def clean_filter(**kwargs):
        staging = kwargs["staging"]
        (staging / "template.json").write_text("{}", encoding="utf-8")
        (staging / "template.tsv").write_text("3\t/testbed/x\n", encoding="utf-8")
        return {"file_count": 1}

    monkeypatch.setattr(subject, "_discover", discover)
    monkeypatch.setattr(subject, "_filter_clean_image", clean_filter)
    moved = []
    original_move = subject.shutil.move

    def move(source, destination):
        moved.append(Path(source).name)
        return original_move(source, destination)

    monkeypatch.setattr(subject.shutil, "move", move)
    out = tmp_path / "out"

    artifact = subject.discover_task(
        "task-a", prepared_path, out, executable="docker"
    )

    assert artifact["source_target_exit_code"] == 0
    assert moved[-1] == "discovery.json"
    assert (out / "task-a.discovery.json").exists()
    assert sorted(path.suffix for path in out.iterdir()) == [
        ".json",
        ".json",
        ".json",
        ".strace",
        ".tsv",
    ]
