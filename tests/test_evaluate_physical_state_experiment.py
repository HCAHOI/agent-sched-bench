import hashlib
import json

import pytest

from scripts.evaluation import evaluate_physical_state_experiment as physical


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    task_ids = tuple(f"task-{index:02d}" for index in range(12))
    monkeypatch.setattr(physical, "FROZEN_TASK_IDS", task_ids)
    manifest_path = repo / "manifest.json"
    manifest_tasks = []
    for index, task_id in enumerate(task_ids):
        source = repo / f"{task_id}.source.jsonl"
        source.write_text("{}\n", encoding="utf-8")
        command = f"python -m pytest tests/test_{index}.py"
        manifest_tasks.append(
            {
                "task_id": task_id,
                "source_trace": source.name,
                "target_action_id": f"{task_id}-target",
                "target_command": command,
                "target_tool_args": {
                    "command": command,
                    "timeout": 300,
                    "working_dir": "/testbed",
                },
                "target_source_duration_ms": 1000.0,
            }
        )
    manifest = {"schema": physical.MANIFEST_SCHEMA, "tasks": manifest_tasks}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = _sha(manifest_path)
    run_dir = repo / "baseline-run"
    run_dir.mkdir()
    attempt_inputs = []
    result_lines = []
    for index in range(100):
        task_id = f"baseline-{index:03d}"
        attempt = run_dir / task_id / "attempt_1"
        attempt.mkdir(parents=True)
        resource = attempt / "resource_observations.json"
        calls = attempt / "tool_calls.json"
        resource.write_text(json.dumps({"task_id": task_id}), encoding="utf-8")
        calls.write_text("[]\n", encoding="utf-8")
        attempt_inputs.append(
            {
                "task_id": task_id,
                "attempt_dir": str(attempt),
                "resource_observations.json": {
                    "path": str(resource),
                    "sha256": _sha(resource),
                },
                "tool_calls.json": {"path": str(calls), "sha256": _sha(calls)},
            }
        )
        result_lines.append(json.dumps({"instance_id": task_id}) + "\n")
    results_path = run_dir / "results.jsonl"
    results_path.write_text("".join(result_lines), encoding="utf-8")
    public_inputs = []
    for index in range(2):
        path = repo / f"public-{index}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        public_inputs.append({"path": str(path), "sha256": _sha(path)})
    baseline_path = repo / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": physical.BASELINE_SCHEMA,
                "manifest_sha256": manifest_hash,
                "target_run_dir": str(run_dir),
                "target_results_sha256": _sha(results_path),
                "target_attempt_inputs": attempt_inputs,
                "public_telemetry": public_inputs,
                "protocol": {"bucket_edges": list(physical.FROZEN_CPU_EDGES)},
                "tasks": [
                    {
                        "task_id": task_id,
                        "prediction_class": "low",
                        "prediction_class_id": 0,
                        "source_trace_sha256": _sha(
                            repo / manifest_tasks[index]["source_trace"]
                        ),
                    }
                    for index, task_id in enumerate(task_ids)
                ],
            }
        ),
        encoding="utf-8",
    )
    entries = []
    telemetry = []
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    task_inputs = {}
    for task_index, task_id in enumerate(task_ids):
        image_id = f"sha256:{task_index:064x}"
        for kind in ("discovery", "template"):
            (repo / f"{task_id}.{kind}.json").write_text(
                json.dumps({"task_id": task_id, "kind": kind}), encoding="utf-8"
            )
        (repo / f"{task_id}.prepared.json").write_text(
            json.dumps(
                {
                    "schema": physical.PREPARED_SCHEMA,
                    "task_id": task_id,
                    "prepared_image_id": image_id,
                }
            ),
            encoding="utf-8",
        )
        probe_input = repo / f"{task_id}.tsv"
        probe_input.write_text("4096\t/testbed/file.py\n", encoding="utf-8")
        task_inputs[task_id] = {
            "prepared_image_id": image_id,
            "prepared": repo / f"{task_id}.prepared.json",
            "discovery": repo / f"{task_id}.discovery.json",
            "template": repo / f"{task_id}.template.json",
            "probe_input": probe_input,
        }
    for index, schedule in enumerate(physical.condition_schedule(list(task_ids))):
        task_id = schedule["task_id"]
        condition = schedule["condition"]
        label = f"{task_id}/r{schedule['repeat']}/{condition}"
        inputs = task_inputs[task_id]
        trace = repo / f"condition-{index:02d}.jsonl"
        trace.write_text(
            physical._trace(
                task_id=task_id,
                condition=condition,
                repeat=schedule["repeat"],
                probe_input=inputs["probe_input"].read_text(),
                target={
                    "action_id": task_by_id[task_id]["target_action_id"],
                    "tool_args": task_by_id[task_id]["target_tool_args"],
                    "source_duration_ms": 1000.0,
                },
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                **schedule,
                "order_index": index,
                "label": label,
                "trace": str(trace),
                "target_action_id": task_by_id[task_id]["target_action_id"],
                "prepared_image_id": inputs["prepared_image_id"],
                "prepared_artifact": str(inputs["prepared"]),
                "prepared_artifact_sha256": _sha(inputs["prepared"]),
                "discovery_artifact": str(inputs["discovery"]),
                "discovery_artifact_sha256": _sha(inputs["discovery"]),
                "template_artifact": str(inputs["template"]),
                "template_artifact_sha256": _sha(inputs["template"]),
                "probe_input": str(inputs["probe_input"]),
                "probe_input_sha256": _sha(inputs["probe_input"]),
            }
        )
        warm = condition == "warm"
        resident_pages = 9 if warm else 1
        disk_bytes = 10 * 1024 * 1024 if warm else 150 * 1024 * 1024
        probe_output = json.dumps(
            {
                "condition": condition,
                "file_count": 2,
                "total_bytes": 8192,
                "total_pages": 10,
                "resident_pages": resident_pages,
                "resident_fraction": resident_pages / 10,
                "intervention_ms": 20.0,
                "probe_ms": 10.0,
            },
            separators=(",", ":"),
        )
        measured = repo / "measured" / f"{index:02d}"
        measured.mkdir(parents=True)
        resource_artifact = measured / "resource_observations.json"
        resource_artifact.write_text("{}\n", encoding="utf-8")
        container_id = f"container-{index:02d}"
        (measured / "container_startup.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "task_instance_id": task_id,
                    "manifest_index": index,
                    "label": label,
                    "source_trace": str(trace),
                    "source_image": physical.normalize_image_reference(
                        inputs["prepared_image_id"]
                    ),
                    "container_id": container_id,
                }
            ),
            encoding="utf-8",
        )
        common = {"manifest_index": index, "task_instance_id": task_id}
        telemetry.extend(
            [
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "physical_state_template",
                    "data": {
                        **common,
                        "tool_name": "write_file",
                        "tool_args": json.dumps(
                            {
                                "path": physical.TEMPLATE_CONTAINER_PATH,
                                "content": inputs["probe_input"].read_text(),
                            },
                            sort_keys=True,
                        ),
                        "tool_result": "File written",
                        "success": True,
                    },
                },
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": f"physical_state_probe_{condition}",
                    "data": {
                        **common,
                        "tool_name": "exec",
                        "tool_args": json.dumps(
                            {
                                "command": (
                                    f"{physical.PROBE_CONTAINER_PATH} {condition} "
                                    f"{physical.TEMPLATE_CONTAINER_PATH}"
                                ),
                                "timeout": 600,
                                "working_dir": "/testbed",
                            },
                            sort_keys=True,
                        ),
                        "tool_result": probe_output + "\nExit code: 0",
                        "success": True,
                    },
                },
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": task_by_id[task_id]["target_action_id"],
                    "data": {
                        **common,
                        "tool_name": "exec",
                        "tool_args": json.dumps(
                            task_by_id[task_id]["target_tool_args"], sort_keys=True
                        ),
                        "tool_result": "1 passed\nExit code: 0",
                        "duration_ms": 1000.0,
                        "success": True,
                        "resource_observation": {
                            "eligible_for_kb": True,
                            "clauses": [
                                {
                                    "eligible_for_kb": True,
                                    "peak_cpu_cores": 5.0,
                                    "cpu_window_profile": [
                                        {
                                            "start_offset_s": 0.0,
                                            "end_offset_s": 0.5,
                                            "span_s": 0.5,
                                            "cpu_ns": 2_500_000_000,
                                            "cpu_cores": 5.0,
                                        }
                                    ],
                                    "disk_io": {
                                        "read_bytes_total": disk_bytes,
                                        "read_write_bytes_total": disk_bytes,
                                    },
                                }
                            ],
                        },
                    },
                },
                {
                    "type": "summary",
                    "manifest_index": index,
                    "label": label,
                    "task_instance_id": task_id,
                    "success": True,
                    "collection_validity": "valid",
                    "telemetry_quality": "ok",
                    "formal_completeness": "complete",
                    "replay_execution": "completed",
                    "worker_returncode": 0,
                    "action_sequence_matches": True,
                    "telemetry_integrity_failed": False,
                    "replay_failed_actions": 0,
                    "unexpected_replay_failed_actions": 0,
                    "missing_source_action_count": 0,
                    "source_trace": str(trace),
                    "resource_artifact_path": str(resource_artifact),
                    "tool_container_id": container_id,
                },
            ]
        )
    protocol_path = repo / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "schema": physical.PROTOCOL_SCHEMA,
                "manifest": manifest_path.name,
                "manifest_sha256": manifest_hash,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    telemetry_path = repo / "telemetry.jsonl"
    telemetry_path.write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8"
    )
    return repo, protocol_path, baseline_path, telemetry_path


def test_scores_frozen_paired_disk_and_cpu_gates(tmp_path, monkeypatch) -> None:
    repo, protocol, baseline, telemetry = _fixture(tmp_path, monkeypatch)

    result, rows = physical.score_experiment(
        protocol, baseline, telemetry, repo_root=repo
    )

    assert len(rows) == 48
    assert result["disk"]["go"] is True
    assert len(result["disk"]["lower_bucket_tasks"]) == 12
    assert result["cpu"]["go"] is True
    assert result["cpu"]["helpful_changes"] == 48
    assert result["cost"]["probe_ms_total"] == 480.0


def test_rejects_command_outside_frozen_protocol(tmp_path, monkeypatch) -> None:
    repo, protocol, baseline, telemetry = _fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
    target = next(
        row
        for row in rows
        if row.get("type") == "action" and "resource_observation" in row.get("data", {})
    )
    target["data"]["tool_args"] = json.dumps({"command": "pytest hidden.py"})
    telemetry.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="measured action"):
        physical.score_experiment(protocol, baseline, telemetry, repo_root=repo)


def test_rejects_changed_probe_payload_and_prepared_image(
    tmp_path, monkeypatch
) -> None:
    repo, protocol, baseline, telemetry = _fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
    write = next(
        row for row in rows if row.get("action_id") == "physical_state_template"
    )
    args = json.loads(write["data"]["tool_args"])
    args["content"] = "4096\t/testbed/other.py\n"
    write["data"]["tool_args"] = json.dumps(args, sort_keys=True)
    telemetry.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="measured action"):
        physical.score_experiment(protocol, baseline, telemetry, repo_root=repo)

    repo, protocol, baseline, telemetry = _fixture(tmp_path / "second", monkeypatch)
    protocol_value = json.loads(protocol.read_text())
    changed_image = "sha256:" + "f" * 64
    protocol_value["entries"][0]["prepared_image_id"] = changed_image
    protocol.write_text(json.dumps(protocol_value), encoding="utf-8")
    summary = next(
        json.loads(line)
        for line in telemetry.read_text().splitlines()
        if json.loads(line).get("type") == "summary"
    )
    startup = (
        physical._repo_path(summary["resource_artifact_path"], repo).parent
        / "container_startup.json"
    )
    value = json.loads(startup.read_text())
    value["source_image"] = changed_image
    startup.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared artifact image"):
        physical.score_experiment(protocol, baseline, telemetry, repo_root=repo)


def test_rejects_changed_buckets_or_baseline_evidence(tmp_path, monkeypatch) -> None:
    repo, protocol, baseline, telemetry = _fixture(tmp_path, monkeypatch)
    monkeypatch.setitem(
        physical.CANONICAL_RESOURCE_BUCKET_EDGES,
        physical.CPU,
        (1.0, 4.0),
    )
    with pytest.raises(ValueError, match="bucket edges changed"):
        physical.score_experiment(protocol, baseline, telemetry, repo_root=repo)

    monkeypatch.setitem(
        physical.CANONICAL_RESOURCE_BUCKET_EDGES,
        physical.CPU,
        physical.FROZEN_CPU_EDGES,
    )
    baseline_value = json.loads(baseline.read_text())
    evidence = physical._repo_path(
        baseline_value["target_attempt_inputs"][0]["resource_observations.json"][
            "path"
        ],
        repo,
    )
    evidence.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="attempt artifact changed"):
        physical.score_experiment(protocol, baseline, telemetry, repo_root=repo)
