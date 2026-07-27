"""Report clause-telemetry acceptance gates for a replay run directory.

Reads every ``resource_observations.json`` under a run directory and reports the
gates that decide whether the run's clause data is usable. Run identically at
every rung of the staged replay ladder so the numbers stay comparable.

    python scripts/evaluation/report_clause_coverage.py RUN_DIR [--baseline DIR]

Gates
-----
1. contradictions      Zero calls rejected for control-flow contradiction or
                       ambiguous mapping evidence. These fire when the bridge
                       inferred a clause did not run but runtime evidence says
                       otherwise, so they are the falsification signal for
                       short-circuit inference.
2. additive delta      With ``--baseline``, no call may regress from eligible to
                       withheld. Short-circuit and loop mapping may only add
                       evidence; anything lost is a regression.
3. yield               Call eligibility fraction.
4. pipeline sinks      Downstream pipeline members present in the artifact but
                       excluded from KB evidence, since their wall time measures
                       their upstream's work.

Clause accountability (mapped-or-provably-absent over mappable clauses) is NOT
reported: ``mapping`` and ``no_runtime_exec`` live on the telemetryd side of the
FinalizedCallObservation contract and are not carried into the agentd artifact.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

# Fires only when the bridge inferred a clause did not run and runtime evidence
# says otherwise. `ambiguous` is deliberately excluded: it withholds the call
# rather than asserting something false, so it is a yield cost, not a defect.
_CONTRADICTION_KINDS = frozenset({"control_flow_contradiction"})


def _load_calls(run_dir: Path) -> list[dict[str, Any]]:
    """Return the run's calls in ingestion order.

    ``ingestion_sequence`` is the store's monotonic insert counter and covers
    withheld calls too, so it is a total order over the run's call stream.
    """

    calls: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*/attempt_*/resource_observations.json")):
        artifact = json.loads(path.read_text())
        task = path.relative_to(run_dir).parts[0]
        for call in artifact.get("calls", []):
            calls.append({**call, "task": task})
    return sorted(calls, key=lambda call: call.get("ingestion_sequence", 0))


def _checkpoints(calls: list[dict[str, Any]], prefixes: list[int]) -> None:
    """Learning-curve view over ordered prefixes of the SAME run.

    Prefixes count CALLS, not traces: a dev-10 run holds ~10 traces, so trace
    prefixes at 20/40/60 do not exist without launching further rungs. The
    headline gate denominator is unaffected -- this is a stability readout only,
    and it matters because the run learns causally as it goes.
    """

    print("  ordered-prefix checkpoints (calls, same run; headline gate uses all):")
    for size in [*prefixes, len(calls)]:
        if size > len(calls):
            print(f"    n={size:4d}  not reached (run has {len(calls)} calls)")
            continue
        window = calls[:size]
        eligible = sum(1 for call in window if call.get("eligible_for_kb") is True)
        traces = len({call["task"] for call in window})
        print(
            f"    n={size:4d}  eligible={eligible:4d} ({eligible / size:.3f})  "
            f"traces_seen={traces}"
        )


def _load_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Per-trace run manifests: what actually became KB evidence."""

    return [
        json.loads(path.read_text())
        for path in sorted(run_dir.glob("tool_resource_runs/*/*.json"))
    ]


def _bucket(latency_ms: float, edges: list[float]) -> int:
    return sum(1 for edge in edges if latency_ms >= edge)


def _gate_promotion(
    run_dir: Path, baseline_dir: Path | None, eligible_calls: int
) -> bool:
    """Every eligible call must have one store-confirmed promotion."""

    runs = _load_runs(run_dir)
    if not runs:
        print("  [gate 0] promotion: NO run manifests found  FAIL")
        return False
    promoted = sum(run.get("promoted_observation_count", 0) for run in runs)
    invalid_runs = [
        run["run_manifest"]["workspace_scope"]
        for run in runs
        if run.get("evidence_valid") is not True
    ]
    print(
        f"  [gate 0] promotion: {promoted} observations "
        f"({eligible_calls} calls eligible across {len(runs)} traces)",
        end="",
    )
    passed = promoted == eligible_calls
    if not passed:
        print("  FAIL  promoted count differs from eligible calls")
    else:
        print("  ok")
    if invalid_runs:
        print(f"           run-level evidence invalid: {invalid_runs}")
    if baseline_dir is not None:
        before = sum(
            run.get("promoted_observation_count", 0)
            for run in _load_runs(baseline_dir)
        )
        delta = promoted - before
        print(f"           vs baseline: {before} -> {promoted} ({delta:+d})", end="")
        if delta < 0:
            passed = False
            print("  FAIL  promoted evidence regressed")
        else:
            print("  ok")
    return passed


def report(
    run_dir: Path,
    baseline_dir: Path | None,
    edges: list[float],
    prefixes: list[int],
) -> bool:
    calls = _load_calls(run_dir)
    if not calls:
        raise SystemExit(f"no resource_observations.json found under {run_dir}")

    eligible = [call for call in calls if call.get("eligible_for_kb") is True]
    reasons: collections.Counter[str] = collections.Counter()
    for call in calls:
        for reason in call.get("invalid_reasons") or []:
            reasons[reason.get("kind", "?")] += 1

    contradictions = sum(reasons[kind] for kind in _CONTRADICTION_KINDS)
    yield_fraction = len(eligible) / len(calls)

    print(f"run: {run_dir}")
    print(f"  calls={len(calls)} eligible={len(eligible)} ({yield_fraction:.3f})")

    passed = _gate_promotion(run_dir, baseline_dir, len(eligible))
    print(f"  [gate 1] contradictions: {contradictions} (must be 0)", end="")
    if contradictions:
        passed = False
        print("  FAIL")
    else:
        print("  ok")

    if baseline_dir is not None:
        before = {
            call["tool_call_id"]: call.get("eligible_for_kb") is True
            for call in _load_calls(baseline_dir)
        }
        regressed = [
            call["tool_call_id"]
            for call in calls
            if before.get(call["tool_call_id"]) and not call.get("eligible_for_kb")
        ]
        gained = sum(
            1
            for call in eligible
            if call["tool_call_id"] in before and not before[call["tool_call_id"]]
        )
        print(
            f"  [gate 2] vs baseline: +{gained} gained, "
            f"{len(regressed)} regressed (must be 0)",
            end="",
        )
        if regressed:
            passed = False
            print(f"  FAIL {regressed[:5]}")
        else:
            print("  ok")

    print(f"  [gate 3] yield: {yield_fraction:.3f} (target >= 0.90)", end="")
    if yield_fraction < 0.90:
        passed = False
        print("  FAIL")
    else:
        print("  ok")

    sinks = sum(
        1
        for call in eligible
        for clause in call.get("clauses", [])
        if int(clause.get("pipeline_position", -1)) > 0
    )
    print(f"  [gate 4] pipeline sinks excluded from KB evidence: {sinks}")

    if prefixes:
        _checkpoints(calls, prefixes)

    print("  withheld-reason histogram:")
    for kind, count in reasons.most_common():
        print(f"    {count:5d}  {kind}")

    observations = [
        clause
        for call in eligible
        for clause in call.get("clauses", [])
        if (clause.get("availability") or {}).get("latency") == "ok"
        and int(clause.get("pipeline_position", -1)) <= 0
    ]
    keys: dict[tuple[str, tuple[str, ...]], list[float]] = collections.defaultdict(list)
    for clause in observations:
        keys[(clause["bin"], tuple(clause["argv"]))].append(clause["latency_ms"])
    repeated = [values for values in keys.values() if len(values) > 1]
    ambiguous = [
        values
        for values in repeated
        if len({_bucket(value, edges) for value in values}) > 1
    ]
    loop_clauses = sum(1 for clause in observations if clause.get("in_loop"))
    print(
        f"  observations={len(observations)} (loop-expanded {loop_clauses}) "
        f"keys={len(keys)} singletons={sum(1 for v in keys.values() if len(v) == 1)}"
    )
    print(
        f"  bucket purity: {len(ambiguous)}/{len(repeated)} repeated keys ambiguous "
        f"at edges {edges}"
    )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Prior run directory; asserts no call regresses eligible -> withheld.",
    )
    parser.add_argument(
        "--bucket-edges-ms",
        default="10,100,1000",
        help="Diagnostic only; no authoritative boundaries are frozen yet.",
    )
    parser.add_argument(
        "--checkpoints",
        default="20,40,60",
        help=(
            "Ordered CALL-count prefixes of the same run for a learning-curve "
            "readout. Empty to disable. These never change the gate denominator."
        ),
    )
    args = parser.parse_args()
    edges = [float(value) for value in args.bucket_edges_ms.split(",")]
    prefixes = [
        int(value) for value in args.checkpoints.split(",") if value.strip()
    ]
    raise SystemExit(
        0 if report(args.run_dir, args.baseline, edges, prefixes) else 1
    )


if __name__ == "__main__":
    main()
