#!/usr/bin/env python3
"""Candidate B overlap diagnostic: would a stable-atom gate fire where the
chain-prefix trie is weak, or only where it is already strong?

Pre-registered in ``analysis/CLOSED-QUESTIONS.md`` (Candidate B
falsification, "Diagnostic before any replay"). Candidate B pools a heavy,
cross-task-stable atom's samples across every chain context and lets that node
fire when a call contains the atom. Its ONLY route to union gain is firing on
calls the chain-prefix trie serves at fallback or thin support - if the stable
atom always lands where the trie already has a strong prefix node, B just
overlaps H1's certified mass and dies on arithmetic before any replay.

Three parts, all cross-fitted over the study's task-grouped folds and reusing
its atom extraction / stability machinery
(``scripts.analyze_segment_variance``); node selection reuses the PRODUCTION
prior hierarchy (``trace_collect.tool_latency_profiled.latency_prior_hierarchy``)
at the certified fit config, not a reinvented trie:

  (1) Stability screen. An atom qualifies iff heavy (per-task median duration
      above ``--action-relevance-floor-ms``) AND cross-task CV below
      ``--cv-cap`` AND observed in at least ``--min-task-support`` tasks. All
      three are documented config applied to FIT-fold statistics; no atom name
      appears in the logic, so apt-get (or nothing) must EMERGE from the
      numbers.
  (2) Fold stability of the selection: the qualifying set is recomputed on
      each fold's fit split; the intersection/union and pairwise Jaccard say
      whether the screen generalizes. A screen that selects different atoms on
      different folds does not generalize.
  (3) Overlap count: among held-out calls (the "fresh-277 decisions" - the
      same 277-task replay corpus), count those where the chain-prefix trie is
      at fallback/thin support but a fold-qualifying stable atom is present, so
      the pooled atom node would fire. Reported as a fraction of divergent
      (atom-firing) decisions and of all decisions.

Kill readout (printed explicitly): KILL if the overlap fraction of divergent
decisions is below ``--kill-overlap-frac`` (~5%) OR the atom selection is
fold-unstable; SURVIVE otherwise.

NO ORACLE LEAKAGE: for each held-out fold the screen and the prior are both fit
on the task-disjoint remainder; observed durations are never used as features.
Exploratory; durations replayed on our own hardware. Emits JSON + MD to
``analysis/`` (``-PARTIAL`` suffix unless ``--final``).

Usage:
  uv run python scripts/analyze_stable_atom_overlap.py \
    --traces-dir traces/fresh-277-segtimeline --fold-count 5 --final
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime as _dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

# Allow `python scripts/analyze_stable_atom_overlap.py ...` to import the
# sibling study module as a package (pytest adds the repo root itself).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_segment_variance import (  # noqa: E402
    Chain,
    Config,
    _task_folds,
    atom_stability,
    build_chains,
)
from trace_collect.command_features import make_row_command_prefix_keys
from trace_collect.tool_latency_dataset import extract_many_segment_latency_samples
from trace_collect.tool_latency_profiled import (
    build_latency_prior,
    latency_prior_hierarchy,
)

# Certified fit config (fresh-corpus certification manifest): the prefix-tree
# depth, cd handling and support gates the trie was certified at. The overlap
# diagnostic must judge the trie at exactly the config it is deployed at.
_CERT_MAX_PREFIX_DEPTH = 4
_CERT_SKIP_LEADING_CD = False
_CERT_MIN_TOOL_HISTORY = 1
_CERT_MIN_PROFILE_TASKS = 1
_CERT_COMMAND_FIELD = "command"


@dataclass(frozen=True)
class ScreenConfig:
    action_relevance_floor_ms: float
    cv_cap: float
    min_task_support: int


def stable_atoms(chains: Sequence[Chain], screen: ScreenConfig) -> set[str]:
    """Atoms passing the heavy AND low-CV AND min-support screen on ``chains``.

    Reuses the study's cross-task CV (per-task medians). ``min_task_support``
    is enforced via ``atom_stability(min_count=...)``; the floor and cap are
    applied on top. No atom name is referenced - selection is purely numeric.
    """

    qualifying: set[str] = set()
    for row in atom_stability(chains, min_count=screen.min_task_support):
        cv = row["cv_across_tasks"]
        if (
            row["median_ms"] > screen.action_relevance_floor_ms
            and cv == cv  # finite (NaN CV = single-task or zero-mean: reject)
            and cv < screen.cv_cap
        ):
            qualifying.add(row["atom"])
    return qualifying


def _chain_to_prior_row(chain: Chain, index: int) -> dict[str, Any]:
    """A tool-latency-style row for the production prior builder.

    The stored latency value never affects node SELECTION (the hierarchy gates
    on sample/task counts, not magnitudes); it is required only to satisfy the
    row schema. The command drives the prefix-tree keys.
    """

    return {
        "sample_id": f"{chain.source_trace}:{chain.action_id}:{index}",
        "source_trace": chain.source_trace,
        "task_id": chain.task_id,
        "tool_name": chain.tool_name,
        "latency_ms": chain.parent_total_ms,
        "tool_args": {_CERT_COMMAND_FIELD: chain.parent_command},
    }


@dataclass(frozen=True)
class OverlapConfig:
    fold_count: int
    screen: ScreenConfig
    thin_support_cap: int
    kill_overlap_frac: float
    fold_jaccard_floor: float


def _mean_pairwise_jaccard(sets: Sequence[set[str]]) -> float:
    """Mean Jaccard over fold pairs; 1.0 for identical (or all-empty) sets."""

    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    if not pairs:
        return 1.0
    scores = []
    for a, b in pairs:
        union = a | b
        scores.append(1.0 if not union else len(a & b) / len(union))
    return sum(scores) / len(scores)


def run_overlap(chains: Sequence[Chain], cfg: OverlapConfig) -> dict[str, Any]:
    """Cross-fitted stable-atom overlap diagnostic over task-grouped folds."""

    usable = [chain for chain in chains if chain.parent_command]
    fold_cfg = Config(
        fold_count=cfg.fold_count,
        prefix_depth=4,
        min_prefix_evidence=1,
        atom_depth=4,
        token_bin_count=3,
        min_atom_count=10,
        min_family_count=20,
        tail_percentile=90.0,
    )
    row_group_keys = make_row_command_prefix_keys(
        _CERT_COMMAND_FIELD,
        max_depth=_CERT_MAX_PREFIX_DEPTH,
        skip_leading_cd=_CERT_SKIP_LEADING_CD,
    )

    per_fold_qualifying: list[set[str]] = []
    total_calls = 0
    divergent_calls = 0
    overlap_calls = 0  # divergent AND (fallback OR thin)
    strict_fallback_overlap = 0  # divergent AND source != prior_group
    firing_atom_divergent: Counter[str] = Counter()
    firing_atom_overlap: Counter[str] = Counter()

    for train, test in _task_folds(usable, fold_cfg):
        qualifying = stable_atoms(train, cfg.screen)
        per_fold_qualifying.append(qualifying)
        prior = build_latency_prior(
            [_chain_to_prior_row(c, i) for i, c in enumerate(train)],
            row_group_keys=row_group_keys,
        )
        for chain in test:
            total_calls += 1
            row = _chain_to_prior_row(chain, 0)
            group_keys = row_group_keys(row)
            selected = latency_prior_hierarchy(
                prior,
                chain.tool_name,
                group_keys,
                min_tool_history=_CERT_MIN_TOOL_HISTORY,
                min_profile_tasks=_CERT_MIN_PROFILE_TASKS,
            )[-1]
            atoms_present = {seg.atom for seg in chain.segments}
            firing = atoms_present & qualifying
            if not firing:
                continue
            divergent_calls += 1
            for atom in firing:
                firing_atom_divergent[atom] += 1
            is_fallback = selected.source != "prior_group"
            is_thin = (
                selected.source == "prior_group"
                and len(selected.values) <= cfg.thin_support_cap
            )
            if is_fallback:
                strict_fallback_overlap += 1
            if is_fallback or is_thin:
                overlap_calls += 1
                for atom in firing:
                    firing_atom_overlap[atom] += 1

    intersection = set.intersection(*per_fold_qualifying) if per_fold_qualifying else set()
    union = set.union(*per_fold_qualifying) if per_fold_qualifying else set()
    jaccard = _mean_pairwise_jaccard(per_fold_qualifying)
    fold_stable = bool(intersection) and jaccard >= cfg.fold_jaccard_floor

    overlap_frac_divergent = (
        overlap_calls / divergent_calls if divergent_calls else None
    )
    overlap_frac_total = overlap_calls / total_calls if total_calls else None
    overlap_below_bar = (
        overlap_frac_divergent is None
        or overlap_frac_divergent < cfg.kill_overlap_frac
    )
    killed = bool(overlap_below_bar or not fold_stable)

    return {
        "full_corpus_stability": atom_stability(usable, min_count=cfg.screen.min_task_support),
        "screen_selection": {
            "per_fold_qualifying": [sorted(s) for s in per_fold_qualifying],
            "intersection": sorted(intersection),
            "union": sorted(union),
            "mean_pairwise_jaccard": jaccard,
            "fold_stable": fold_stable,
        },
        "overlap": {
            "total_calls": total_calls,
            "divergent_calls": divergent_calls,
            "overlap_calls": overlap_calls,
            "strict_fallback_overlap_calls": strict_fallback_overlap,
            "overlap_fraction_of_divergent": overlap_frac_divergent,
            "overlap_fraction_of_total": overlap_frac_total,
            "firing_atom_divergent": dict(firing_atom_divergent.most_common()),
            "firing_atom_overlap": dict(firing_atom_overlap.most_common()),
        },
        "kill_readout": {
            "kill_overlap_frac": cfg.kill_overlap_frac,
            "overlap_below_bar": overlap_below_bar,
            "fold_stable": fold_stable,
            "verdict": "KILL" if killed else "SURVIVE",
        },
    }


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _banner(final: bool) -> str:
    if final:
        return "FINAL - complete corpus"
    return (
        "PARTIAL - not final: validated against a subset; numbers are for "
        "script validation only, not findings."
    )


def render_markdown(results: dict[str, Any], provenance: dict[str, Any]) -> str:
    sel = results["screen_selection"]
    ov = results["overlap"]
    kill = results["kill_readout"]
    lines: list[str] = []
    lines.append("# Candidate B stable-atom overlap diagnostic")
    lines.append("")
    lines.append(f"> **{_banner(provenance['final'])}**")
    lines.append(">")
    lines.append(
        "> EXPLORATORY. Durations replayed on our own hardware "
        f"({provenance['replayed_on']}). Generated {provenance['generated']}."
    )
    lines.append("")
    lines.append(f"**Verdict: {kill['verdict']}**")
    lines.append("")
    lines.append("## Screen selection (cross-fitted)")
    lines.append("")
    lines.append(f"- per-fold qualifying: {sel['per_fold_qualifying']}")
    lines.append(f"- intersection (all folds): {sel['intersection']}")
    lines.append(f"- union (any fold): {sel['union']}")
    lines.append(f"- mean pairwise Jaccard: {sel['mean_pairwise_jaccard']:.3f}")
    lines.append(f"- fold-stable: {sel['fold_stable']}")
    lines.append("")
    lines.append("## Overlap with the chain-prefix trie")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("| --- | --- |")
    lines.append(f"| held-out calls | {ov['total_calls']} |")
    lines.append(f"| divergent (stable atom fires) | {ov['divergent_calls']} |")
    lines.append(f"| overlap (trie fallback/thin AND fires) | {ov['overlap_calls']} |")
    lines.append(
        f"| strict fallback (source != prior_group) | {ov['strict_fallback_overlap_calls']} |"
    )
    ofd = ov["overlap_fraction_of_divergent"]
    oft = ov["overlap_fraction_of_total"]
    lines.append(
        f"| overlap / divergent | {ofd:.3%} |" if ofd is not None else "| overlap / divergent | - |"
    )
    lines.append(
        f"| overlap / total | {oft:.3%} |" if oft is not None else "| overlap / total | - |"
    )
    lines.append("")
    lines.append(f"Firing atoms (divergent): {ov['firing_atom_divergent']}")
    lines.append("")
    lines.append(f"Firing atoms (overlap): {ov['firing_atom_overlap']}")
    lines.append("")
    lines.append("## Kill readout")
    lines.append("")
    lines.append(
        f"KILL if overlap/divergent < {kill['kill_overlap_frac']:.1%} OR the "
        "atom selection is fold-unstable; SURVIVE otherwise."
    )
    lines.append("")
    lines.append(f"- overlap below bar: {kill['overlap_below_bar']}")
    lines.append(f"- fold-stable: {kill['fold_stable']}")
    lines.append(f"- **verdict: {kill['verdict']}**")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.wave_*.worker_*.jsonl")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument(
        "--limit", type=int, default=None, help="Smoke only: cap wave files loaded."
    )
    parser.add_argument(
        "--action-relevance-floor-ms",
        type=float,
        default=500.0,
        help="Heavy-atom floor: an atom below the smallest certified kv cost "
        "(500ms) cannot move a swap decision. Default 500.",
    )
    parser.add_argument(
        "--cv-cap",
        type=float,
        default=1.0,
        help="Cross-task CV cap: CV < 1 means the std is below the mean "
        "(concentrated positive distribution). Default 1.0.",
    )
    parser.add_argument(
        "--min-task-support",
        type=int,
        default=10,
        help="Distinct tasks required for a meaningful cross-task CV. Default 10.",
    )
    parser.add_argument(
        "--thin-support-cap",
        type=int,
        default=5,
        help="A prefix node with <= this many observed calls counts as thin "
        "(a per-prefix quantile on a handful of calls is untrustworthy). Default 5.",
    )
    parser.add_argument(
        "--kill-overlap-frac",
        type=float,
        default=0.05,
        help="Overlap kill threshold on divergent decisions (memo ~5%%).",
    )
    parser.add_argument(
        "--fold-jaccard-floor",
        type=float,
        default=1.0,
        help="Minimum mean pairwise Jaccard of per-fold qualifying sets for a "
        "stable selection. Default 1.0 (identical sets across folds).",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--final", action="store_true")
    return parser


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/stable-atom-overlap-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    files = sorted(args.traces_dir.glob(args.glob))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"no trace files under {args.traces_dir}/{args.glob}")
    samples = extract_many_segment_latency_samples(files, skip_concurrent=True)
    chains = build_chains(samples)

    cfg = OverlapConfig(
        fold_count=args.fold_count,
        screen=ScreenConfig(
            action_relevance_floor_ms=args.action_relevance_floor_ms,
            cv_cap=args.cv_cap,
            min_task_support=args.min_task_support,
        ),
        thin_support_cap=args.thin_support_cap,
        kill_overlap_frac=args.kill_overlap_frac,
        fold_jaccard_floor=args.fold_jaccard_floor,
    )
    results = run_overlap(chains, cfg)
    provenance = {
        "exploratory": True,
        "final": bool(args.final),
        "replayed_on": "our_hardware",
        "traces_dir": str(args.traces_dir),
        "corpus_file_count": len(files),
        "git_sha": _git_sha(),
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "cert_fit_config": {
            "max_prefix_depth": _CERT_MAX_PREFIX_DEPTH,
            "skip_leading_cd": _CERT_SKIP_LEADING_CD,
            "min_tool_history": _CERT_MIN_TOOL_HISTORY,
            "min_profile_tasks": _CERT_MIN_PROFILE_TASKS,
        },
    }
    payload = {
        "provenance": provenance,
        "config": {
            "fold_count": cfg.fold_count,
            "screen": cfg.screen.__dict__,
            "thin_support_cap": cfg.thin_support_cap,
            "kill_overlap_frac": cfg.kill_overlap_frac,
            "fold_jaccard_floor": cfg.fold_jaccard_floor,
        },
        **results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(results, provenance), encoding="utf-8")

    kill = results["kill_readout"]
    ov = results["overlap"]
    print(
        f"verdict={kill['verdict']} fold_stable={kill['fold_stable']} "
        f"divergent={ov['divergent_calls']} overlap={ov['overlap_calls']} "
        + (
            f"overlap/divergent={ov['overlap_fraction_of_divergent']:.3%}"
            if ov["overlap_fraction_of_divergent"] is not None
            else "overlap/divergent=-"
        )
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
