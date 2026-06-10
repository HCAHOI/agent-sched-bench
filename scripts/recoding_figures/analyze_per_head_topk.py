"""Per-head counterfactual top-k analysis for block_topk recordings.

Reads the UNCENSORED per-head top-R block selections recorded by
``--record-per-head-topk`` (attention.npz ``per_head_topk_csr_*``) and answers,
per (layer, call), how much query heads disagree when each picks blocks alone:

  1. Per-head pairwise top-k Jaccard for k in {8, 16, 32, 64} (k <= R_ph).
  2. 32-head top-k union size, and its relation to the pooled keep set actually
     retained by block_topk (sparse_attention.npz ``selected_blocks_kept``):
     |union ∩ kept|, |union ∪ kept|, |union \\ kept|, |kept \\ union|.
  3. n90: min blocks covering 90% of a head's SELECTION-SIGNAL mass, defined as
     softmax over that head's recorded top-R pre-softmax QK block scores. This
     is selection-signal concentration, NOT post-softmax attention mass (this
     run does not record per-head attention) — labelled as such throughout.
  4. Consensus core: blocks chosen by >= 75% of heads at k = 16.
  5. vote-vs-max selection-set comparison: from the SAME per-head scores, re-run
     the two block_topk aggregations offline — max (per-block cross-head max
     score, top-B) and vote (per-block cross-head top-B vote count, top-B) —
     and report their Jaccard plus the role composition of the complementary
     blocks (which blocks swapping the aggregation adds/drops). B is the middle
     budget the step actually filled (sparse_attention.npz selected_middle_count).

Re-simulation caveat (metric 5): we only know each head's top-R block scores, so
a block outside every head's top-R contributes neither a max-score nor a vote.
For B <= R this is exact for any block in >= 1 head's top-B; it can only miss
blocks no head ranked highly, which by construction cannot enter a top-B set.

Pure numpy/pandas (local .venv has no scipy/torch/matplotlib). Outputs
per_head_topk_metrics.csv, per_head_topk_summary.json, per_head_topk_summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.recoding_figures.recording_loader import (  # noqa: E402
    IterationRecord,
    find_attempt_dirs,
    load_per_head_topk,
    load_iteration_records,
)

# --- Tunable constants (documented; change here, not inline) ---------------
TOP_K_VALUES: tuple[int, ...] = (8, 16, 32, 64)  # per-head counterfactual budgets
CONSENSUS_K: int = 16                            # k for the consensus-core metric
CONSENSUS_FRAC: float = 0.75                     # >= 75% of heads must agree
N90_MASS: float = 0.90                           # selection-signal coverage target
MAX_JACCARD_PAIRS: int = 496                     # 32 heads -> C(32,2)=496 (use all)


@dataclass(frozen=True)
class StepSelections:
    """One (layer, decode_step) of per-head top-R selections plus the kept set."""

    layer: int
    decode_step: int
    # head_blocks[h] / head_scores[h]: descending-score block ids / scores.
    head_blocks: list[np.ndarray]
    head_scores: list[np.ndarray]
    kept_blocks: frozenset[int]  # pooled block_topk selected_blocks_kept
    middle_budget: int           # selected middle TOKEN POSITIONS this step (not blocks)


def _decode_per_head_topk(iter_dir: Path) -> dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]]:
    """Decode the CSR per-head topk into {(layer, step): [(block_ids, scores)]*H}.

    Empty when the iter has no per_head_topk record. Block ids are absolute.
    """
    data = load_per_head_topk(iter_dir)
    offsets = data["per_head_topk_csr_offsets"]
    if offsets.shape[0] <= 1:
        return {}
    rank = int(data["per_head_topk_rank"])
    if rank and max(TOP_K_VALUES) > rank:
        raise ValueError(
            f"analysis k values {TOP_K_VALUES} exceed recorded per_head_topk_rank="
            f"{rank}: top-k sets would be silently truncated. Re-record with a "
            f"larger --per-head-topk-rank or drop the offending k."
        )
    layers = data["per_head_topk_layers"]
    H = int(data["per_head_topk_head_count"])
    step_arr = data["per_head_topk_decode_step"]  # [L_s, T_max]
    block_ids = data["per_head_topk_csr_block_ids"]
    scores = data["per_head_topk_csr_scores"]
    L_s, T_max = step_arr.shape
    out: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for li in range(L_s):
        layer = int(layers[li])
        for ti in range(T_max):
            step = int(step_arr[li, ti])
            if step < 0:
                continue
            heads: list[tuple[np.ndarray, np.ndarray]] = []
            for h in range(H):
                row = (li * T_max + ti) * H + h
                lo, hi = int(offsets[row]), int(offsets[row + 1])
                heads.append((block_ids[lo:hi], scores[lo:hi]))
            out[(layer, step)] = heads
    return out


def _kept_lookup(iter_dir: Path) -> dict[tuple[int, int], tuple[frozenset[int], int]]:
    """Map (layer, decode_step) -> (selected_blocks_kept, selected_middle_count).

    Read from sparse_attention.npz extras_json (decode rows only). Empty when the
    npz is absent (e.g. the arm was run without --sparse-attn-record).
    """
    npz_path = iter_dir / "sparse_attention.npz"
    if not npz_path.is_file():
        return {}
    out: dict[tuple[int, int], tuple[frozenset[int], int]] = {}
    with np.load(npz_path, allow_pickle=True) as data:
        layer = data["record_layer"].astype(np.int32)
        phase = data["record_phase"]
        dstep = data["record_decode_step"].astype(np.int32)
        extras = data["extras_json"]
        for i in range(layer.shape[0]):
            if str(phase[i]) != "decode":
                continue
            meta = json.loads(str(extras[i])) if extras[i] else {}
            kept = frozenset(int(b) for b in meta.get("selected_blocks_kept", []))
            budget = int(meta.get("selected_middle_count", 0))
            out[(int(layer[i]), int(dstep[i]))] = (kept, budget)
    return out


def _block_roles(iter_dir: Path, block_size: int) -> dict[int, str]:
    """Map block_id -> role (segment role of the block's first token).

    Blocks are <= block_size contiguous tokens; the first-token role labels the
    block. Straddling blocks are rare and the first-token role is a faithful,
    documented summary for role-composition counts.
    """
    seg_path = iter_dir / "segments.json"
    if not seg_path.is_file():
        return {}
    payload = json.loads(seg_path.read_text(encoding="utf-8"))
    token_seg = payload.get("token_segment_id") or []
    segments = payload.get("segments") or []
    roles_by_seg = {i: str(seg.get("role", "unknown")) for i, seg in enumerate(segments)}
    n_tok = len(token_seg)
    roles: dict[int, str] = {}
    n_blocks = (n_tok + block_size - 1) // block_size
    for b in range(n_blocks):
        tok = b * block_size
        if tok < n_tok:
            roles[b] = roles_by_seg.get(int(token_seg[tok]), "unknown")
    return roles


def _topk_set(blocks: np.ndarray, scores: np.ndarray, k: int) -> set[int]:
    """Top-k block ids by descending score (rows are already score-sorted)."""
    if blocks.shape[0] == 0:
        return set()
    if blocks.shape[0] <= k:
        return set(int(b) for b in blocks)
    return set(int(b) for b in blocks[:k])


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def _n90(scores: np.ndarray) -> int:
    """Min blocks to cover N90_MASS of softmax(scores) selection-signal mass."""
    if scores.shape[0] == 0:
        return 0
    s = scores.astype(np.float64)
    s = s - s.max()
    w = np.exp(s)
    total = w.sum()
    if total <= 0:
        return 0
    order = np.sort(w)[::-1]
    cumsum = np.cumsum(order)
    return int(np.searchsorted(cumsum, N90_MASS * total)) + 1


def _reaggregate(step: StepSelections, block_size: int) -> tuple[set[int], set[int]]:
    """Offline re-run of max vs vote aggregation from per-head top-R scores.

    Returns (max_topB, vote_topB). max: per-block cross-head max score, top-B by
    score. vote: each head votes for its own top-B blocks, rank by vote count
    (ties: cross-head max score desc, block id asc), top-B. B is in BLOCKS:
    ceil(middle_budget / block_size), mirroring the online conversion in
    BlockTopKSparseAttention._rank_middle_positions (middle_budget counts token
    positions). See module caveat.
    """
    b = -(-int(step.middle_budget) // int(block_size))
    cross_max: dict[int, float] = defaultdict(lambda: float("-inf"))
    for blocks, scores in zip(step.head_blocks, step.head_scores):
        for blk, sc in zip(blocks.tolist(), scores.tolist()):
            blk = int(blk)
            if float(sc) > cross_max[blk]:
                cross_max[blk] = float(sc)
    if b <= 0 or not cross_max:
        return set(), set()
    max_ranked = sorted(cross_max.items(), key=lambda kv: (-kv[1], kv[0]))
    max_set = {blk for blk, _ in max_ranked[:b]}

    votes: dict[int, int] = defaultdict(int)
    for blocks, _scores in zip(step.head_blocks, step.head_scores):
        for blk in _topk_set(blocks, _scores, b):
            votes[blk] += 1
    vote_ranked = sorted(
        votes.items(), key=lambda kv: (-kv[1], -cross_max[kv[0]], kv[0])
    )
    vote_set = {blk for blk, _ in vote_ranked[:b]}
    return max_set, vote_set


def _iter_steps(record: IterationRecord) -> list[StepSelections]:
    per_head = _decode_per_head_topk(record.iter_dir)
    if not per_head:
        return []
    kept = _kept_lookup(record.iter_dir)
    steps: list[StepSelections] = []
    for (layer, step), heads in per_head.items():
        kept_blocks, budget = kept.get((layer, step), (frozenset(), 0))
        steps.append(
            StepSelections(
                layer=layer,
                decode_step=step,
                head_blocks=[h[0] for h in heads],
                head_scores=[h[1] for h in heads],
                kept_blocks=kept_blocks,
                middle_budget=budget,
            )
        )
    return steps


def _step_rows(
    task: str, call_idx: int, step: StepSelections, roles: dict[int, str], block_size: int
) -> list[dict]:
    """One metric row per (task, call, layer, step, k) plus k-agnostic columns."""
    n_heads = len(step.head_blocks)
    pairs = [(i, j) for i in range(n_heads) for j in range(i + 1, n_heads)]
    if len(pairs) > MAX_JACCARD_PAIRS:
        pairs = pairs[:MAX_JACCARD_PAIRS]

    n90_vals = [
        _n90(sc) for sc in step.head_scores if sc.shape[0] > 0
    ]
    n90_mean = float(np.mean(n90_vals)) if n90_vals else float("nan")

    # Consensus core at CONSENSUS_K.
    votes_c: dict[int, int] = defaultdict(int)
    for blocks, scores in zip(step.head_blocks, step.head_scores):
        for blk in _topk_set(blocks, scores, CONSENSUS_K):
            votes_c[blk] += 1
    min_consensus = int(np.ceil(CONSENSUS_FRAC * n_heads))
    consensus_core = {blk for blk, v in votes_c.items() if v >= min_consensus}

    # vote-vs-max re-aggregation (k-agnostic; B = ceil(middle_budget / block_size)).
    max_set, vote_set = _reaggregate(step, block_size)
    vote_max_jac = _jaccard(max_set, vote_set)
    only_vote = vote_set - max_set
    only_max = max_set - vote_set
    role_only_vote = _role_counts(only_vote, roles)
    role_only_max = _role_counts(only_max, roles)

    rows: list[dict] = []
    for k in TOP_K_VALUES:
        head_sets = [
            _topk_set(b, s, k) for b, s in zip(step.head_blocks, step.head_scores)
        ]
        jac = [
            _jaccard(head_sets[i], head_sets[j])
            for i, j in pairs
            if head_sets[i] or head_sets[j]
        ]
        union: set[int] = set()
        for s in head_sets:
            union |= s
        kept = step.kept_blocks
        rows.append(
            {
                "task": task,
                "call_idx": call_idx,
                "layer": step.layer,
                "decode_step": step.decode_step,
                "k": k,
                "n_heads": n_heads,
                "jaccard_mean": float(np.mean(jac)) if jac else float("nan"),
                "jaccard_median": float(np.median(jac)) if jac else float("nan"),
                "union_abs": len(union),
                "kept_abs": len(kept),
                "union_and_kept": len(union & kept),
                "union_or_kept": len(union | kept),
                "union_minus_kept": len(union - kept),
                "kept_minus_union": len(kept - union),
                "n90_mean": n90_mean,
                "consensus_core_size": len(consensus_core),
                "middle_budget": step.middle_budget,
                "vote_vs_max_jaccard": vote_max_jac,
                "vote_only_count": len(only_vote),
                "max_only_count": len(only_max),
                "vote_only_roles": json.dumps(role_only_vote, sort_keys=True),
                "max_only_roles": json.dumps(role_only_max, sort_keys=True),
            }
        )
    return rows


def _role_counts(blocks: set[int], roles: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for blk in blocks:
        counts[roles.get(blk, "unknown")] += 1
    return dict(counts)


def analyze(records: list[IterationRecord], block_size: int) -> pd.DataFrame:
    all_rows: list[dict] = []
    for record in records:
        steps = _iter_steps(record)
        if not steps:
            continue
        roles = _block_roles(record.iter_dir, block_size)
        for step in steps:
            all_rows.extend(
                _step_rows(record.task, record.call_idx, step, roles, block_size)
            )
    if not all_rows:
        raise FileNotFoundError(
            "no per_head_topk records found — were the attempts collected with "
            "--record-per-head-topk?"
        )
    return pd.DataFrame(all_rows)


def summarize(df: pd.DataFrame) -> dict:
    """Per (layer, k) aggregates plus a global block of headline numbers."""
    grouped = (
        df.groupby(["layer", "k"])
        .agg(
            n_steps=("decode_step", "size"),
            jaccard_mean=("jaccard_mean", "mean"),
            union_abs_mean=("union_abs", "mean"),
            union_minus_kept_mean=("union_minus_kept", "mean"),
            kept_minus_union_mean=("kept_minus_union", "mean"),
            n90_mean=("n90_mean", "mean"),
            consensus_core_mean=("consensus_core_size", "mean"),
            vote_vs_max_jaccard_mean=("vote_vs_max_jaccard", "mean"),
            vote_only_mean=("vote_only_count", "mean"),
        )
        .reset_index()
    )
    return {
        "per_layer_k": grouped.to_dict(orient="records"),
        "n_rows": int(df.shape[0]),
        "layers": sorted(int(x) for x in df["layer"].unique()),
        "k_values": list(TOP_K_VALUES),
    }


def _render_markdown(df: pd.DataFrame, summary: dict) -> str:
    lines = ["# Per-head counterfactual top-k analysis", ""]
    lines.append(f"- rows: {summary['n_rows']}")
    lines.append(f"- layers: {summary['layers']}")
    lines.append(f"- k values: {summary['k_values']}")
    lines.append("")
    lines.append("## Per (layer, k)")
    lines.append("")
    headers = [
        "layer", "k", "n_steps", "jaccard_mean", "union_abs_mean",
        "union\\kept_mean", "kept\\union_mean", "n90_mean", "consensus_mean",
        "vote~max_jac", "vote_only_mean",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary["per_layer_k"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["layer"])),
                    str(int(row["k"])),
                    str(int(row["n_steps"])),
                    f"{row['jaccard_mean']:.3f}",
                    f"{row['union_abs_mean']:.1f}",
                    f"{row['union_minus_kept_mean']:.1f}",
                    f"{row['kept_minus_union_mean']:.1f}",
                    f"{row['n90_mean']:.1f}",
                    f"{row['consensus_core_mean']:.1f}",
                    f"{row['vote_vs_max_jaccard_mean']:.3f}",
                    f"{row['vote_only_mean']:.1f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze block_topk per-head counterfactual top-k recordings."
    )
    parser.add_argument(
        "--attempt-dir",
        type=Path,
        action="append",
        required=True,
        help="Attempt directory containing recordings/. Pass multiple to aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write per_head_topk_metrics.csv + summary.{json,md}.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="block_topk block size for block->role mapping (default 16, the "
        "project default; must match the recorded run).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Optional cap on iterations loaded (debugging).",
    )
    args = parser.parse_args(argv)

    attempt_dirs = find_attempt_dirs(args.attempt_dir)
    if not attempt_dirs:
        parser.error(f"no attempt dirs found under {args.attempt_dir}")
    records = load_iteration_records(attempt_dirs, max_iters=args.max_iters)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = analyze(records, block_size=args.block_size)
    summary = summarize(df)

    df.to_csv(args.output_dir / "per_head_topk_metrics.csv", index=False)
    (args.output_dir / "per_head_topk_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    md = _render_markdown(df, summary)
    (args.output_dir / "per_head_topk_summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {args.output_dir / 'per_head_topk_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'per_head_topk_summary.json'}")
    print(f"Wrote {args.output_dir / 'per_head_topk_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
