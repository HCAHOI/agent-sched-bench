"""Two-stage training for the BERT tool-resource quantile predictor.

Recipe per repo-disjoint fold:
  stage 1 -- joint finetune of encoder + MLP heads (default 3 epochs)
  stage 2 -- re-init MLP heads, freeze the encoder, train heads alone
             (default 150 epochs)

Plain torch loop, no trainer framework. Writes a checkpoint and a metrics JSON
per fold to ``--out``. Ablations ``--text-only`` / ``--numeric-only`` drop one
input branch. ``--smoke`` runs a tiny subset with 1 + 5 epochs to validate
plumbing (not evidence).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_resource.bert_dataset import (  # noqa: E402
    DEFAULT_TASKS_JSON,
    TARGET_INVERSE,
    TARGET_NAMES,
    ResourceExample,
    build_merged_dataset,
)
from tool_resource.bert_model import (  # noqa: E402
    BertModelConfig,
    ToolResourceBert,
    collate_examples,
    load_tokenizer,
    pinball_loss_torch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {k: v.to(device) for k, v in value.items()}
        else:
            moved[key] = value.to(device)
    return moved


def _make_loader(
    examples: list[ResourceExample],
    tokenizer,
    target_names,
    use_text: bool,
    batch_size: int,
    shuffle: bool,
    max_length: int,
) -> DataLoader:
    def collate(batch):
        return collate_examples(
            batch,
            tokenizer,
            target_names,
            use_text=use_text,
            max_length=max_length,
        )

    return DataLoader(
        examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
    )


def _run_epochs(
    model: ToolResourceBert,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    if epochs == 0:
        return
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = _to_device(batch, device)
            preds = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                numeric=batch["numeric"],
            )
            loss = model.loss(preds, batch["targets"], batch["masks"])
            if not loss.requires_grad:  # batch has no unmasked target -> no signal
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()


@torch.no_grad()
def _precompute_features(
    model: ToolResourceBert,
    loader: DataLoader,
    target_names: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run the (frozen) encoder once and cache pooled+numeric features.

    Stage 2 keeps the encoder frozen, so its features never change across the
    150 head-only epochs. Caching them turns those epochs from encoder forwards
    (hours on CPU) into pure MLP passes (seconds).
    """

    model.eval()
    feats: list[torch.Tensor] = []
    targ: dict[str, list[torch.Tensor]] = {n: [] for n in target_names}
    msk: dict[str, list[torch.Tensor]] = {n: [] for n in target_names}
    for batch in loader:
        batch = _to_device(batch, device)
        feats.append(
            model.encode_features(
                batch.get("input_ids"), batch.get("attention_mask"), batch["numeric"]
            ).clone()  # avoid pinning each batch's full activation (see _pooled_cache)
        )
        for name in target_names:
            targ[name].append(batch["targets"][name])
            msk[name].append(batch["masks"][name])
    features = torch.cat(feats)
    targets = {n: torch.cat(targ[n]) for n in target_names}
    masks = {n: torch.cat(msk[n]) for n in target_names}
    return features, targets, masks


def _run_head_epochs(
    model: ToolResourceBert,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Train MLP heads on cached features (encoder frozen and not re-run)."""

    if epochs == 0:
        return
    params = list(model.heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = features.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=features.device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            preds = model.heads_forward(features[idx])
            loss = model.loss(
                preds,
                {k: v[idx] for k, v in targets.items()},
                {k: v[idx] for k, v in masks.items()},
            )
            if not loss.requires_grad:  # batch has no unmasked target -> no signal
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()


@torch.no_grad()
def _evaluate(
    model: ToolResourceBert,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    device: torch.device,
    quantiles: tuple[float, ...],
) -> dict:
    """Report pinball in ORIGINAL units (exponentiate log-quantiles; monotone so
    quantiles are preserved) and q-error of the p50 head, per target."""

    model.eval()
    preds = model.heads_forward(features)
    qs = list(quantiles)
    i50, i90 = qs.index(0.5), qs.index(0.9)
    metrics = {}
    for name in TARGET_NAMES:
        mask = masks[name]
        if not bool(mask.any()):
            continue
        inv, unit = TARGET_INVERSE[name]
        f = torch.expm1 if inv == "expm1" else torch.exp
        log_actual = targets[name][mask]
        log_pred = preds[name][mask]                 # [n, n_quantiles], log space
        actual = f(log_actual)                       # original units
        pinball = {}
        for tag, qi in (("p50", i50), ("p90", i90)):
            err = actual - f(log_pred[:, qi])
            pinball[tag] = float(pinball_loss_torch(err, torch.tensor(qs[qi])).mean())
        # q-error of the p50 head = geometric error = exp(|log_pred_p50 - log_actual|)
        qerr = torch.exp((log_pred[:, i50] - log_actual).abs())
        metrics[name] = {
            "n": int(mask.sum().item()),
            "unit": unit,
            "pinball_orig": pinball,
            "qerror_p50": {
                "median": float(qerr.median()),
                "p90": float(qerr.quantile(0.9)),
                "max": float(qerr.max()),
            },
        }
    return metrics


def _stack_targets(
    examples: list[ResourceExample], target_names, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    targets = {
        n: torch.tensor(
            [e.targets[n] if e.target_mask[n] else 0.0 for e in examples],
            dtype=torch.float32, device=device,
        )
        for n in target_names
    }
    masks = {
        n: torch.tensor(
            [e.target_mask[n] for e in examples], dtype=torch.bool, device=device
        )
        for n in target_names
    }
    return targets, masks


@torch.no_grad()
def _pooled_cache(dataset, args, device: torch.device) -> torch.Tensor:
    """Pooled text embeddings for every example, computed once (frozen encoder).

    Encoding ~13k examples costs ~22 min on CPU, so this must happen once and be
    reused across folds and across the full/text-only ablations. With
    ``--embedding-cache`` it also persists to disk, keyed by text content,
    encoder, and max_length.
    """

    import hashlib

    text_hash = hashlib.sha1(
        "\x00".join(e.text for e in dataset.examples).encode("utf-8")
    ).hexdigest()
    cache_path = args.embedding_cache
    if cache_path and Path(cache_path).exists():
        blob = torch.load(cache_path)
        if (
            blob.get("text_hash") == text_hash
            and blob.get("encoder") == args.encoder
            and blob.get("max_length") == args.max_length
        ):
            print(f"embedding cache hit: {cache_path}", flush=True)
            return blob["pooled"].to(device)

    tokenizer = load_tokenizer(args.encoder)
    config = BertModelConfig(encoder_name=args.encoder, numeric_dim=dataset.numeric_dim)
    model = ToolResourceBert(config).to(device)
    model.eval()
    loader = _make_loader(
        dataset.examples, tokenizer, args.targets, True,
        args.batch_size, False, args.max_length,
    )
    n_batches = len(loader)
    print(f"encoding {len(dataset.examples)} examples in {n_batches} batches", flush=True)
    pooled_chunks = []
    for i, batch in enumerate(loader):
        batch = _to_device(batch, device)
        # .clone(): the pooled slice otherwise pins the whole [B,L,768] batch
        # activation, leaking ~12 MB/batch -> OOM at ~batch 800 of the full run.
        pooled_chunks.append(
            model.encode(batch["input_ids"], batch["attention_mask"]).clone()
        )
        if (i + 1) % 50 == 0 or i + 1 == n_batches:
            print(f"  encoded batch {i + 1}/{n_batches}", flush=True)
    pooled = torch.cat(pooled_chunks)
    if cache_path:
        torch.save(
            {"text_hash": text_hash, "encoder": args.encoder,
             "max_length": args.max_length, "pooled": pooled.cpu()},
            cache_path,
        )
    return pooled


def _assemble_features(
    ablation: str, pooled: torch.Tensor | None, numeric: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    if ablation == "numeric_only":
        return numeric[idx]
    if ablation == "text_only":
        return pooled[idx]
    return torch.cat([pooled[idx], numeric[idx]], dim=1)  # full


def _train_frozen(dataset, args, device: torch.device, folds: list[int]) -> dict:
    """Frozen pretrained encoder: encode once, then head-only MLP per ablation/fold."""

    ablations = (
        ["full", "text_only", "numeric_only"]
        if args.run_ablations else [_single_ablation(args)]
    )
    need_text = any(a in ("full", "text_only") for a in ablations)
    pooled = _pooled_cache(dataset, args, device) if need_text else None
    numeric = torch.tensor(
        [e.numeric for e in dataset.examples], dtype=torch.float32, device=device
    )
    targets, masks = _stack_targets(dataset.examples, args.targets, device)
    repos = [e.repo for e in dataset.examples]
    quantiles = BertModelConfig().quantiles
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {}
    for ablation in ablations:
        fold_results = []
        for fold in folds:
            val_repos = {r for r, f in dataset.repo_folds.items() if f == fold}
            val_flag = torch.tensor([r in val_repos for r in repos])
            tr = (~val_flag).nonzero(as_tuple=True)[0]
            va = val_flag.nonzero(as_tuple=True)[0]

            # Standardize numeric features on TRAIN-fold stats only (no leakage).
            # Features span ~0/1 flags to memory in the thousands; without this the
            # numeric-only head's AdamW diverges on some folds. Std floored so the
            # constant column (ambient_before_present) maps to 0 instead of blowing up.
            num_mean = numeric[tr].mean(0)
            num_std = numeric[tr].std(0).clamp(min=1e-6)
            numeric_z = (numeric - num_mean) / num_std

            feats_tr = _assemble_features(ablation, pooled, numeric_z, tr)
            feats_va = _assemble_features(ablation, pooled, numeric_z, va)
            config = BertModelConfig(
                numeric_dim=dataset.numeric_dim,
                target_names=tuple(args.targets),
                hidden_dim=args.hidden_dim,
                text_only=(ablation == "text_only"),
                numeric_only=(ablation == "numeric_only"),
            )
            model = ToolResourceBert.heads_only(feats_tr.shape[1], config).to(device)
            _run_head_epochs(
                model, feats_tr,
                {k: v[tr] for k, v in targets.items()},
                {k: v[tr] for k, v in masks.items()},
                args.stage2_epochs, args.head_lr, args.head_batch_size,
            )
            metrics = _evaluate(
                model, feats_va,
                {k: v[va] for k, v in targets.items()},
                {k: v[va] for k, v in masks.items()},
                device, quantiles,
            )
            torch.save(
                {"heads": model.heads.state_dict(),
                 "numeric_mean": num_mean.cpu(), "numeric_std": num_std.cpu()},
                out_dir / f"{ablation}_fold{fold}_heads.pt",
            )
            fold_results.append({
                "fold": fold,
                "n_train": int(tr.numel()),
                "n_val": int(va.numel()),
                "val_repos": sorted(val_repos),
                "metrics": metrics,
            })
            print(f"done: ablation={ablation} fold={fold} "
                  f"n_train={int(tr.numel())} n_val={int(va.numel())}", flush=True)
        results[ablation] = fold_results
    return results


def _single_ablation(args) -> str:
    if args.text_only:
        return "text_only"
    if args.numeric_only:
        return "numeric_only"
    return "full"


def train_fold(
    dataset,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    train_examples, val_examples = dataset.fold_split(fold)
    use_text = not args.numeric_only
    tokenizer = load_tokenizer(args.encoder) if use_text else None

    config = BertModelConfig(
        encoder_name=args.encoder,
        numeric_dim=dataset.numeric_dim,
        target_names=tuple(args.targets),
        hidden_dim=args.hidden_dim,
        use_lora=args.lora,
        text_only=args.text_only,
        numeric_only=args.numeric_only,
    )
    model = ToolResourceBert(config).to(device)

    train_loader = _make_loader(
        train_examples, tokenizer, args.targets, use_text,
        args.batch_size, True, args.max_length,
    )
    val_loader = _make_loader(
        val_examples, tokenizer, args.targets, use_text,
        args.batch_size, False, args.max_length,
    )

    # Stage 1: joint finetune of everything trainable.
    _run_epochs(model, train_loader, device, args.stage1_epochs, args.lr)
    # Stage 2: fresh heads, frozen encoder, heads-only training on cached
    # features so the encoder runs once, not once per epoch.
    model.reinit_heads()
    model.freeze_encoder()
    train_feats = _precompute_features(model, train_loader, args.targets, device)
    _run_head_epochs(
        model, *train_feats, args.stage2_epochs, args.head_lr, args.head_batch_size
    )

    val_feats = _precompute_features(model, val_loader, args.targets, device)
    metrics = _evaluate(model, *val_feats, device, config.quantiles)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"fold{fold}_model.pt")
    fold_result = {
        "fold": fold,
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "val_repos": sorted(
            {r for r, f in dataset.repo_folds.items() if f == fold}
        ),
        "metrics": metrics,
    }
    (out_dir / f"fold{fold}_metrics.json").write_text(
        json.dumps(fold_result, indent=2), encoding="utf-8"
    )
    return fold_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus", type=Path, action="append", required=True,
        help="trace root dir; repeat with a matching --manifest to merge corpora",
    )
    p.add_argument(
        "--manifest", type=Path, action="append", required=True,
        help="corpus task manifest; paired positionally with --corpus",
    )
    p.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS_JSON)
    p.add_argument("--out", type=Path, required=True, help="output dir")
    p.add_argument(
        "--targets", nargs="+", default=list(TARGET_NAMES), choices=list(TARGET_NAMES)
    )
    p.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="fold indices to run (default: all)",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-prev", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--numeric-only", action="store_true")
    p.add_argument("--encoder", default="distilbert-base-uncased")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--head-batch-size", type=int, default=256,
        help="batch size for head-only training on cached features; larger than "
             "--batch-size (which is encoder/memory-bound) since heads are tiny",
    )
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--stage1-epochs", type=int, default=3)
    p.add_argument("--stage2-epochs", type=int, default=150)
    p.add_argument("--limit-tasks", type=int, default=None)
    p.add_argument(
        "--num-threads", type=int, default=max(1, (os.cpu_count() or 8) // 2),
        help="torch CPU threads (default: half the cores, shared host)",
    )
    p.add_argument(
        "--run-ablations", action="store_true",
        help="run full/text_only/numeric_only in one invocation, sharing the "
             "frozen-encoder embedding cache; implies --stage1-epochs 0",
    )
    p.add_argument(
        "--embedding-cache", type=Path, default=None,
        help="persist/reuse pooled embeddings across runs (frozen encoder only)",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="tiny subset, 1 + 5 epochs; plumbing check only, not evidence",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if len(args.corpus) != len(args.manifest):
        raise SystemExit("--corpus and --manifest must be given in matching pairs")
    if args.run_ablations:
        # Sharing one embedding cache across ablations is only valid with a
        # frozen encoder; finetuning would make each ablation's embeddings differ.
        args.stage1_epochs = 0
    if args.smoke:
        args.limit_tasks = args.limit_tasks or 4
        if not args.run_ablations:
            args.stage1_epochs = 1
        args.stage2_epochs = 5
        args.folds = args.folds or [0]
        args.batch_size = min(args.batch_size, 8)

    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.embedding_cache:
        args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_merged_dataset(
        list(zip(args.corpus, args.manifest)),
        tasks_json=args.tasks_json,
        n_prev=args.n_prev,
        n_folds=args.n_folds,
        seed=args.seed,
        limit_tasks=args.limit_tasks,
    )
    print(
        f"dataset built: n_examples={len(dataset.examples)} "
        f"n_repos={len(set(dataset.repo_folds))} n_folds={args.n_folds}",
        flush=True,
    )
    folds = args.folds if args.folds is not None else list(range(args.n_folds))

    summary = {
        "corpora": [
            {"corpus": str(c), "manifest": str(m)}
            for c, m in zip(args.corpus, args.manifest)
        ],
        "n_examples": len(dataset.examples),
        "n_repos": len(set(dataset.repo_folds)),
        "targets": args.targets,
        "seed": args.seed,
    }
    frozen = args.stage1_epochs == 0
    if frozen:
        summary["mode"] = "frozen_pretrained_encoder"
        summary["ablations"] = _train_frozen(dataset, args, device, folds)
    else:
        summary["mode"] = "finetune"
        summary["ablation"] = _single_ablation(args)
        summary["folds"] = [train_fold(dataset, fold, args, device) for fold in folds]

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"out": str(out_dir), "mode": summary["mode"],
                      "n_examples": len(dataset.examples), "folds": folds}))


if __name__ == "__main__":
    main()
