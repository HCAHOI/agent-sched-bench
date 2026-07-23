"""DistilBERT + MLP quantile predictor for per-call tool resources.

The encoder produces a pooled sentence embedding; numeric causal features are
concatenated to it; per-target 3-layer MLP heads emit p50/p90/p99 quantiles
trained with pinball loss. Text-only / numeric-only ablations drop one branch.
Torch lives here only; :mod:`tool_resource.bert_dataset` stays torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from tool_resource.bert_dataset import TARGET_NAMES, ResourceExample


@dataclass
class BertModelConfig:
    encoder_name: str = "distilbert-base-uncased"
    numeric_dim: int = 0
    target_names: tuple[str, ...] = TARGET_NAMES
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.99)
    hidden_dim: int = 256
    dropout: float = 0.1
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    text_only: bool = False
    numeric_only: bool = False

    def __post_init__(self) -> None:
        if self.text_only and self.numeric_only:
            raise ValueError("text_only and numeric_only are mutually exclusive")


def pinball_loss_torch(
    error: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """Pinball loss for error = observation - prediction, broadcast over quantiles.

    Formula matches ``tool_resource.metrics.pinball_loss``:
    ``max(q * e, (q - 1) * e)`` (re-implemented for tensors, metrics unchanged).
    """

    return torch.maximum(quantiles * error, (quantiles - 1.0) * error)


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class ToolResourceBert(nn.Module):
    """Multi-target quantile regressor over pooled text + numeric features."""

    def __init__(self, config: BertModelConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "_quantiles", torch.tensor(config.quantiles, dtype=torch.float32)
        )

        encoder_dim = 0
        self.encoder: nn.Module | None = None
        if not config.numeric_only:
            # token=False: skip implicit HF auth (a stale local token breaks it);
            # weights load from the local cache.
            self.encoder = AutoModel.from_pretrained(config.encoder_name, token=False)
            encoder_dim = self.encoder.config.hidden_size
            if config.use_lora:
                self.encoder = _wrap_lora(self.encoder, config)

        numeric_dim = 0 if config.text_only else config.numeric_dim
        self._head_in_dim = encoder_dim + numeric_dim
        if self._head_in_dim == 0:
            raise ValueError("model has neither a text nor a numeric branch")

        self.heads = self._build_heads()

    @classmethod
    def heads_only(cls, in_dim: int, config: BertModelConfig) -> "ToolResourceBert":
        """Build just the MLP heads for training on cached features.

        No encoder is loaded: features (pooled text + numeric) are precomputed
        once by the caller, so the frozen-encoder run reuses one cache across
        folds and ablations instead of re-encoding.
        """

        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.config = config
        model.register_buffer(
            "_quantiles", torch.tensor(config.quantiles, dtype=torch.float32)
        )
        model.encoder = None
        model._head_in_dim = in_dim
        model.heads = model._build_heads()
        return model

    def _build_heads(self) -> nn.ModuleDict:
        n_q = len(self.config.quantiles)
        return nn.ModuleDict(
            {
                name: _mlp(
                    self._head_in_dim, self.config.hidden_dim, n_q, self.config.dropout
                )
                for name in self.config.target_names
            }
        )

    def reinit_heads(self) -> None:
        """Re-initialise MLP heads (stage-2 recipe: fresh heads, frozen encoder)."""

        self.heads = self._build_heads().to(self._quantiles.device)

    def freeze_encoder(self) -> None:
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def encode(
        self, input_ids: torch.Tensor | None, attention_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        if self.encoder is None:
            return None
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return _mean_pool(out.last_hidden_state, attention_mask)

    def encode_features(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        numeric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenated pooled-text + numeric features fed to the heads.

        With the encoder frozen (stage 2) these are constant, so training caches
        them once instead of re-running the encoder every epoch.
        """

        parts: list[torch.Tensor] = []
        pooled = self.encode(input_ids, attention_mask)
        if pooled is not None:
            parts.append(pooled)
        if not self.config.text_only and self.config.numeric_dim > 0:
            if numeric is None:
                raise ValueError("numeric features required but not provided")
            parts.append(numeric)
        return torch.cat(parts, dim=-1)

    def heads_forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(features) for name, head in self.heads.items()}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        numeric: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.heads_forward(
            self.encode_features(input_ids, attention_mask, numeric)
        )

    def loss(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Mean pinball loss over targets that have any unmasked rows."""

        quantiles = self._quantiles.view(1, -1)
        total = torch.zeros((), device=self._quantiles.device)
        n_targets = 0
        for name in preds:
            mask = masks[name]
            if not bool(mask.any()):
                continue
            error = targets[name][mask].unsqueeze(1) - preds[name][mask]
            total = total + pinball_loss_torch(error, quantiles).mean()
            n_targets += 1
        return total / max(n_targets, 1)


def _wrap_lora(encoder: nn.Module, config: BertModelConfig) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention projections
        bias="none",
    )
    return get_peft_model(encoder, lora)


def load_tokenizer(encoder_name: str = "distilbert-base-uncased"):
    return AutoTokenizer.from_pretrained(encoder_name, token=False)


def collate_examples(
    batch: Sequence[ResourceExample],
    tokenizer,
    target_names: Sequence[str] = TARGET_NAMES,
    *,
    use_text: bool = True,
    max_length: int = 256,
) -> dict[str, object]:
    """Tensorise a batch of examples; NaN targets become 0 under a False mask."""

    out: dict[str, object] = {
        "numeric": torch.tensor([ex.numeric for ex in batch], dtype=torch.float32),
    }
    if use_text:
        enc = tokenizer(
            [ex.text for ex in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out["input_ids"] = enc["input_ids"]
        out["attention_mask"] = enc["attention_mask"]

    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for name in target_names:
        flags = [ex.target_mask[name] for ex in batch]
        values = [
            ex.targets[name] if ex.target_mask[name] else 0.0 for ex in batch
        ]
        targets[name] = torch.tensor(values, dtype=torch.float32)
        masks[name] = torch.tensor(flags, dtype=torch.bool)
    out["targets"] = targets
    out["masks"] = masks
    return out


__all__ = [
    "BertModelConfig",
    "ToolResourceBert",
    "pinball_loss_torch",
    "load_tokenizer",
    "collate_examples",
]
