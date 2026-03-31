from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def safe_version(package: str) -> str | None:
    """Return the installed package version when available."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_file_inventory(model_path: Path) -> dict[str, Any]:
    """Collect a shallow inventory of files within the downloaded model directory."""
    files = sorted(path.relative_to(model_path).as_posix() for path in model_path.rglob("*") if path.is_file())
    total_size_bytes = sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())
    return {
        "file_count": len(files),
        "total_size_bytes": total_size_bytes,
        "sample_files": files[:20],
    }


def load_transformers_metadata(model_path: Path, verify_load_mode: str) -> dict[str, Any]:
    """Load model configuration and optionally the full model weights."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    metadata: dict[str, Any] = {
        "config_class": config.__class__.__name__,
        "architectures": getattr(config, "architectures", None),
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "torch_dtype": str(getattr(config, "torch_dtype", None)),
        "verified_load_mode": verify_load_mode,
    }

    if verify_load_mode == "full":
        model = AutoModelForCausalLM.from_pretrained(model_path)
        metadata["model_class"] = model.__class__.__name__
        metadata["state_dict_size"] = len(model.state_dict())
        del model
    elif verify_load_mode != "config":
        raise ValueError(f"Unsupported verify_load_mode: {verify_load_mode}")

    return metadata


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Build the ENV-2 verification report."""
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    report = {
        "model_path": str(model_path.resolve()),
        "backend": args.backend,
        "model_repo": args.model_repo,
        "modelscope_model": args.modelscope_model,
        "verify_load_mode": args.verify_load_mode,
        "acceptance_ready": args.verify_load_mode == "full",
        "expected": {
            "hidden_size": args.expected_hidden_size,
            "num_layers": args.expected_num_layers,
        },
        "package_specs": {
            "transformers": args.transformers_spec,
            "huggingface_hub": args.hf_hub_spec,
            "modelscope": args.modelscope_spec,
        },
        "installed_versions": {
            "transformers": safe_version("transformers"),
            "huggingface_hub": safe_version("huggingface_hub"),
            "modelscope": safe_version("modelscope"),
            "torch": safe_version("torch"),
        },
        "inventory": collect_file_inventory(model_path),
        "transformers_metadata": load_transformers_metadata(model_path, args.verify_load_mode),
    }
    return report


def validate_report(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Validate report contents against the expected model characteristics."""
    errors: list[str] = []
    metadata = report["transformers_metadata"]
    if metadata["hidden_size"] != args.expected_hidden_size:
        errors.append(
            f"hidden_size mismatch: expected {args.expected_hidden_size}, got {metadata['hidden_size']}"
        )
    if metadata["num_hidden_layers"] != args.expected_num_layers:
        errors.append(
            "num_hidden_layers mismatch: "
            f"expected {args.expected_num_layers}, got {metadata['num_hidden_layers']}"
        )
    if report["inventory"]["file_count"] == 0:
        errors.append("downloaded model directory was empty")
    if args.verify_load_mode == "full" and "model_class" not in metadata:
        errors.append("full model verification did not record a loaded model class")
    return errors


def enforce_acceptance_mode(args: argparse.Namespace) -> None:
    """Reject non-acceptance verification modes when strict validation is requested."""
    if args.fail_on_mismatch and args.verify_load_mode != "full":
        raise SystemExit(
            "ENV-2 acceptance requires --verify-load-mode=full when --fail-on-mismatch is set"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a downloaded model artifact and emit an audit report."
    )
    parser.add_argument("--output", required=True, help="Path to the JSON report.")
    parser.add_argument("--model-path", required=True, help="Local model directory.")
    parser.add_argument("--backend", required=True, help="Download backend used.")
    parser.add_argument("--model-repo", required=True, help="HuggingFace repo id.")
    parser.add_argument(
        "--modelscope-model",
        required=True,
        help="ModelScope model id used for the alternative backend.",
    )
    parser.add_argument(
        "--verify-load-mode",
        default="full",
        choices=["full", "config"],
        help="Whether to load config only or fully instantiate the model.",
    )
    parser.add_argument(
        "--expected-hidden-size",
        type=int,
        default=4096,
        help="Expected hidden size for the target model.",
    )
    parser.add_argument(
        "--expected-num-layers",
        type=int,
        default=32,
        help="Expected number of hidden layers for the target model.",
    )
    parser.add_argument("--transformers-spec", required=True)
    parser.add_argument("--hf-hub-spec", required=True)
    parser.add_argument("--modelscope-spec", required=True)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when the report does not match the expected model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_acceptance_mode(args)
    report = build_report(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.fail_on_mismatch:
        errors = validate_report(report, args)
        if errors:
            raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
