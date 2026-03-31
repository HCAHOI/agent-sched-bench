#!/usr/bin/env bash
set -euo pipefail

echo "ENV-2 model download script is not executable during BOOTSTRAP-0." >&2
echo "Choose one of the approved paths before execution:" >&2
echo "  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \\" >&2
echo "      --local-dir /data/models/Llama-3.1-8B-Instruct" >&2
echo "or" >&2
echo "  modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct \\" >&2
echo "      --local_dir /data/models/Llama-3.1-8B-Instruct" >&2
exit 1
