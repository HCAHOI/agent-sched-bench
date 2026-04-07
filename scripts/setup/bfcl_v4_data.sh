#!/usr/bin/env bash
# Download BFCL v4 (Berkeley Function-Calling Leaderboard v4) data and
# write a merged JSONL manifest at data/bfcl-v4/tasks.json.
#
# Usage:
#   conda activate ML
#   ./scripts/setup/bfcl_v4_data.sh
#
# Sources (tried in order; first success wins):
#   1. git clone (sparse) https://github.com/ShishirPatil/gorilla.git
#      → canonical source, always has the current v4 files
#   2. huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard
#      → mirror, may lag behind GitHub
#
# Output layout:
#   data/bfcl-v4/
#   ├── raw/                      # copied source files (one per category)
#   │   ├── BFCL_v4_simple.json
#   │   ├── possible_answer/
#   │   │   └── BFCL_v4_simple.json
#   │   └── ...
#   └── tasks.json                # merged JSONL used by BFCLv4Benchmark.load_tasks
#
# Idempotent: if tasks.json already exists, prints the row count and exits 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="data/bfcl-v4"
RAW_ROOT="${DATA_ROOT}/raw"
TASKS_FILE="${DATA_ROOT}/tasks.json"

if [[ -f "$TASKS_FILE" ]]; then
    count=$(wc -l < "$TASKS_FILE" | tr -d ' ')
    echo "[setup] SKIP bfcl_v4_data: ${TASKS_FILE} already exists (${count} rows)"
    exit 0
fi

mkdir -p "$RAW_ROOT"

download_via_git() {
    local tmpdir
    tmpdir="$(mktemp -d)"
    echo "[setup] Trying git sparse-clone of gorilla-llm/gorilla..."
    if ! git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/ShishirPatil/gorilla.git "$tmpdir/gorilla" 2>&1 | tail -5; then
        rm -rf "$tmpdir"
        return 1
    fi
    (
        cd "$tmpdir/gorilla"
        git sparse-checkout set berkeley-function-call-leaderboard/bfcl_eval/data
    ) || { rm -rf "$tmpdir"; return 1; }

    local src="$tmpdir/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
    if [[ ! -d "$src" ]]; then
        echo "[setup] ERROR: BFCL data directory not found in clone at $src" >&2
        rm -rf "$tmpdir"
        return 1
    fi
    echo "[setup] Copying BFCL v4 data files from $src to $RAW_ROOT..."
    cp -R "$src/." "$RAW_ROOT/"
    rm -rf "$tmpdir"
    return 0
}

download_via_hf() {
    echo "[setup] Trying huggingface-cli fallback..."
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "[setup] ERROR: huggingface-cli not installed; cannot fall back." >&2
        return 1
    fi
    local tmpdir
    tmpdir="$(mktemp -d)"
    if ! huggingface-cli download \
            gorilla-llm/Berkeley-Function-Calling-Leaderboard \
            --repo-type dataset \
            --local-dir "$tmpdir/hf" 2>&1 | tail -5; then
        rm -rf "$tmpdir"
        return 1
    fi
    echo "[setup] Copying BFCL data files from HF mirror to $RAW_ROOT..."
    cp -R "$tmpdir/hf/." "$RAW_ROOT/"
    rm -rf "$tmpdir"
    return 0
}

if ! download_via_git; then
    echo "[setup] git clone failed; trying HuggingFace fallback..."
    if ! download_via_hf; then
        echo "[setup] ERROR: both git and HuggingFace fallback failed." >&2
        echo "[setup]        Check network access, or manually place BFCL v4 JSONL files under ${RAW_ROOT}/." >&2
        exit 1
    fi
fi

# Count what we got before invoking the Python merger.
raw_files=$(find "$RAW_ROOT" -maxdepth 2 -name 'BFCL_v4_*.json' -not -path '*/possible_answer/*' | wc -l | tr -d ' ')
if [[ "$raw_files" -eq 0 ]]; then
    # HF mirror may still be at v3 — check for those too as a fallback.
    raw_files=$(find "$RAW_ROOT" -maxdepth 2 -name 'BFCL_v3_*.json' -not -path '*/possible_answer/*' | wc -l | tr -d ' ')
    if [[ "$raw_files" -gt 0 ]]; then
        echo "[setup] WARNING: only v3 files found in download — HF mirror may still be at v3. Proceeding with v3 files."
    else
        echo "[setup] ERROR: no BFCL_v4_*.json files found under ${RAW_ROOT}/." >&2
        echo "[setup]        Inspect ${RAW_ROOT}/ manually to diagnose." >&2
        exit 1
    fi
fi

echo "[setup] Found ${raw_files} BFCL category files; merging into ${TASKS_FILE}..."

# Delegate the merging logic to Python — more robust than bash JSON manipulation.
# Use the PYTHON env var if set, otherwise try `python`, then `python3`.
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "[setup] ERROR: neither python nor python3 found on PATH." >&2
        exit 1
    fi
fi
PYTHONPATH="${REPO_ROOT}/src" "$PYTHON_BIN" - <<'PYEOF'
import json
import re
from pathlib import Path
from collections import Counter

RAW_ROOT = Path("data/bfcl-v4/raw")
OUT = Path("data/bfcl-v4/tasks.json")

# Match either BFCL_v4_*.json or BFCL_v3_*.json (HF mirror fallback).
CATEGORY_RE = re.compile(r"^BFCL_v[34]_(?P<cat>.+)\.json$")

def iter_task_files(root: Path):
    # Top-level (non-possible_answer) files.
    for p in sorted(root.iterdir()):
        if p.is_file() and CATEGORY_RE.match(p.name):
            yield p

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[setup] WARNING: {path.name}:{lineno} malformed JSON ({exc}); skipping")
    return rows

category_counts: Counter[str] = Counter()
total = 0
with OUT.open("w", encoding="utf-8") as out_f:
    for task_file in iter_task_files(RAW_ROOT):
        m = CATEGORY_RE.match(task_file.name)
        assert m is not None
        category = m.group("cat")

        task_rows = load_jsonl(task_file)

        # Try matching possible_answer/<same_filename> (canonical location).
        gt_file = RAW_ROOT / "possible_answer" / task_file.name
        if not gt_file.exists():
            # HF mirror may flatten possible_answer/ alongside tasks.
            alt = RAW_ROOT / f"possible_answer_{task_file.name}"
            gt_file = alt if alt.exists() else gt_file

        gt_by_id: dict[str, list] = {}
        if gt_file.exists():
            for gt_row in load_jsonl(gt_file):
                rid = str(gt_row.get("id", ""))
                if rid:
                    gt_by_id[rid] = list(gt_row.get("ground_truth", []))

        for row in task_rows:
            rid = str(row.get("id", ""))
            merged = {
                "category": category,
                "id": rid,
                "question": row.get("question", []),
                "function": row.get("function", []),
                "ground_truth": gt_by_id.get(rid, []),
            }
            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            category_counts[category] += 1
            total += 1

print(f"[setup] Wrote {total} rows to {OUT}")
print("[setup] Per-category counts:")
for cat, count in sorted(category_counts.items()):
    print(f"[setup]   {cat}: {count}")
PYEOF

echo "[setup] bfcl_v4_data done"
