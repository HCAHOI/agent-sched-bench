#!/usr/bin/env bash
# Phase 6 of trace-sim-vastai-pipeline plan: full 4-cell matrix smoke +
# always-on BFCL refusal cell.
#
# Cells:
#   1. mini-swe × swe-bench-verified  — collect → simulate → gantt
#   2. mini-swe × swe-rebench         — collect → simulate → gantt
#   3. openclaw × swe-bench-verified --mcp-config configs/mcp/context7.yaml
#                                     — collect → simulate (P1.5.1) → gantt
#   4. openclaw × swe-rebench --mcp-config configs/mcp/context7.yaml
#                                     — collect → simulate (P1.5.1) → gantt
#   5. BFCL v4 refusal guard          — simulate exits with the exact
#                                       NotImplementedError message
#
# Cells 1-4 require infrastructure (cloud LLM API key, real vast.ai vLLM
# A100, podman bootstrap from US-006, etc.). When their preconditions
# are not met, the script reports "SKIPPED: <reason>" on stdout and
# moves on. Cell 5 (BFCL refusal) ALWAYS runs because it requires no
# infra — it's the canary that catches Phase 6 regressions in any
# environment, including the local Ralph loop.
#
# Per-cell exit codes flow into a matrix log at
# .omc/logs/smoke-matrix-$(date +%s).log so US-010 manual smoke
# verification can post-process them.

set -uo pipefail   # NOT -e: we want to continue past failed cells

SCRIPT_NAME="smoke_full_matrix"
log() { echo "[${SCRIPT_NAME}] $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p .omc/logs
LOG_FILE=".omc/logs/smoke-matrix-$(date +%s).log"
log "Matrix log: $LOG_FILE"

# Track per-cell results
declare -a CELL_NAMES=()
declare -a CELL_STATUS=()
declare -a CELL_DURATIONS=()
OVERALL_EXIT=0

run_cell() {
    local name="$1"
    shift
    local start
    start="$(date +%s)"
    log "──────────────────────────────────────────────"
    log "Cell: $name"
    log "  $*"

    if "$@" >>"$LOG_FILE" 2>&1; then
        local end
        end="$(date +%s)"
        local duration=$((end - start))
        log "  PASS (${duration}s)"
        CELL_NAMES+=("$name")
        CELL_STATUS+=("PASS")
        CELL_DURATIONS+=("${duration}")
        return 0
    else
        local rc=$?
        local end
        end="$(date +%s)"
        local duration=$((end - start))
        log "  FAIL (rc=${rc}, ${duration}s) — see $LOG_FILE"
        CELL_NAMES+=("$name")
        CELL_STATUS+=("FAIL")
        CELL_DURATIONS+=("${duration}")
        OVERALL_EXIT=1
        return "$rc"
    fi
}

skip_cell() {
    local name="$1"
    local reason="$2"
    log "──────────────────────────────────────────────"
    log "Cell: $name"
    log "  SKIPPED: $reason"
    CELL_NAMES+=("$name")
    CELL_STATUS+=("SKIPPED")
    CELL_DURATIONS+=("0")
}

# ─── Precondition checks ──
HAS_CLOUD_LLM_KEY=0
if [ -n "${OPENROUTER_API_KEY:-}" ] || [ -n "${DASHSCOPE_API_KEY:-}" ] || [ -n "${OPENAI_API_KEY:-}" ]; then
    HAS_CLOUD_LLM_KEY=1
fi

HAS_VLLM=0
if [ -n "${VLLM_URL:-}" ]; then
    if curl -sS --max-time 5 "${VLLM_URL%/}/health" >/dev/null 2>&1; then
        HAS_VLLM=1
    fi
fi

HAS_PODMAN=0
if command -v podman >/dev/null 2>&1; then
    if podman info >/dev/null 2>&1; then
        HAS_PODMAN=1
    fi
fi

HAS_CONTEXT7_KEY=0
if [ -n "${CONTEXT7_API_KEY:-}" ]; then
    HAS_CONTEXT7_KEY=1
fi

log "Preconditions:"
log "  cloud LLM key: $HAS_CLOUD_LLM_KEY"
log "  vLLM /health:  $HAS_VLLM"
log "  podman:        $HAS_PODMAN"
log "  context7 key:  $HAS_CONTEXT7_KEY"

# ─── Cell 1: mini-swe × swe-bench-verified ──
if [ "$HAS_CLOUD_LLM_KEY" = "1" ] && [ "$HAS_PODMAN" = "1" ]; then
    run_cell "mini-swe × swe-bench-verified" \
        conda run -n ML python -m trace_collect.cli \
        --benchmark swe-bench-verified \
        --scaffold mini-swe-agent \
        --sample 1 \
        --evaluate || true
else
    skip_cell "mini-swe × swe-bench-verified" \
        "needs cloud LLM key + podman; see docs/vastai_setup.md US-010 (a)"
fi

# ─── Cell 2: mini-swe × swe-rebench ──
if [ "$HAS_CLOUD_LLM_KEY" = "1" ] && [ "$HAS_PODMAN" = "1" ]; then
    run_cell "mini-swe × swe-rebench" \
        conda run -n ML python -m trace_collect.cli \
        --benchmark swe-rebench \
        --scaffold mini-swe-agent \
        --sample 1 \
        --evaluate || true
else
    skip_cell "mini-swe × swe-rebench" \
        "needs cloud LLM key + podman; see docs/vastai_setup.md US-010 (a)"
fi

# ─── Cell 3: openclaw × swe-bench-verified + context7 ──
if [ "$HAS_CLOUD_LLM_KEY" = "1" ] && [ "$HAS_PODMAN" = "1" ] && [ "$HAS_CONTEXT7_KEY" = "1" ]; then
    run_cell "openclaw × swe-bench-verified + context7" \
        conda run -n ML python -m trace_collect.cli \
        --benchmark swe-bench-verified \
        --scaffold openclaw \
        --mcp-config configs/mcp/context7.yaml \
        --sample 1 \
        --evaluate || true
else
    skip_cell "openclaw × swe-bench-verified + context7" \
        "needs cloud LLM key + podman + CONTEXT7_API_KEY; see docs/vastai_setup.md US-010 (d)"
fi

# ─── Cell 4: openclaw × swe-rebench + context7 ──
if [ "$HAS_CLOUD_LLM_KEY" = "1" ] && [ "$HAS_PODMAN" = "1" ] && [ "$HAS_CONTEXT7_KEY" = "1" ]; then
    run_cell "openclaw × swe-rebench + context7" \
        conda run -n ML python -m trace_collect.cli \
        --benchmark swe-rebench \
        --scaffold openclaw \
        --mcp-config configs/mcp/context7.yaml \
        --sample 1 \
        --evaluate || true
else
    skip_cell "openclaw × swe-rebench + context7" \
        "needs cloud LLM key + podman + CONTEXT7_API_KEY; see docs/vastai_setup.md US-010 (d)"
fi

# ─── Cell 5: BFCL v4 refusal guard (ALWAYS RUNS — no infra needed) ──
log "──────────────────────────────────────────────"
log "Cell 5: BFCL v4 refusal guard (always runs)"

# Use the synthetic fixture from tests/fixtures
BFCL_FIXTURE="tests/fixtures/bfcl_v4_minimal_header.jsonl"
if [ ! -f "$BFCL_FIXTURE" ]; then
    log "  FAIL: missing fixture $BFCL_FIXTURE"
    OVERALL_EXIT=1
    CELL_NAMES+=("BFCL v4 refusal guard")
    CELL_STATUS+=("FAIL")
    CELL_DURATIONS+=("0")
else
    # The simulator should exit nonzero with the exact message
    bfcl_output="$(conda run -n ML python -m trace_collect.cli simulate \
        --source-trace "$BFCL_FIXTURE" \
        --model dummy 2>&1 || true)"

    if echo "$bfcl_output" | grep -q "BFCL v4 traces have task_shape='function_call'"; then
        log "  PASS: BFCL fixture refused with the expected message"
        CELL_NAMES+=("BFCL v4 refusal guard")
        CELL_STATUS+=("PASS")
        CELL_DURATIONS+=("0")
    else
        log "  FAIL: BFCL fixture did NOT produce the expected refusal message"
        log "    output: ${bfcl_output:0:300}"
        OVERALL_EXIT=1
        CELL_NAMES+=("BFCL v4 refusal guard")
        CELL_STATUS+=("FAIL")
        CELL_DURATIONS+=("0")
    fi
fi

# ─── Summary ──
log "══════════════════════════════════════════════"
log "Smoke matrix summary"
log "══════════════════════════════════════════════"
for i in "${!CELL_NAMES[@]}"; do
    log "  ${CELL_STATUS[$i]} ${CELL_DURATIONS[$i]}s  ${CELL_NAMES[$i]}"
done
log ""
log "Detail log: $LOG_FILE"
log "Overall exit code: $OVERALL_EXIT"

exit $OVERALL_EXIT
