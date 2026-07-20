#!/bin/bash
# Chain B (H2), fold-sharded. Step 8 fans out 5 concurrent --only-fold procs then
# one --aggregate-only pass (byte-identical to sequential; proven by diff, 54/54
# files). Step 9 sequential. Stops on first failure.
cd /home/chiyu/workspace/agent-sched-bench || exit 99
source .venv/bin/activate; export PYTHONPATH=src
B=analysis/fresh-corpus-certification-20260717
FR=traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200
FOLDS=5

run(){ n="$1"; shift; echo "[$(date +%H:%M:%S)] STEP $n START"; "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[$(date +%H:%M:%S)] STEP $n FAILED (exit=$rc)"; exit $rc; fi; echo "[$(date +%H:%M:%S)] STEP $n OK"; }

shard_step(){
  n="$1"; shift; script="$1"; shift
  echo "[$(date +%H:%M:%S)] STEP $n START (sharded x$FOLDS folds)"
  pids=()
  for f in $(seq 1 "$FOLDS"); do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      python "$script" "$@" --only-fold "$f" > "$B/step${n}_fold${f}.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then echo "[$(date +%H:%M:%S)] STEP $n fold $((i+1)) FAILED (see step${n}_fold$((i+1)).log)"; fail=1; fi
  done
  if [ $fail -ne 0 ]; then echo "[$(date +%H:%M:%S)] STEP $n FAILED (fold proc)"; exit 21; fi
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 \
    python "$script" "$@" --aggregate-only; rc=$?
  if [ $rc -ne 0 ]; then echo "[$(date +%H:%M:%S)] STEP $n FAILED aggregate (exit=$rc)"; exit $rc; fi
  echo "[$(date +%H:%M:%S)] STEP $n OK"
}

shard_step 8 scripts/run_benchmark_frontier.py \
  --config-manifest $B/offline-gated-robust/manifest.json --trace-root $FR --tool-name-trie \
  --num-intervals 40 --model-family gbm --feature-set full --ensemble-members 0 \
  --restore-cost-fractions 0.0,0.94 --replicates 50000 --confidence-level 0.95 --seed 0 \
  --exposure-note "Fresh SWE-ReBench seed42 skip150 n=277; disjoint from all dev roots; pre-registration 20260716." \
  --output-root $B/frontier-p1

run 9 python scripts/analyze_frontier_permutation.py \
  --decisions $B/frontier-p1/rho_0.94_decisions.jsonl \
  --treatment-field offline_gated_robust_trigger_ms --baseline-field offline_gated_tool_name_trigger_ms \
  --restore-cost-fraction 0.94 --replicates 20000 --seed 0 \
  --output $B/frontier-p1/permutation_p1_rho094.json

echo "CHAIN_B_H2_DONE $(date +%H:%M:%S)"
