#!/bin/bash
# Chain B rest: finish step 8 after the OOM (folds 2,5 in flight; rerun 1,3,4
# memory-gated max-2-concurrent), aggregate, then step 9. Appends to chain_B.log.
cd /home/chiyu/workspace/agent-sched-bench || exit 99
source .venv/bin/activate; export PYTHONPATH=src
B=analysis/fresh-corpus-certification-20260717
FR=traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200
MAXC=2; MINFREE_GB=7

FRONTIER_ARGS=(--config-manifest $B/offline-gated-robust/manifest.json --trace-root $FR --tool-name-trie \
  --num-intervals 40 --model-family gbm --feature-set full --ensemble-members 0 \
  --restore-cost-fractions 0.0,0.94 --replicates 50000 --confidence-level 0.95 --seed 0 \
  --exposure-note "Fresh SWE-ReBench seed42 skip150 n=277; disjoint from all dev roots; pre-registration 20260716." \
  --output-root $B/frontier-p1)

echo "[$(date +%H:%M:%S)] STEP 8 RESUME (folds 1,3,4 mem-gated maxc=$MAXC)"
# wait for the surviving in-flight folds (2,5) to finish
while pgrep -f "run_benchmark_frontier.py.*--only-fold" >/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] in-flight folds drained"

pids=(); folds=()
for f in 1 3 4; do
  # gate: bounded concurrency AND enough free memory for one ~5.5GB fold
  while [ "$(jobs -rp | wc -l)" -ge "$MAXC" ] || [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MINFREE_GB" ]; do sleep 15; done
  echo "[$(date +%H:%M:%S)] STEP 8 launching fold $f"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python scripts/run_benchmark_frontier.py "${FRONTIER_ARGS[@]}" --only-fold "$f" > "$B/step8_fold${f}.log" 2>&1 &
  pids+=($!); folds+=($f)
done
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "[$(date +%H:%M:%S)] STEP 8 fold ${folds[$i]} FAILED (see step8_fold${folds[$i]}.log)"; fail=1; fi
done
[ $fail -ne 0 ] && { echo "[$(date +%H:%M:%S)] STEP 8 FAILED (fold proc)"; exit 21; }

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 \
  python scripts/run_benchmark_frontier.py "${FRONTIER_ARGS[@]}" --aggregate-only
rc=$?; [ $rc -ne 0 ] && { echo "[$(date +%H:%M:%S)] STEP 8 FAILED aggregate (exit=$rc)"; exit $rc; }
echo "[$(date +%H:%M:%S)] STEP 8 OK"

echo "[$(date +%H:%M:%S)] STEP 9 START"
python scripts/analyze_frontier_permutation.py \
  --decisions $B/frontier-p1/rho_0.94_decisions.jsonl \
  --treatment-field offline_gated_robust_trigger_ms --baseline-field offline_gated_tool_name_trigger_ms \
  --restore-cost-fraction 0.94 --replicates 20000 --seed 0 \
  --output $B/frontier-p1/permutation_p1_rho094.json
rc=$?; [ $rc -ne 0 ] && { echo "[$(date +%H:%M:%S)] STEP 9 FAILED (exit=$rc)"; exit $rc; }
echo "[$(date +%H:%M:%S)] STEP 9 OK"
echo "CHAIN_B_H2_DONE $(date +%H:%M:%S)"
