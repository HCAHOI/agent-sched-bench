#!/bin/bash
# Chain A (H1), fold-sharded. Steps 2 & 5 fan out 5 concurrent --only-fold procs
# then one --aggregate-only pass; steps 3,4,6,7 sequential. Numerics byte-identical
# to chain_A_h1.sh (orchestration-only change; proven by diff). Stops on first failure.
cd /home/chiyu/workspace/agent-sched-bench || exit 99
source .venv/bin/activate; export PYTHONPATH=src
B=analysis/fresh-corpus-certification-20260717
FR=traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200
FOLDS=5

run(){ n="$1"; shift; echo "[$(date +%H:%M:%S)] STEP $n START"; "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[$(date +%H:%M:%S)] STEP $n FAILED (exit=$rc)"; exit $rc; fi; echo "[$(date +%H:%M:%S)] STEP $n OK"; }

# shard_step <n> <script> <args...>   (args must NOT include --only-fold/--aggregate-only)
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

shard_step 2 scripts/certification/run_offline_gated_robust_confirmation.py \
  --manifest $B/offline-gated-robust/manifest.json --output-root $B/offline-gated-robust/results

run 3 python scripts/exploration/run_restore_cost_mode_b.py \
  --confirmation-root $B/offline-gated-robust/results --restore-cost-fractions 0.0,0.94 \
  --replicates 50000 --confidence-level 0.95 --seed 0 --output-root $B/restore-cost-mode-b

run 4 python scripts/exploration/run_within_task_baseline.py \
  --confirmation-root $B/offline-gated-robust/results --mode-b-root $B/restore-cost-mode-b \
  --restore-cost-fractions 0.0,0.94 --replicates 50000 --confidence-level 0.95 --seed 0 \
  --output-root $B/within-task-gated

shard_step 5 scripts/certification/run_hazard_model_confirmation.py \
  --confirmation-root $B/offline-gated-robust/results --mode-b-root $B/restore-cost-mode-b \
  --gated-b1-root $B/within-task-gated --restore-cost-fractions 0.0,0.94 \
  --num-intervals 40 --model-family gbm --feature-set full --ensemble-members 0 \
  --replicates 50000 --confidence-level 0.95 --seed 0 --output-root $B/hazard-model-gbm-full

run 6 python scripts/certification/analyze_certified_union.py \
  --hazard-root $B/hazard-model-gbm-full --restore-cost-fractions 0.0,0.94 \
  --inclusion-criterion loo_lcb --replicates 50000 --confidence-level 0.95 --seed 0 \
  --output-root $B/certified-union-loo-lcb

run 7 python scripts/certification/analyze_gate_robustness.py \
  --decisions $B/certified-union-loo-lcb/rho_0.94_decisions.jsonl --restore-cost-fraction 0.94 \
  --replicates 20000 --confidence-level 0.95 --seed 0 \
  --output $B/gate-robustness/gate_robustness_rho094.json

echo "CHAIN_A_H1_DONE $(date +%H:%M:%S)"
