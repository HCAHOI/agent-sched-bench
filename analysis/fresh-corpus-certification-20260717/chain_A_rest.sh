#!/bin/bash
# Chain A rest: after step-2 folds drain, aggregate step 2, run steps 3,4,
# step 5 sharded MEMORY-GATED (max 2 concurrent GBM folds; 5-way OOM'd chain B),
# then steps 6,7. Appends to chain_A.log. Replaces the killed chain_A_sharded.sh wrapper.
cd /home/chiyu/workspace/agent-sched-bench || exit 99
source .venv/bin/activate; export PYTHONPATH=src
B=analysis/fresh-corpus-certification-20260717
MAXC=2; MINFREE_GB=7; FOLDS=5

run(){ n="$1"; shift; echo "[$(date +%H:%M:%S)] STEP $n START"; "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[$(date +%H:%M:%S)] STEP $n FAILED (exit=$rc)"; exit $rc; fi; echo "[$(date +%H:%M:%S)] STEP $n OK"; }

# --- step 2: wait for the 5 in-flight fold procs, verify outputs, aggregate ---
echo "[$(date +%H:%M:%S)] STEP 2 waiting for in-flight folds"
while pgrep -f "run_offline_gated_robust_confirmation.py.*--only-fold" >/dev/null; do sleep 20; done
n_done=$(ls -1 $B/offline-gated-robust/results/cv/f*_decisions.jsonl 2>/dev/null | wc -l)
if [ "$n_done" -ne "$FOLDS" ]; then echo "[$(date +%H:%M:%S)] STEP 2 FAILED (only $n_done/$FOLDS fold outputs)"; exit 22; fi
run 2-agg env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 \
  python scripts/run_offline_gated_robust_confirmation.py \
  --manifest $B/offline-gated-robust/manifest.json --output-root $B/offline-gated-robust/results \
  --aggregate-only

run 3 python scripts/run_restore_cost_mode_b.py \
  --confirmation-root $B/offline-gated-robust/results --restore-cost-fractions 0.0,0.94 \
  --replicates 50000 --confidence-level 0.95 --seed 0 --output-root $B/restore-cost-mode-b

run 4 python scripts/run_within_task_baseline.py \
  --confirmation-root $B/offline-gated-robust/results --mode-b-root $B/restore-cost-mode-b \
  --restore-cost-fractions 0.0,0.94 --replicates 50000 --confidence-level 0.95 --seed 0 \
  --output-root $B/within-task-gated

# --- step 5: sharded, memory-gated max-2 ---
HAZ_ARGS=(--confirmation-root $B/offline-gated-robust/results --mode-b-root $B/restore-cost-mode-b \
  --gated-b1-root $B/within-task-gated --restore-cost-fractions 0.0,0.94 \
  --num-intervals 40 --model-family gbm --feature-set full --ensemble-members 0 \
  --replicates 50000 --confidence-level 0.95 --seed 0 --output-root $B/hazard-model-gbm-full)
echo "[$(date +%H:%M:%S)] STEP 5 START (sharded, mem-gated maxc=$MAXC)"
pids=(); folds=()
for f in $(seq 1 $FOLDS); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXC" ] || [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MINFREE_GB" ]; do sleep 15; done
  echo "[$(date +%H:%M:%S)] STEP 5 launching fold $f"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python scripts/run_hazard_model_confirmation.py "${HAZ_ARGS[@]}" --only-fold "$f" > "$B/step5_fold${f}.log" 2>&1 &
  pids+=($!); folds+=($f)
done
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "[$(date +%H:%M:%S)] STEP 5 fold ${folds[$i]} FAILED (see step5_fold${folds[$i]}.log)"; fail=1; fi
done
[ $fail -ne 0 ] && { echo "[$(date +%H:%M:%S)] STEP 5 FAILED (fold proc)"; exit 21; }
run 5-agg env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 \
  python scripts/run_hazard_model_confirmation.py "${HAZ_ARGS[@]}" --aggregate-only

run 6 python scripts/analyze_certified_union.py \
  --hazard-root $B/hazard-model-gbm-full --restore-cost-fractions 0.0,0.94 \
  --inclusion-criterion loo_lcb --replicates 50000 --confidence-level 0.95 --seed 0 \
  --output-root $B/certified-union-loo-lcb

run 7 python scripts/analyze_gate_robustness.py \
  --decisions $B/certified-union-loo-lcb/rho_0.94_decisions.jsonl --restore-cost-fraction 0.94 \
  --replicates 20000 --confidence-level 0.95 --seed 0 \
  --output $B/gate-robustness/gate_robustness_rho094.json

echo "CHAIN_A_H1_DONE $(date +%H:%M:%S)"
