#!/bin/bash
cd /home/chiyu/workspace/agent-sched-bench || exit 99
source .venv/bin/activate; export PYTHONPATH=src
B=analysis/fresh-corpus-certification-20260717
run(){ n="$1"; shift; echo "[$(date +%H:%M:%S)] STEP $n START"; "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[$(date +%H:%M:%S)] STEP $n FAILED (exit=$rc)"; exit $rc; fi; echo "[$(date +%H:%M:%S)] STEP $n OK"; }
run 2 python scripts/certification/run_offline_gated_robust_confirmation.py --manifest $B/offline-gated-robust/manifest.json --output-root $B/offline-gated-robust/results
run 3 python scripts/exploration/run_restore_cost_mode_b.py \
  --confirmation-root $B/offline-gated-robust/results --restore-cost-fractions 0.0,0.94 \
  --replicates 50000 --confidence-level 0.95 --seed 0 --output-root $B/restore-cost-mode-b
run 4 python scripts/exploration/run_within_task_baseline.py \
  --confirmation-root $B/offline-gated-robust/results --mode-b-root $B/restore-cost-mode-b \
  --restore-cost-fractions 0.0,0.94 --replicates 50000 --confidence-level 0.95 --seed 0 \
  --output-root $B/within-task-gated
run 5 python scripts/certification/run_hazard_model_confirmation.py \
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
