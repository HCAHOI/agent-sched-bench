#!/bin/bash
cd /home/chiyu/workspace/agent-sched-bench || exit 99
source .venv/bin/activate; export PYTHONPATH=src
B=analysis/fresh-corpus-certification-20260717
FR=traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200
run(){ n="$1"; shift; echo "[$(date +%H:%M:%S)] STEP $n START"; "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[$(date +%H:%M:%S)] STEP $n FAILED (exit=$rc)"; exit $rc; fi; echo "[$(date +%H:%M:%S)] STEP $n OK"; }
run 8 python scripts/certification/run_benchmark_frontier.py \
  --config-manifest $B/offline-gated-robust/manifest.json --trace-root $FR --tool-name-trie \
  --num-intervals 40 --model-family gbm --feature-set full --ensemble-members 0 \
  --restore-cost-fractions 0.0,0.94 --replicates 50000 --confidence-level 0.95 --seed 0 \
  --exposure-note "Fresh SWE-ReBench seed42 skip150 n=277; disjoint from all dev roots; pre-registration 20260716." \
  --output-root $B/frontier-p1
run 9 python scripts/certification/analyze_frontier_permutation.py \
  --decisions $B/frontier-p1/rho_0.94_decisions.jsonl \
  --treatment-field offline_gated_robust_trigger_ms --baseline-field offline_gated_tool_name_trigger_ms \
  --restore-cost-fraction 0.94 --replicates 20000 --seed 0 \
  --output $B/frontier-p1/permutation_p1_rho094.json
echo "CHAIN_B_H2_DONE $(date +%H:%M:%S)"
