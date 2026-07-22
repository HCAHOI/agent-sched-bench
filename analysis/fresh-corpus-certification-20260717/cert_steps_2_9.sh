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

echo "=========================================================="
echo "PIPELINE COMPLETE. VERDICTS:"
python3 - "$B" <<'PY'
import json,sys
B=sys.argv[1]
def cells(d,keyfn):
    out=[]
    for c,v in d.items():
        try: fc=float(c)
        except: continue
        p=keyfn(v)
        if p is not None: out.append((fc,p,v))
    return sorted(out)
print("\n--- H1: certified-union vs deadline (permutation, rho=0.94) ---")
try:
    g=json.load(open(f"{B}/gate-robustness/gate_robustness_rho094.json"))
    dc=g["deployed_certificate_task_cluster"]
    any_cert=False
    for c in sorted(dc,key=float):
        v=dc[c]; p=v.get("permutation_p_positive"); lab=v.get("permutation_label"); dm=v.get("paired_delta_ms")
        mark="  <== CERTIFIED" if (p is not None and p<=0.0025) else ""
        if p is not None and p<=0.0025: any_cert=True
        print(f"  kv={float(c):>6.0f}  p+={p}  label={lab}  delta_ms={dm}{mark}")
    print("  H1:", "CERTIFIES (>=1 cell p<=0.0025)" if any_cert else "does NOT certify")
except Exception as e: print("  H1 read error:",e)
print("\n--- H2: command-prefix trie vs tool-name/Continuum (permutation, rho=0.94) ---")
try:
    h=json.load(open(f"{B}/frontier-p1/permutation_p1_rho094.json"))
    any_cert=False
    for c in sorted(h,key=float):
        v=h[c]; p=v.get("permutation_p_positive"); lab=v.get("permutation_label"); dm=v.get("paired_delta_ms")
        mark="  <== CERTIFIED" if (p is not None and p<=0.0025) else ""
        if p is not None and p<=0.0025: any_cert=True
        print(f"  kv={float(c):>6.0f}  p+={p}  label={lab}  delta_ms={dm}{mark}")
    print("  H2:", "CERTIFIES (>=1 cell p<=0.0025)" if any_cert else "does NOT certify")
except Exception as e: print("  H2 read error:",e)
PY
echo "ALL DONE $(date +%H:%M:%S)"
