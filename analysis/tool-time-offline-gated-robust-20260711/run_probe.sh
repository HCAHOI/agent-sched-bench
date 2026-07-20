#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
artifact_root="${repo_root}/analysis/tool-time-offline-gated-robust-20260711"
previous_root="${repo_root}/.omc/artifacts/tool-time-biclassifier-probe-20260710"
sab_root="${repo_root}/.omc/artifacts/tool-time-biclassifier-sab-20260711"
python="${repo_root}/.venv/bin/python"

for corpus in swe terminal sab; do
  if [[ -e "${artifact_root}/${corpus}_cv" ]]; then
    printf 'refusing to mix stale output directory: %s\n' \
      "${artifact_root}/${corpus}_cv" >&2
    exit 1
  fi
done

mkdir -p "${artifact_root}/provenance"
exec > >(tee "${artifact_root}/run.log") 2>&1
set -x
printf 'invocation:'
printf ' %q' "$0" "$@"
printf '\n'

git rev-parse HEAD > "${artifact_root}/provenance/git_head.txt"
git status --short --branch > "${artifact_root}/provenance/git_status.txt"
git diff --binary > "${artifact_root}/provenance/working_tree.diff"
git diff --cached --binary > "${artifact_root}/provenance/index.diff"
"${python}" --version > "${artifact_root}/provenance/python_version.txt" 2>&1
"${python}" -m pip freeze > "${artifact_root}/provenance/pip_freeze.txt"

snapshot_root="${artifact_root}/provenance/source_snapshot"
snapshot_files=(
  scripts/aggregate_offline_probe_cv.py
  scripts/evaluate_offline_probe_clock.py
  src/trace_collect/causal_history.py
  src/trace_collect/classification_metrics.py
  src/trace_collect/cli_helpers.py
  src/trace_collect/command_features.py
  src/trace_collect/latency_validation.py
  src/trace_collect/tool_latency_dataset.py
  src/trace_collect/tool_latency_offline_probe.py
  src/trace_collect/tool_latency_profiled.py
  src/trace_collect/tool_latency_utility_clock.py
  tests/test_tool_latency_offline_probe.py
  tests/test_tool_latency_utility_clock.py
)
for relative_path in "${snapshot_files[@]}"; do
  mkdir -p "${snapshot_root}/$(dirname "${relative_path}")"
  cp "${repo_root}/${relative_path}" "${snapshot_root}/${relative_path}"
done
printf '%s\n' "${snapshot_files[@]}" \
  > "${artifact_root}/provenance/source_snapshot_files.txt"

{
  find "${previous_root}/data/swe" "${previous_root}/data/terminal" \
    "${sab_root}/data/sab" -type f -name 'f*_*.jsonl' -print0
  printf '%s\0' \
    "${artifact_root}/protocol.md" \
    "${artifact_root}/run_probe.sh" \
    "${repo_root}/scripts/aggregate_offline_probe_cv.py" \
    "${repo_root}/scripts/evaluate_offline_probe_clock.py" \
    "${repo_root}/src/trace_collect/causal_history.py" \
    "${repo_root}/src/trace_collect/classification_metrics.py" \
    "${repo_root}/src/trace_collect/cli_helpers.py" \
    "${repo_root}/src/trace_collect/command_features.py" \
    "${repo_root}/src/trace_collect/latency_validation.py" \
    "${repo_root}/src/trace_collect/tool_latency_dataset.py" \
    "${repo_root}/src/trace_collect/tool_latency_offline_probe.py" \
    "${repo_root}/src/trace_collect/tool_latency_profiled.py" \
    "${repo_root}/src/trace_collect/tool_latency_utility_clock.py" \
    "${repo_root}/tests/test_tool_latency_offline_probe.py" \
    "${repo_root}/tests/test_tool_latency_utility_clock.py"
} | LC_ALL=C sort -zu | xargs -0 sha256sum \
  > "${artifact_root}/provenance/input_hashes.sha256"

corpora=(swe terminal sab)
data_roots=(
  "${previous_root}/data/swe"
  "${previous_root}/data/terminal"
  "${sab_root}/data/sab"
)
for corpus_index in "${!corpora[@]}"; do
  corpus=${corpora[corpus_index]}
  data_root=${data_roots[corpus_index]}
  output_root="${artifact_root}/${corpus}_cv"
  mkdir -p "${output_root}"
  for fold in 1 2 3 4 5; do
    "${python}" "${repo_root}/scripts/evaluate_offline_probe_clock.py" \
      --profile-latencies "${data_root}/f${fold}_profile.jsonl" \
      --eval-latencies "${data_root}/f${fold}_eval.jsonl" \
      --kv-costs-ms 500,1000,1500,2000,2500,3000,3500,4000,4500,5000 \
      --guard-ms 0 \
      --inner-folds 4 \
      --min-tool-history 1 \
      --min-profile-tasks 1 \
      --command-field command \
      --max-prefix-depth 4 \
      --output "${output_root}/f${fold}_summary.json" \
      --decisions-output "${output_root}/f${fold}_decisions.jsonl"
  done
  "${python}" "${repo_root}/scripts/aggregate_offline_probe_cv.py" \
    "${output_root}" \
    --expected-fold-count 5 \
    --output "${output_root}/pooled_results.json"
done

find "${artifact_root}" -type f \
  ! -name run.log \
  ! -name result_hashes.sha256 \
  ! -name '*.pyc' \
  ! -path '*/__pycache__/*' \
  -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
  > "${artifact_root}/result_hashes.sha256"
