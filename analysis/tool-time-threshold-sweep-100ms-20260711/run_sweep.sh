#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
artifact_root="${repo_root}/analysis/tool-time-threshold-sweep-100ms-20260711"
previous_root="${repo_root}/.omc/artifacts/tool-time-biclassifier-probe-20260710"
sab_root="${repo_root}/.omc/artifacts/tool-time-biclassifier-sab-20260711"
manifest="${artifact_root}/sweep_manifest.json"
python="${repo_root}/.venv/bin/python"

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
if ! "${python}" -c 'import matplotlib' >/dev/null 2>&1; then
  printf '%s\n' \
    'threshold-sweep figures require Matplotlib.' \
    'Run: uv sync --extra dev --extra figures' >&2
  exit 1
fi

costs_csv=$("${python}" -c \
  'import json, sys; print(",".join(str(x) for x in json.load(open(sys.argv[1], encoding="utf-8"))["expected_costs_ms"]))' \
  "${manifest}")

snapshot_root="${artifact_root}/provenance/source_snapshot"
snapshot_files=(
  scripts/analyze_utility_threshold_sweep.py
  scripts/evaluate_utility_clock_policy.py
  src/trace_collect/tool_latency_headroom.py
  src/trace_collect/tool_latency_threshold_sweep.py
  src/trace_collect/tool_latency_utility_clock.py
  tests/test_tool_latency_threshold_sweep.py
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
    "${manifest}" \
    "${artifact_root}/run_sweep.sh" \
    "${repo_root}/scripts/analyze_utility_threshold_sweep.py" \
    "${repo_root}/scripts/evaluate_utility_clock_policy.py" \
    "${repo_root}/src/trace_collect/tool_latency_headroom.py" \
    "${repo_root}/src/trace_collect/tool_latency_threshold_sweep.py" \
    "${repo_root}/src/trace_collect/tool_latency_utility_clock.py" \
    "${repo_root}/tests/test_tool_latency_threshold_sweep.py"
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
    "${python}" "${repo_root}/scripts/evaluate_utility_clock_policy.py" \
      --profile-latencies "${data_root}/f${fold}_profile.jsonl" \
      --eval-latencies "${data_root}/f${fold}_eval.jsonl" \
      --kv-costs-ms "${costs_csv}" \
      --guard-ms 0 \
      --min-tool-history 1 \
      --min-profile-tasks 1 \
      --command-field command \
      --max-prefix-depth 4 \
      --output "${output_root}/f${fold}_summary.json" \
      --decisions-output "${output_root}/f${fold}_decisions.jsonl"
  done
done

"${python}" "${repo_root}/scripts/analyze_utility_threshold_sweep.py" \
  --manifest "${manifest}" \
  --output "${artifact_root}/sweep_results.json" \
  --figures-dir "${artifact_root}/figures"

find "${artifact_root}" -type f \
  ! -name run.log \
  ! -name result_hashes.sha256 \
  ! -name '*.pyc' \
  ! -path '*/__pycache__/*' \
  -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
  > "${artifact_root}/result_hashes.sha256"
