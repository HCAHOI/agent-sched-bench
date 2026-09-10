#!/usr/bin/env bash
# Two independent copies of the reviewed Continuum fork, with FCFS or Continuum.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
python=${RUNNER_PYTHON:-$repo/.venv/bin/python}
dram_metrics=${DRAM_METRICS:-on}
[[ "$dram_metrics" == on || "$dram_metrics" == off ]] || exit 2
run=${RUN_ROOT:?RUN_ROOT must be a new absolute output directory}
manifest=${MANIFEST:-$repo/analysis/development/mixed28-l40s-closed-calibration-v1/manifest.yaml}
model=${MODEL:-Qwen/Qwen3-4B-Instruct-2507-FP8}
gpu_memory_util=${GPU_MEMORY_UTILIZATION:-0.95}
timeout_s=${SHADOW_LLM_TIMEOUT_S:-1800}
concurrency=${CONCURRENCY:-16}
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || exit 2
task_sticky=${TASK_STICKY:-0}
[[ "$task_sticky" == 0 || "$task_sticky" == 1 ]] || exit 2
instance_policy=${INSTANCE_POLICY:-fcfs}
router_policy=${ROUTER_POLICY:-least-requests}
mode=${1:---run}
shadow_mode=continuum-public
ppd_mode=
case "$router_policy" in
  least-requests) ;;
  thunderagent)
    [[ "$instance_policy" == fcfs && "$task_sticky" == 0 && "$timeout_s" == 3600 ]] || exit 2
    shadow_mode=thunderagent ;;
  dualmap)
    [[ "$instance_policy" == fcfs && "$task_sticky" == 0 ]] || exit 2
    [[ "$mode" == --calibrate || -n "${DUALMAP_PREFILL_TPOT:-}" ]] || { echo "DUALMAP_PREFILL_TPOT requires hardware calibration" >&2; exit 2; }
    shadow_mode=thunderagent ;;
  pd|ppd|static-x1|profile)
    [[ "$instance_policy" == fcfs && "$task_sticky" == 0 ]] || exit 2
    [[ "$router_policy" != static-x1 || "$mode" == --smoke || "$mode" == --profile-ppd ]] || exit 2
    [[ "$router_policy" != profile || "$mode" == --profile-lengths ]] || exit 2
    [[ "$router_policy" != ppd || -d "${PPD_BENCHMARK_DATA:-}" ]] || { echo "PPD needs measured decision tables" >&2; exit 2; }
    ppd_mode=$router_policy
    shadow_mode=thunderagent ;;
  *) echo "Unsupported ROUTER_POLICY: $router_policy" >&2; exit 2 ;;
esac
case "$instance_policy" in
  fcfs) serve_command=serve-fcfs ;;
  continuum) serve_command=serve ;;
  *) echo "INSTANCE_POLICY must be fcfs or continuum" >&2; exit 2 ;;
esac
[[ "$mode" == --run || "$mode" == --smoke || "$mode" == --calibrate || "$mode" == --profile-ppd || "$mode" == --profile-lengths || "$mode" == --external-replay ]] || exit 2
[[ "$mode" != --calibrate || "$router_policy" == dualmap ]] || exit 2
[[ "$mode" != --profile-ppd || "$ppd_mode" == pd || "$ppd_mode" == static-x1 ]] || exit 2
[[ "$mode" != --profile-lengths || "$ppd_mode" == profile ]] || exit 2
[[ "$run" == /* && ! -e "$run" && -f "$manifest" ]] || exit 2
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) == 2 ]] || exit 2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | \
  awk -F, '$1 !~ /L40S/ || $2 < 45000 {bad=1} END {exit bad}'
ports=(8000 8001 9000 5557 5558 5559 5560)
[[ "$router_policy" != dualmap ]] || ports+=(8101 8111)
[[ -z "$ppd_mode" ]] || ports+=(14579 14580)
for port in "${ports[@]}"; do
  if timeout 1 bash -c "</dev/tcp/127.0.0.1/$port" 2>/dev/null; then
    echo "port $port is busy" >&2; exit 1
  fi
done
if [[ "$mode" == --run ]]; then docker info >/dev/null; fi
if [[ -n "$ppd_mode" ]]; then
  bash scripts/baselines/ppd_official.sh verify-installed
  ppd_checkout=${PPD_CHECKOUT:-${XDG_CACHE_HOME:-$HOME/.cache}/agent-sched-bench/PPD-28aaa63c6d7a0a0e00d97f8c958291bb6b6a4367}
  ppd_python=${PPD_VENV:-$ppd_checkout/.venv}/bin/python
else
  bash scripts/baselines/continuum_public.sh verify
fi
mkdir -p "$run"
cp "$manifest" "$run/manifest.yaml"
printf '{"dram_metrics":"%s","replay_mode":"%s"}\n' "$dram_metrics" "$mode" > "$run/collection-config.json"
if [[ -n "${SOURCE_BASE_REV:-}" ]]; then
  printf '%s\n' "$SOURCE_BASE_REV" > "$run/source-base.txt"
  tar -czf "$run/deployed-source.tar.gz" src scripts configs pyproject.toml uv.lock
else
  git rev-parse HEAD > "$run/source-base.txt"
  git diff > "$run/source.patch"
fi
tar -czf "$run/launch-source.tar.gz" scripts/baselines/least_requests_proxy.py \
  scripts/evaluation/run_two_instance_fcfs.sh tests/test_least_requests_proxy.py scripts/evaluation/collect_vllm_kv_events.py \
  scripts/evaluation/collect_http_metrics.py
cp scripts/baselines/thunderagent_official{.sh,_launcher.py} "$run/"
cp scripts/evaluation/check_two_instance_request_metrics.py "$run/"
checker=("$python" scripts/evaluation/check_two_instance_request_metrics.py "$run")
if [[ -n "$ppd_mode" ]]; then
  cp scripts/baselines/ppd_official{.sh,_proxy.py} scripts/baselines/ppd_request_metrics.patch scripts/baselines/ppd_nixl_metrics.patch "$run/"
  cp scripts/evaluation/check_ppd_request_metrics.py "$run/"
  cp scripts/evaluation/profile_ppd.py "$run/"
  if [[ "$mode" == --profile-lengths ]]; then
    cp scripts/evaluation/profile_serving_lengths.py "$run/"
    cp "${LENGTH_PROFILE_PLAN:?Length profiling requires its declared plan}" "$run/length-profile-plan.md"
  fi
  cp scripts/baselines/ppd_policy.py "$run/"
  [[ "${PPD_STATE_AWARE:-0}" != 1 ]] || cp scripts/baselines/ppd_state_query.patch "$run/"
  [[ "${PPD_NATIVE_PUSH:-0}" != 1 ]] || cp scripts/baselines/ppd_push_metrics.patch "$run/"
  num_layers=$("$ppd_python" -c 'import sys; from transformers import AutoConfig; print(AutoConfig.from_pretrained(sys.argv[1], local_files_only=True).num_hidden_layers)' "$model")
  checker=(env PYTHONPATH="$repo" "$python" scripts/evaluation/check_ppd_request_metrics.py "$run" --num-layers "$num_layers")
  uv pip freeze --python "$ppd_python" > "$run/engine-packages.txt"
  if [[ "$ppd_mode" == ppd ]]; then
    cp -a "$PPD_BENCHMARK_DATA" "$run/ppd-calibration"
  fi
fi
if [[ "$router_policy" == dualmap ]]; then
  cp scripts/baselines/dualmap_official{.sh,_proxy.py} "$run/"
  cpu_cache_gib=${DUALMAP_CPU_CACHE_GIB:-48}
  kv_bytes_per_token=$("$python" - "$model" <<'PY'
import sys
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True).to_dict()
assert cfg.get('dtype', cfg.get('torch_dtype')) == 'bfloat16', cfg
print(2 * 2 * cfg['num_hidden_layers'] * cfg['num_key_value_heads'] * cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads']))
PY
  )
fi
nvidia-smi -q > "$run/hardware.txt"
nvidia-smi topo -m > "$run/topology.txt"
lscpu > "$run/cpu.txt"
"$python" -m pip --version >/dev/null 2>&1 || true
uv pip freeze --python "$python" > "$run/packages.txt"
overlay=${CUPTI_DRAM_OVERLAY:-$HOME/.cache/agent-sched-bench/cupti-dram-13.3.1}
lib=$overlay/nvidia/cu13/lib
cap=$(setpriv --list-caps | awk '$0=="perfmon" || $0=="cap_38" {print;exit}')
if [[ "$dram_metrics" == on ]]; then [[ -n "$cap" && -f "$lib/libcupti.so.13" ]]; fi
export OPENCLAW_REPLAY_PAIRED_WORKLOAD_CONTRACT=2 OPENCLAW_REPLAY_TRACE_TOOLS=1
export OPENCLAW_REPLAY_TRACE_TOOL_SPEED=4
servers=() collectors=() monitors=() sim_pid= proxy_pid=
stop_group() {
  local pid=$1
  kill -TERM -- -"$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 -- -"$pid" 2>/dev/null || break
    sleep 0.2
  done
  kill -KILL -- -"$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  [[ -z "$sim_pid" ]] || stop_group "$sim_pid"
  [[ -z "$proxy_pid" ]] || stop_group "$proxy_pid"
  for pid in "${collectors[@]}"; do kill -TERM "$pid" 2>/dev/null; wait "$pid"; done
  for pid in "${monitors[@]}"; do stop_group "$pid"; done
  for pid in "${servers[@]}"; do stop_group "$pid"; done
  date -u +%FT%TZ > "$run/end-utc.txt"
  echo "$rc" > "$run/exit-code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'timestamp_utc,gpu_index,gpu_uuid,utilization_pct,memory_activity_pct,memory_mib,memory_total_mib,power_w,power_limit_w,sm_clock_mhz,memory_clock_mhz\n' > "$run/gpu-paired.csv"
setsid env TZ=UTC nvidia-smi \
  --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem \
  --format=csv,noheader,nounits -lms 200 >> "$run/gpu-paired.csv" 2> "$run/gpu-paired.err" &
monitors+=("$!")
wait_http() {
  local port=$1 pid=$2 endpoint=${3:-health}
  for _ in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:$port/$endpoint" >/dev/null 2>&1; then return; fi
    kill -0 "$pid" || return 1
    sleep 0.5
  done
  return 1
}
for i in 0 1; do
  cell="$run/instance-$i"
  mkdir "$cell"
  port=$((8000+i)); kv_port=$((5557+2*i)); replay_port=$((5558+2*i))
  cpuset=${PREFILL_CPUSET:-0-2}; [[ "$i" == 0 ]] || cpuset=${DECODE_CPUSET:-12-14}
  cache_env=() cache_args=()
  cell_gpu_memory_util=$gpu_memory_util
  engine_command=(bash scripts/baselines/continuum_public.sh "$serve_command" "$model")
  if [[ "$router_policy" == dualmap ]]; then
    cat > "$cell/lmcache.yaml" <<YAML
chunk_size: 256
local_cpu: true
max_local_cpu_size: $cpu_cache_gib
save_decode_cache: true
extra_config:
  force_store_wait: true
internal_api_server_enabled: true
internal_api_server_host: 127.0.0.1
internal_api_server_port_start: $((8100+10*i))
internal_api_server_include_index_list: [1]
YAML
    cache_env=(LMCACHE_CONFIG_FILE="$cell/lmcache.yaml" LMCACHE_USE_EXPERIMENTAL=True PYTHONHASHSEED=42)
    cache_args=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
  fi
  if [[ -n "$ppd_mode" ]]; then
    engine_command=("$ppd_python" -m vllm.entrypoints.cli.main serve "$model")
    [[ "$i" != 0 ]] || cell_gpu_memory_util=${PREFILL_GPU_MEMORY_UTILIZATION:-$gpu_memory_util}
    # Hold P blocks through the entire client timeout; D pulls only after allocation.
    cache_env=(VLLM_HOST_IP=127.0.0.1 PYTHONHASHSEED=42
      UCX_TLS=${PD_UCX_TLS:-tcp,cuda_copy,self} UCX_NET_DEVICES=${PD_UCX_NET_DEVICES:-lo} UCX_LOG_LEVEL=info
      VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=$((14579+i))
      VLLM_NIXL_ABORT_REQUEST_TIMEOUT=$((timeout_s+60)))
    kv_buffer_device=${PPD_KV_BUFFER_DEVICE:-cuda}
    [[ "$kv_buffer_device" == cuda || "$kv_buffer_device" == cpu ]] || exit 2
    connector_name=NixlConnector
    if [[ "${PPD_NATIVE_PUSH:-0}" == 1 ]]; then
      connector_name=NixlPushConnector
      # Router UUIDs already ensure uniqueness; keep IDs joinable across P and D.
      cache_env+=(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1)
    fi
    cache_args=(--kv-transfer-config "{\"kv_connector\":\"$connector_name\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"$kv_buffer_device\"}")
  fi
  privilege=() telemetry_env=() telemetry_args=()
  if [[ "$dram_metrics" == on ]]; then
    privilege=(sudo -n setpriv --reuid="$(id -un)" --regid="$(id -gn)" --init-groups
      --inh-caps="+$cap" --ambient-caps="+$cap")
    telemetry_env=(CUPTI_DRAM_CSV="$cell/dram-bandwidth.csv" CUPTI_DRAM_READY="$cell/dram-bandwidth-ready"
      CUPTI_DRAM_ERROR="$cell/dram-bandwidth.err" LD_LIBRARY_PATH="$lib" LD_PRELOAD="$lib/libcupti.so.13")
    telemetry_args=(--worker-cls scripts.evaluation.cupti_dram_worker.CuptiDramWorker)
  fi
  launch=(setsid "${privilege[@]}" env HOME="$HOME" PATH="$PATH"
    VLLM_NO_USAGE_STATS=1 CUDA_VISIBLE_DEVICES="$i" VLLM_SERVER_DEV_MODE=1
    PYTHONPATH="$overlay/cuda-bindings:$overlay:$repo" RUN_OUTPUT_DIR="$cell/continuum"
    VLLM_REQUEST_TELEMETRY_PATH="$cell/vllm-request-telemetry.jsonl" "${telemetry_env[@]}"
    "${cache_env[@]}"
    taskset -c "$cpuset" "${engine_command[@]}"
    --host 127.0.0.1 --port "$port" --tensor-parallel-size 1
    --gpu-memory-utilization "$cell_gpu_memory_util" --max-model-len 131072 --max-num-seqs 8
    --enable-prefix-caching --kv-cache-dtype auto --enforce-eager
    --enable-chunked-prefill --max-num-batched-tokens 2048
    "${telemetry_args[@]}"
    --enable-prompt-tokens-details
    "${cache_args[@]}"
    --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$kv_port\",\"replay_endpoint\":\"tcp://127.0.0.1:$replay_port\",\"buffer_steps\":1000000,\"hwm\":1000000,\"max_queue_size\":1000000}")
  printf '%q ' "${launch[@]}" > "$cell/vllm.argv"
  "${launch[@]}" > "$cell/vllm.log" 2>&1 &
  servers+=("$!")
done
for i in 0 1; do
  cell="$run/instance-$i"; port=$((8000+i))
  wait_http "$port" "${servers[i]}"
  if [[ "$dram_metrics" == on ]]; then [[ -f "$cell/dram-bandwidth-ready" && ! -s "$cell/dram-bandwidth.err" ]]; fi
  curl -fsS "http://127.0.0.1:$port/metrics" > "$cell/vllm-metrics-start.prom"
  cache_metric=gpu_cache_usage_perc; [[ -z "$ppd_mode" ]] || cache_metric=kv_cache_usage_perc
  for metric in num_requests_running num_requests_waiting "$cache_metric" prefix_cache_hits_total; do
    grep -q "vllm:$metric" "$cell/vllm-metrics-start.prom"
  done
  "$python" scripts/evaluation/collect_vllm_kv_events.py \
    --endpoint "tcp://127.0.0.1:$((5557+2*i))" --replay-endpoint "tcp://127.0.0.1:$((5558+2*i))" \
    --ready-file "$cell/kv-events-ready" --events-jsonl "$cell/kv-events.jsonl" \
    --summary-json "$cell/kv-events-summary.json" > "$cell/kv-collector.log" 2>&1 &
  collectors+=("$!")
  setsid bash -c '
    printf "timestamp_s,power_w,memory_mib,utilization_pct,memory_activity_pct\n"
    while true; do
      printf "%s," "$(date +%s.%N)"
      nvidia-smi -i "$1" --query-gpu=power.draw,memory.used,utilization.gpu,utilization.memory --format=csv,noheader,nounits || exit
      sleep 1
    done' _ "$i" > "$cell/gpu.csv" 2> "$cell/gpu.err" &
  monitors+=("$!")
  setsid "$python" scripts/evaluation/collect_http_metrics.py "http://127.0.0.1:$port/metrics" \
    --output "$cell/vllm-metrics-series.prom" --gaps "$cell/vllm-metrics-gaps.jsonl" \
    2> "$cell/vllm-metrics-series.err" &
  monitors+=("$!")
  if [[ "$router_policy" == dualmap ]]; then
    lmcache_port=$((8101+10*i))
    wait_http "$lmcache_port" "${servers[i]}" metrics
    curl -fsS "http://127.0.0.1:$lmcache_port/metrics" > "$cell/lmcache-metrics-start.prom"
    grep -q 'lmcache:num_stored_tokens' "$cell/lmcache-metrics-start.prom"
    setsid "$python" scripts/evaluation/collect_http_metrics.py "http://127.0.0.1:$lmcache_port/metrics" \
      --output "$cell/lmcache-metrics-series.prom" --gaps "$cell/lmcache-metrics-gaps.jsonl" \
      2> "$cell/lmcache-metrics-series.err" &
    monitors+=("$!")
  fi
done
for i in 0 1; do
  for _ in $(seq 1 100); do
    [[ -f "$run/instance-$i/kv-events-ready" ]] && break
    kill -0 "${collectors[i]}"
    sleep 0.1
  done
  [[ -f "$run/instance-$i/kv-events-ready" ]]
done
if [[ "$mode" == --calibrate ]]; then
  "$python" - "$run" "$model" <<'PY'
import json, pathlib, statistics, sys, time, uuid
import httpx
run, model = pathlib.Path(sys.argv[1]), sys.argv[2]
measurements = []
for i in range(2):
    for j, length in enumerate((256, 2048, 8192, 32768)):
        rid = uuid.uuid4().hex
        payload = dict(model=model, prompt=[100+j] + [1000] * (length-1), max_tokens=1,
                       ignore_eos=True, job_id=rid, this_func_call='', is_last_step=False)
        response = httpx.post(f'http://127.0.0.1:{8000+i}/v1/completions', json=payload,
                              headers={'x-request-id':rid}, timeout=300)
        response.raise_for_status()
        request_id = response.json()['id'] + '-0'  # Single completion prompt's engine request ID.
        path = run / f'instance-{i}/vllm-request-telemetry.jsonl'
        found = []
        for _ in range(100):
            if path.exists():
                found = [json.loads(s) for s in path.read_text().splitlines() if json.loads(s)['request_id'] == request_id]
            if found:
                break
            time.sleep(.1)
        assert len(found) == 1 and found[0]['prompt_tokens'] == length, found
        if j:  # First call warms kernels; all measured prompts have distinct first tokens.
            measurements.append(dict(instance=i, **found[0]))
        released = httpx.post(f'http://127.0.0.1:{8000+i}/continuum/programs/release', json={'job_id':rid})
        released.raise_for_status()
calibration = dict(measurements=measurements,
                   prefill_tpot=statistics.median(m['prefill_s']/m['prompt_tokens'] for m in measurements))
(run/'prefill-calibration.json').write_text(json.dumps(calibration, indent=2))
print(json.dumps(calibration))
PY
  exit 0
fi
proxy=(setsid taskset -c "${ROUTER_CPUSET:-15}" "$python" scripts/baselines/least_requests_proxy.py
  --backends http://127.0.0.1:8000 http://127.0.0.1:8001 --events "$run/routing.jsonl")
[[ "$task_sticky" == 0 ]] || proxy+=(--task-sticky)
if [[ "$router_policy" == thunderagent ]]; then
  proxy=(setsid taskset -c "${ROUTER_CPUSET:-15}" env THUNDERAGENT_BACKENDS=http://127.0.0.1:8000,http://127.0.0.1:8001
    THUNDERAGENT_CONTINUUM_FCFS=1
    THUNDERAGENT_PROFILE_DIR="$run/thunderagent-profiles" THUNDERAGENT_ROUTING_EVENTS="$run/routing.jsonl"
    SHADOW_LLM_TIMEOUT_S="$timeout_s" bash scripts/baselines/thunderagent_official.sh serve)
fi
if [[ "$router_policy" == dualmap ]]; then
  proxy=(setsid taskset -c "${ROUTER_CPUSET:-15}" bash scripts/baselines/dualmap_official.sh serve
    --model "$model" --backends http://127.0.0.1:8000 http://127.0.0.1:8001
    --output "$run" --prefill-tpot "$DUALMAP_PREFILL_TPOT" --kv-bytes-per-token "$kv_bytes_per_token"
    --cpu-cache-gib "$cpu_cache_gib" --timeout-s "$timeout_s")
  [[ "${DUALMAP_AGENT_PROGRESS:-0}" != 1 ]] || proxy+=(--agent-progress)
fi
if [[ -n "$ppd_mode" ]]; then
  proxy=(setsid taskset -c "${ROUTER_CPUSET:-15}" env PYTHONPATH="$ppd_checkout:$repo" "$ppd_python" -m scripts.baselines.ppd_official_proxy
    --mode "$ppd_mode" --model "$model" --backends http://127.0.0.1:8000 http://127.0.0.1:8001
    --transport nixl --output "$run" --timeout-s "$timeout_s")
  [[ "$ppd_mode" != ppd ]] || proxy+=(--benchmark-data "$run/ppd-calibration")
  [[ "${PPD_EXTENDED_CONTEXT:-0}" != 1 ]] || proxy+=(--extended-context)
  [[ "${PPD_STATE_AWARE:-0}" != 1 ]] || proxy+=(--state-aware)
fi
printf '%q ' "${proxy[@]}" > "$run/proxy.argv"
"${proxy[@]}" > "$run/proxy.log" 2>&1 &
proxy_pid=$!
wait_http 9000 "$proxy_pid"
date -u +%FT%TZ > "$run/start-utc.txt"
if [[ "$mode" == --smoke ]]; then
  "$python" - "$model" "$run" "$task_sticky" "$router_policy" "$dram_metrics" <<'PY'
import json, pathlib, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import httpx
model, run = sys.argv[1], pathlib.Path(sys.argv[2])
def cpu_hit_tokens(instance):
    response = httpx.get(f"http://127.0.0.1:{8101+10*instance}/metrics")
    response.raise_for_status()
    hits = re.findall(r'^lmcache:num_hit_tokens_total\{[^}]*\}\s+([\d.eE+]+)', response.text, re.M)
    assert hits, "Missing CPU KV hit counter"
    return sum(map(float, hits))
thunderagent = sys.argv[4] in {"thunderagent", "dualmap", "pd", "ppd", "static-x1"}
pd_smoke = sys.argv[4] in {"pd", "ppd", "static-x1"}
requests = [("routing-smoke", 0), ("routing-smoke", 1)]
if sys.argv[3] == "1":
    requests = [("routing-smoke-0", 0), ("routing-smoke-1", 1)] * 2
if thunderagent:
    requests = [(f"routing-smoke-{i}", None) for i in range(4)]
def send_request(item):
    i, (job_id, expected_instance) = item
    payload = dict(model=model, messages=[dict(role="user", content="Say hello.")],
                   job_id=job_id, max_tokens=8, ignore_eos=True, return_token_ids=True,
                   stream=True, stream_options={"include_usage": True}, seed=0, temperature=0,
                   is_last_step=False, this_func_call="")
    if thunderagent:
        payload["program_id"] = payload.pop("job_id")
        del payload["is_last_step"], payload["this_func_call"]
    if sys.argv[4] == "dualmap":
        payload["messages"][0]["content"] = job_id + ": " + "cache memory " * 600
    if pd_smoke:
        payload["messages"][0]["content"] = job_id + ": " + " cache" * 8000
        payload["max_tokens"] = 128
    long_pd = pd_smoke and i == 0
    if long_pd:
        # Exercise direct KV transfer beyond mixed56's observed 109,441-token maximum.
        payload["messages"][0]["content"] = " cache" * 120000
    with httpx.stream("POST", "http://127.0.0.1:9000/v1/chat/completions", json=payload, timeout=300 if pd_smoke else 120) as response:
        response.raise_for_status()
        if not thunderagent:
            assert response.headers["x-serving-instance"] == str(expected_instance)
        chunks = [json.loads(line[5:]) for line in response.iter_lines() if line.startswith("data:") and line[5:].strip() != "[DONE]"]
    (run / f"smoke-response-{i}.json").write_text(json.dumps(chunks))
    assert any(c.get("usage", {}).get("completion_tokens") == payload["max_tokens"] for c in chunks if c.get("usage"))
    if long_pd:
        assert any(110000 < c.get("usage", {}).get("prompt_tokens", 0) < 131064 for c in chunks if c.get("usage"))
    assert chunks[0]["id"].startswith("chatcmpl-")
    time.sleep(0.2)
if thunderagent:
    requests = [(f"routing-smoke-{i}", None) for i in range(32)]
    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(send_request, enumerate(requests)))
    if sys.argv[4] == "dualmap":
        time.sleep(11)  # LMCache publishes counters every 10 seconds; include all first-wave hits.
        hits_before_reset = [cpu_hit_tokens(i) for i in range(2)]
        for port in (8000, 8001):
            response = httpx.post(f"http://127.0.0.1:{port}/reset_prefix_cache")
            response.raise_for_status()
    # Returning requests exercise the same per-job history after a completed call.
    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(send_request, enumerate(requests)))
else:
    for item in enumerate(requests):
        send_request(item)
for job_id in dict.fromkeys(job for job, _ in requests):
    release_path = "/programs/release" if thunderagent else "/continuum/programs/release"
    release_key = "program_id" if thunderagent else "job_id"
    response = httpx.post("http://127.0.0.1:9000" + release_path, json={release_key:job_id})
    response.raise_for_status()
    assert response.json()["released"]
health = httpx.get("http://127.0.0.1:9000/health").json()
if sys.argv[4] in {"dualmap", "pd", "ppd", "static-x1"}:
    assert health["outstanding"] == 0
else:
    assert health["programs_count"] == 0 if thunderagent else health["outstanding"] == [0, 0]
if thunderagent:
    rows = [json.loads(line) for line in (run / "routing.jsonl").read_text().splitlines()]
    assert len([r for r in rows if r["event"] == "dispatch"]) == 2 * len(requests)
    assert all(r["outcome"] == "complete" for r in rows if r["event"] == "finish")
time.sleep(35)  # CUPTI decodes its one-second samples in 30-second batches.
for i in range(2):
    cell = run / f"instance-{i}"
    if sys.argv[5] == "on":
        assert len((cell / "dram-bandwidth.csv").read_text().splitlines()) > 1
    assert (cell / "vllm-request-telemetry.jsonl").stat().st_size > 0
    # Check before shutdown: replay-on-exit must not hide a broken live subscription.
    events = [json.loads(line) for line in (cell / "kv-events.jsonl").read_text().splitlines()]
    assert events, "KV live subscription produced no events"
    assert [e["seq"] for e in events] == list(range(len(events))), "KV live sequence gap"
    if sys.argv[4] == "dualmap":
        hits_after_reset = cpu_hit_tokens(i)
        assert hits_after_reset > hits_before_reset[i], "No new CPU KV retrieval after GPU cache reset"
        (cell / "smoke-cpu-kv-retrieval.json").write_text(json.dumps(dict(
            hit_tokens_before_reset=hits_before_reset[i], hit_tokens_after_second_wave=hits_after_reset,
            hit_tokens_delta=hits_after_reset - hits_before_reset[i])))
print("Both GPU streams, routing, job release and request telemetry passed; DRAM collection:", sys.argv[5])
PY
else
  simulate=(setsid env PYTHONPATH="$repo/src:$repo" "$python" -m trace_collect.cli simulate
    --manifest "$manifest" --output-dir "$run/output" --container docker --network-mode host
    --concurrency "$concurrency" --workers 1 --prep-concurrency 8 --replay-speed 1
    --shadow-llm-api-base http://127.0.0.1:9000/v1 --shadow-llm-model "$model"
    --shadow-llm-timeout-s "$timeout_s" --shadow-llm-seed 0 --shadow-llm-mode "$shadow_mode"
    --resource-monitoring off --pmu-monitoring off --memory-bandwidth-monitoring off
    --replacement-delay-mean-s 10 --replacement-seed 42 --container-cpuset-cpus 3-11 --container-cpus 2)
  if [[ "$mode" == --profile-ppd ]]; then
    simulate=(setsid taskset -c "${CALIBRATION_CPUSET:-${ROUTER_CPUSET:-15}}" env PYTHONPATH="$ppd_checkout:$repo" "$ppd_python"
      scripts/evaluation/profile_ppd.py --mode "$ppd_mode" --model "$model"
      --output "$run/ppd-calibration" --seed 42 --start-point "${PPD_PROFILE_START_POINT:-1}")
    [[ -z "${PPD_PROFILE_CONTEXT_TOKENS:-}" ]] || simulate+=(--context-tokens "$PPD_PROFILE_CONTEXT_TOKENS")
  fi
  if [[ "$mode" == --profile-lengths ]]; then
    simulate=(setsid taskset -c "${CALIBRATION_CPUSET:-${ROUTER_CPUSET:-15}}" env PYTHONPATH="$ppd_checkout:$repo" "$ppd_python"
      scripts/evaluation/profile_serving_lengths.py --model "$model" --plan "$run/length-profile-plan.md"
      --output "$run/length-profile" --stage "${LENGTH_PROFILE_STAGE:-preliminary}" --timeout-s "$timeout_s")
    [[ -z "${LENGTH_PROFILE_RESUME_FROM:-}" ]] || simulate+=(--resume-from "$LENGTH_PROFILE_RESUME_FROM")
  fi
  if [[ "$mode" == --external-replay ]]; then
    # The external replay controller writes its actual exit status atomically.
    simulate=(setsid bash -c 'while [[ ! -f "$1/external-replay-exit-code" ]]; do sleep 2; done
      read -r rc < "$1/external-replay-exit-code"
      [[ "$rc" =~ ^[0-9]+$ ]] && ((rc <= 255)) || exit 2
      exit "$rc"' _ "$run")
    touch "$run/external-replay-ready"
  fi
  printf '%q ' "${simulate[@]}" > "$run/simulate.argv"
  "${simulate[@]}" > "$run/simulate.log" 2>&1 &
  sim_pid=$!
  echo "Workload started; process PID=$sim_pid; results=$run"
  monitor_start=$(date +%s)
  while kill -0 "$sim_pid" 2>/dev/null; do
    for pid in "${servers[@]}" "${collectors[@]}" "${monitors[@]}" "$proxy_pid"; do kill -0 "$pid"; done
    for i in 0 1; do
      cell="$run/instance-$i"
      for error in dram-bandwidth.err gpu.err vllm-metrics-series.err; do [[ ! -s "$cell/$error" ]]; done
      metric_files=(gpu.csv vllm-metrics-series.prom)
      [[ "$dram_metrics" != on ]] || metric_files+=(dram-bandwidth.csv)
      for metric_file in "${metric_files[@]}"; do
        # CUPTI writes 30-second batches; 90 seconds catches a stopped writer.
        [[ $(( $(date +%s) - monitor_start )) -ge 90 ]] || continue
        [[ -s "$cell/$metric_file" && $(( $(date +%s) - $(stat -c %Y "$cell/$metric_file") )) -lt 90 ]]
      done
      if [[ "$router_policy" == dualmap ]]; then
        [[ ! -s "$cell/lmcache-metrics-series.err" ]]
        [[ -s "$cell/lmcache-metrics-series.prom" && $(( $(date +%s) - $(stat -c %Y "$cell/lmcache-metrics-series.prom") )) -lt 90 ]]
      fi
    done
    if [[ "$router_policy" != least-requests ]]; then
      curl --max-time 30 --retry 1 --retry-delay 1 -fsS http://127.0.0.1:9000/health >/dev/null
      "${checker[@]}"
    fi
    sleep 5
  done
  rc=0; wait "$sim_pid" || rc=$?; sim_pid=
  echo "$rc" > "$run/simulate-exit-code"
  [[ "$rc" == 0 ]]
fi
if [[ "$router_policy" != least-requests ]]; then
  stop_group "$proxy_pid"; proxy_pid=
  sleep 2 # Let aborted backend requests write terminal telemetry before final reconciliation.
  if [[ -n "$ppd_mode" ]]; then
    # Native NIXL consumes release notifications on engine steps. Wake idle P
    # after the measured workload and cancellation have ended, before auditing.
    "$python" - "$run" "$model" <<'PYPROBE'
import json, pathlib, sys, time
import httpx
run, model = pathlib.Path(sys.argv[1]), sys.argv[2]
started = time.time()
response = httpx.post("http://127.0.0.1:8000/v1/chat/completions",
    json=dict(model=model, messages=[dict(role="user", content="Release audit probe")],
              max_tokens=1, ignore_eos=True, stream=False),
    headers={"x-request-id": "nixl-release-audit-probe"}, timeout=30)
response.raise_for_status()
assert response.json()["usage"]["completion_tokens"] == 1
(run / "maintenance-probe.json").write_text(json.dumps(dict(
    purpose="Process pending producer release notifications after measurement",
    measurement=False, started_unix_s=started, finished_unix_s=time.time(),
    response=response.json()), indent=2))
PYPROBE
  fi
  "${checker[@]}" --final
fi
for i in 0 1; do
  if [[ "$router_policy" == dualmap ]]; then
    curl -fsS "http://127.0.0.1:$((8101+10*i))/metrics" > "$run/instance-$i/lmcache-metrics-final.prom"
  fi
  curl -fsS "http://127.0.0.1:$((8000+i))/metrics" > "$run/instance-$i/vllm-metrics-final.prom"
  kill -TERM "${collectors[i]}"
  wait "${collectors[i]}"
done
collectors=()

"$python" - "$run" <<'PY'
import json, pathlib, sys
for i in range(2):
    summary = json.loads((pathlib.Path(sys.argv[1]) / f"instance-{i}/kv-events-summary.json").read_text())
    assert summary["batch_count"] > 0 and not summary["sequence_gaps"], summary
    assert summary["tail_replay_complete"], summary
PY
