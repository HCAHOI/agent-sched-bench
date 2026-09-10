#!/usr/bin/env bash
# One-shot environment build for a fresh 2xL40S Vast host. Idempotent per step.
# Ship the repo first (git archive HEAD | ssh ... tar x -C /workspace/agent-sched-bench),
# then: nohup bash scripts/setup/vast_two_instance_host.sh > /workspace/bootstrap.log 2>&1 &
# Checkouts, venvs and the model land under /workspace so they survive the
# home-directory regeneration Vast performs on every container start.
set -euo pipefail
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/workspace/.cache} HF_HOME=${HF_HOME:-/workspace/.hf_home}
export CONTINUUM_PYTHON=/usr/bin/python3.12 PPD_NATIVE_PUSH=1
log(){ printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
log "step 1: cuda-compat-13-0 (vLLM 0.28 cu13 wheels on driver 570)"
dpkg -s cuda-compat-13-0 >/dev/null 2>&1 || { apt-get update -qq; apt-get install -y -qq cuda-compat-13-0; }
log "step 2: model Qwen/Qwen3-4B-Instruct-2507-FP8 @ 8591804019c8"
uvx --from huggingface_hub hf download Qwen/Qwen3-4B-Instruct-2507-FP8 --revision 8591804019c8b22094c3b5b4454e0edc05dffc98
log "step 3: continuum public (vllm 0.10.2 + overlay)"
bash scripts/baselines/continuum_public.sh install
log "step 4: dualmap"
bash scripts/baselines/dualmap_official.sh install
log "step 5: ppd native push (vllm 0.28.0 + nixl 1.4.1)"
bash scripts/baselines/ppd_official.sh install
log "step 6: thunderagent base"
base=$XDG_CACHE_HOME/agent-sched-bench/ThunderAgent-7ddc8610270e56d3b109eed8796b3a4360fc67c9
[[ -x $base/.venv/bin/thunderagent ]] || bash scripts/baselines/thunderagent_official.sh install
log "step 7: thunderagent patched variants"
for v in pending_release capacity_consistent; do
  dir=/workspace/ThunderAgent-${v//_/-}-7ddc861
  patch=$PWD/scripts/baselines/thunderagent_$v.patch
  [[ -e $dir ]] || { git clone -q "$base" "$dir"; git -C "$dir" checkout -q --detach 7ddc8610270e56d3b109eed8796b3a4360fc67c9; }
  git -C "$dir" remote set-url origin https://github.com/ThunderAgent-org/ThunderAgent.git
  git -C "$dir" apply --reverse --check "$patch" 2>/dev/null || git -C "$dir" apply "$patch"
  [[ -x /workspace/venvs/thunderagent-${v//_/-}/bin/thunderagent ]] || THUNDERAGENT_CHECKOUT=$dir THUNDERAGENT_VENV=/workspace/venvs/thunderagent-${v//_/-} bash scripts/baselines/thunderagent_official.sh install
done
log "step 8: verify"
bash scripts/baselines/continuum_public.sh verify
bash scripts/baselines/dualmap_official.sh verify
bash scripts/baselines/ppd_official.sh verify-installed
bash scripts/baselines/thunderagent_official.sh verify
ls -la "$HF_HOME"/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8/snapshots/*/model.safetensors
log "BOOTSTRAP COMPLETE"
