# Experiment Plan

**Updated**: 2026-04-01  
**Hardware**: 1x A100-PCIE-40GB, CUDA 12.8, 503 GiB RAM  
**Model**: Llama-3.1-8B-Instruct (fp16, 16.1 GB)  
**Engine**: vLLM 0.10.2  

---

## Current State

| Item | Status |
|:---|:---|
| ENV-1 Server base | DONE (validated) |
| ENV-2 Model download | DONE (4/4 safetensors verified) |
| ENV-3a vLLM install | DONE (0.10.2 in .venv-server) |
| ENV-3b Continuum | Code ready, runtime not deployed |
| ENV-3c ThunderAgent | Code ready, runtime not deployed |
| ENV-4 Git sync | DONE |
| ENV-5 Preemption config | Code ready, needs runtime validation |
| AGENT-1~4 | Code complete, not live-tested |
| HARNESS-1~3 | Code complete, not live-tested |
| ANALYSIS-1, REPLAY-1 | Code complete |

---

## Phase 1: Foundation Validation (Day 1)

**Goal**: vLLM baseline + single agent end-to-end, trace 落盘。

### Step 1.1: Start vLLM & Verify Serving

```bash
MODEL_PATH=/workspace/agent-sched-bench/models/Llama-3.1-8B-Instruct \
  scripts/serve_vllm.sh
```

**Acceptance**:
- [ ] `curl http://localhost:8000/v1/models` returns model list
- [ ] `/metrics` endpoint returns Prometheus metrics
- [ ] Chat completion returns non-empty response
- [ ] `results/processed/vllm_server_report.json` written

### Step 1.2: Single Agent Smoke Tests

每个 agent 跑 1 个 task，确认 trace 产出正确。

```bash
# Code agent (SWE-bench)
make smoke-code

# Data agent (NL2SQL)
make smoke-data

# Research agent (web search)
make smoke-research
```

**Acceptance**:
- [ ] 每个 agent 运行 >= 3 步
- [ ] StepRecord 包含 prompt_tokens, completion_tokens, llm_latency_ms
- [ ] Tool call 正确执行（bash / sql_execute / web_search）
- [ ] Trace 可被 `pandas.read_json(lines=True)` 读取

### Step 1.3: N=2 Concurrent Smoke

```bash
.venv/bin/python -m harness.runner \
    --agent code \
    --api-base http://localhost:8000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 2 \
    --tasks 4 \
    --output results/raw/
```

**Acceptance**:
- [ ] 4 个 task 全部完成（不论 pass/fail）
- [ ] JSONL trace 文件正确写入
- [ ] 无 deadlock 或 hang

### Step 1.4: Metrics Collection Validation

在 Step 1.3 运行期间同步验证 metrics 采集：

- [ ] `vllm:gpu_cache_usage_perc` 有变化
- [ ] `vllm:num_requests_running` 反映并发数
- [ ] `nvidia-smi` GPU utilization 采样正常
- [ ] 每秒 snapshot 写入 metrics JSON

### Step 1.5: First Benchmark Run (vLLM Baseline)

```bash
# code_agent x N=[1,2,4]
for N in 1 2 4; do
  .venv/bin/python -m harness.runner \
      --agent code \
      --api-base http://localhost:8000/v1 \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --concurrency $N \
      --tasks 20 \
      --output results/raw/
done
```

**Acceptance**:
- [ ] 3 组结果全部落盘
- [ ] 能观察到 throughput (steps/min) 随 N 增加的变化趋势
- [ ] N=4 时 KV cache 使用率明显升高

**Phase 1 Milestone**: 一条完整的 vLLM + code_agent 从启动到 trace 产出的管线跑通。

---

## Phase 2: Full Baseline Sweep (Day 2-3)

**Goal**: 三种 workload 在 vLLM baseline 上完整 sweep，找到 cliff point。

### Step 2.1: 三种 Workload x vLLM Baseline

| Workload | Tasks | N 值 | 预期特征 |
|:---|:---|:---|:---|
| code_agent (SWE-bench) | 20-30 | 1,2,4,6,8 | 长 context, 多 tool call, ~70K tokens/program |
| data_agent (NL2SQL/BIRD) | 50-100 | 1,2,4,6,8 | 短 context, retry pattern, 变化大的 tool latency |
| research_agent (web search) | 20-30 | 1,2,4,6,8 | 最快 context 增长, 外部 API 延迟不可预测 |

```bash
# 用 sweep 跑完整矩阵（仅 vllm-baseline）
.venv/bin/python -m harness.sweep \
    --systems vllm-baseline \
    --workloads code_agent,data_agent,research_agent \
    --concurrency 1,2,4,6,8 \
    --output-root results/raw/
```

**关键观测**:
- [ ] 找到每种 workload 的 cliff point（throughput 开始下降的 N 值）
- [ ] 记录每个 N 值下的 `vllm:num_preemptions_total`
- [ ] 记录 KV cache 使用率时序曲线

### Step 2.2: ENV-5 Preemption Observation

在 Step 2.1 运行期间分析 preemption 行为：

- [ ] 在某个 N 值下观察到 `num_preemptions_total > 0`
- [ ] 如果所有 N <= 8 都无 preemption → 在 Step 2.4 推高到 N=12,16
- [ ] 保存 `results/processed/vllm_preemption_report.json`
- [ ] 分析 preemption 发生时的 KV cache 状态

### Step 2.3: Push to High Concurrency

```bash
# 推高 N 到 cliff point 和 thrashing 区间
.venv/bin/python -m harness.sweep \
    --systems vllm-baseline \
    --workloads code_agent \
    --concurrency 8,12,16 \
    --output-root results/raw/
```

**预期**:
- N=8-12: 频繁 preemption, throughput 开始下降
- N=16: severe thrashing 或 OOM — 这就是 ceiling

### Step 2.4: vLLM Variant Configs

跑 `vllm-preserve` 和 `vllm-low-preempt` 对比：

```bash
# vllm-preserve: 和 baseline 相同（vanilla vLLM 无 preserve API）
# vllm-low-preempt: max_num_seqs=32，减少 preemption
.venv/bin/python -m harness.sweep \
    --systems vllm-preserve,vllm-low-preempt \
    --workloads code_agent \
    --concurrency 1,2,4,6,8 \
    --output-root results/raw/
```

**关键观测**:
- [ ] `vllm-preserve` 和 `vllm-baseline` 结果是否一致（预期一致 — 本身就是 finding）
- [ ] `vllm-low-preempt` 在高 N 下 preemption 频率是否降低
- [ ] low-preempt 是否出现 queue wait time 增加（因为 max_num_seqs=32 限制并发）

**Phase 2 Milestone**: 有 vLLM baseline 在 3 workloads x 7 concurrency levels 的完整数据，找到 cliff point。

---

## Phase 3: Alternative Systems (Day 4-5)

**Goal**: 部署 ThunderAgent / Continuum，或退化到 trace replay。

### Step 3.1: ThunderAgent Deployment

```bash
# 先保持 vLLM 在 port 8000 运行
# 启动 ThunderAgent proxy 在 port 9000
THUNDERAGENT_REF=<pinned-commit> scripts/serve_thunderagent.sh
```

**Acceptance**:
- [ ] `curl http://localhost:9000/v1/models` 返回正确
- [ ] 带 `program_id` 的 multi-turn 请求被正确跟踪
- [ ] `/programs`, `/profiles/{program_id}`, `/metrics` 可访问
- [ ] 保存 `results/processed/thunderagent_report.json`

### Step 3.2: ThunderAgent Sweep

```bash
.venv/bin/python -m harness.sweep \
    --systems thunderagent \
    --workloads code_agent,data_agent,research_agent \
    --concurrency 1,2,4,6,8 \
    --output-root results/raw/
```

### Step 3.3: Continuum Deployment (if feasible)

```bash
CONTINUUM_REF=<pinned-commit> scripts/serve_continuum.sh
```

**风险**: Continuum 基于 vLLM 0.10.x fork，可能与当前 vLLM 0.10.2 冲突。  
**时间上限**: 如果 4 小时内无法 build，放弃本地部署。

### Step 3.4: Fallback — Trace Replay

如果 Continuum 部署失败，使用 trace replay 模式：

```bash
# 用 Phase 2 收集的真实 trace 来 replay
.venv/bin/python -m harness.trace_replayer \
    --trace-file results/raw/<run_id>.jsonl \
    --api-base http://localhost:8000/v1 \
    --concurrency 4,8,12
```

- 引用 Continuum 论文 Table 8B/A100 的数据作为 reference
- Replay 模式下可以精确控制 arrival pattern

**Phase 3 Milestone**: 至少有 vLLM baseline + ThunderAgent 的对比数据；Continuum 有数据或有 justified fallback。

---

## Phase 4: Analysis & Figures (Day 6-7)

**Goal**: 产出论文级 figures 和 findings。

### Step 4.1: Group A Metrics — System-Level

| Figure | 数据源 | X 轴 | Y 轴 |
|:---|:---|:---|:---|
| **Throughput vs N** | trace timestamps | Concurrency N | Steps/min |
| **JCT Distribution** | trace ts_start/ts_end | System | CDF of JCT |
| **KV Cache vs N** | vLLM /metrics | Time | gpu_cache_usage_perc |
| **GPU Utilization** | nvidia-smi | Time | SM util % |

```bash
.venv/bin/python -m analysis.parse_traces results/raw/
.venv/bin/python -m analysis.plots results/processed/
```

### Step 4.2: Group B Metrics — Inefficiency Diagnostic

| Metric | 目的 |
|:---|:---|
| Re-prefill count per program | 量化 KV cache eviction 的代价 |
| Tool-wait idle memory (MB*s) | agent 等 tool 时 KV cache 的浪费 |
| Queue wait time | 请求在 vLLM scheduler 中的排队时间 |
| Eviction log (who, when, tokens) | 精确定位 thrashing pattern |

需要 scheduler hook 数据（`--enable-scheduler-hook`）。

### Step 4.3: Cross-System Comparison

| 对比 | 预期 Finding |
|:---|:---|
| vllm-baseline vs thunderagent | ThunderAgent 在 cliff point 附近性能优势最大 |
| vllm-baseline vs vllm-preserve | 应该一样 — "vanilla vLLM 无 preserve API" 是 finding |
| vllm-baseline vs vllm-low-preempt | low-preempt 在高 N 下 queue wait 更长但 re-prefill 更少 |
| low N vs high N | 低 N 时所有 system 一样，高 N 时分化 |

### Step 4.4: Inefficiency Pattern Analysis

```bash
.venv/bin/python -m analysis.inefficiency_detector results/processed/
```

目标识别的 patterns:
- **Thrashing**: 反复 evict + re-prefill 同一个 program
- **Bubble**: GPU idle 因为所有 agent 都在等 tool
- **Convoy**: 一个 long program 阻塞 short programs
- **Starvation**: 某些 agent 长期得不到 GPU 时间

### Step 4.5: Findings Summary

产出一份简洁的 findings 文档：

1. **Motivation figure**: "在 N=X 时，baseline 丢了 Y% throughput，根因是 Z"
2. **System comparison**: 哪个 system 在什么条件下赢，为什么
3. **Optimization target**: 确定最有价值的优化靶点
4. **Confound control**: preemption 作为 implicit safety valve 的影响

**Phase 4 Milestone**: 有清晰的 motivation figure 和 findings summary，可以直接放进论文。

---

## Experiment Matrix Summary

### Full Matrix (sweep.yaml)

| | code_agent | data_agent | research_agent |
|:---|:---:|:---:|:---:|
| **vllm-baseline** | N=1,2,4,6,8,12,16 | N=1,2,4,6,8,12,16 | N=1,2,4,6,8,12,16 |
| **vllm-preserve** | N=1,2,4,6,8 | N=1,2,4,6,8 | N=1,2,4,6,8 |
| **vllm-low-preempt** | N=1,2,4,6,8 | N=1,2,4,6,8 | N=1,2,4,6,8 |
| **thunderagent** | N=1,2,4,6,8,12,16 | N=1,2,4,6,8,12,16 | N=1,2,4,6,8,12,16 |
| **continuum** | N=1,2,4,6,8 (if deployed) | — | — |

Total cells: ~105 (full matrix) or ~63 (without Continuum high-N)

### Priority Order

1. **P0**: vllm-baseline x code_agent x N=[1,2,4,6,8] — 最重要的 baseline
2. **P1**: vllm-baseline x {data_agent, research_agent} x N=[1,2,4,6,8] — 完整 workload coverage
3. **P2**: vllm-baseline x code_agent x N=[12,16] — 找 cliff/thrashing
4. **P3**: thunderagent x code_agent x N=[1,2,4,6,8] — system comparison
5. **P4**: vllm variants (preserve, low-preempt) — confound control
6. **P5**: thunderagent x other workloads — completeness
7. **P6**: continuum (if available) — bonus comparison

---

## Monitoring & Signals

每个 experiment cell 运行时需要记录：

```
# Group A (自动采集)
- Throughput: steps/min (from trace timestamps)
- JCT: per-agent wall-clock time
- GPU util: nvidia-smi 每秒采样
- KV cache: vllm:gpu_cache_usage_perc 每秒

# Group B (需要 scheduler hook)
- vllm:num_preemptions_total
- Eviction events: seq_id, tokens, reason, gpu_usage
- Queue wait time (if logged)

# 每个 cell 的 output
results/raw/{system}_{workload}_{N}_{timestamp}.jsonl   # trace
results/raw/{system}_{workload}_{N}_{timestamp}_metrics.json  # metrics
```

---

## Risk & Fallback

| 风险 | 触发条件 | Fallback |
|:---|:---|:---|
| vLLM OOM at high N | N >= 12 with code_agent | 降低 `--max-model-len` 到 16384 |
| Agent infinite loop | > MAX_STEPS without submit | Per-task timeout (5min) |
| ThunderAgent build fail | pip install fails | 只跑 vLLM baseline + variants |
| Continuum版本冲突 | > 4h 无法 build | 用论文数据 + trace replay |
| SWE-bench sandbox crash | subprocess leak | temp dir cleanup + per-command timeout (30s) |
| DuckDuckGo rate limit | research_agent 被限流 | 降低 DUCKDUCKGO_RATE_LIMIT_QPS, 增大间隔 |
| Disk space (79 GB) | traces > 50 GB | 及时 rsync 到远程 + 清理 old results |
