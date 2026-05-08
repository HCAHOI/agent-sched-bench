# KV attention + MoE routing per-iter recording

## Context

现有 v5 trace 记录了 segment / token / 时间戳 + CPU/Mem/GPU 间隔采样，但模型内部行为（attention 流向、MoE 专家选择）从未被记录。这阻碍下游分析：哪些 segment 被 attention 重复读取、专家选择在不同 phase 是否集中、prefill/decode 的 segment-bucket 分布。

加一条**可选**通路，跑 agent 时旁路记录每次 LLM call 的 (attention aggregates, MoE routing)。默认关闭、production 跑分继续用外部 OpenAI-compatible 服务；recording 模式自动切 HuggingFace 后端 + 强制 concurrency=1。

下游分析（phase 分类、KV 调度、路由策略）由后续脚本读 npz 完成；**本任务只交付数据采集**。

## Current Checkpoint

- 2026-05-08: autonomous recording Terminal-Bench run requested for
  `fix-git,dna-insert,causal-inference-r,security-celery-redis-rce,schemelike-metacircular-eval`,
  sequentially with `--max-iterations 200` for every task and
  `--record-internals` enabled.
- Active run:
  `traces/terminal-bench/Qwen_Qwen3-Coder-30B-A3B-Instruct/remote-record-internals-5tasks-200iter-20260508T112330Z`
- Early artifact sanity: `fix-git/attempt_1/recordings/iter_0000` opens
  successfully. `attention.npz` has `segment_mass` shape `(221184, 4)` with
  max row-sum error `1.47e-7`; `routing.npz` has 48 routing records over 128
  experts with top-8 choices.
- That run later failed before completing useful formal recordings:
  `fix-git` wrote 37 complete calls, `dna-insert` wrote 23 complete calls, and
  the remaining tasks wrote no calls. Root causes were sidecar request overlap
  (`nested recording sessions`) and the old routing path's second full-sequence
  `output_router_logits=True` forward triggering Qwen MoE aux-loss OOM.
- Fix applied: serialize `HFRecordingProvider.chat()` with a provider-level
  lock, and record routing from `.mlp.gate` forward hooks during generation
  instead of doing a second full forward.
- Rerun after the fix:
  `traces/terminal-bench/Qwen_Qwen3-Coder-30B-A3B-Instruct/remote-record-internals-5tasks-200iter-gatehook-20260508T122208Z`
- Rerun early artifact sanity: `fix-git/attempt_1/recordings/iter_0000`
  opens successfully; `routing.npz` has `record_path == "gate"` only,
  `expert_choice` shape `(123552, 8)`, and `expert_load` shape
  `(3744, 4, 128)`. `attention.npz` segment-mass max row-sum error is
  `1.4e-7`.
- The gatehook rerun was stopped after the first completed tasks all failed
  by Terminal-Bench agent timeout. Treat these as invalid timeout artifacts,
  not model-quality results: earlier token-as-a-service runs generated much
  faster, while local HuggingFace recording is expected to be substantially
  slower. The interrupted task was `security-celery-redis-rce`; artifacts were
  left in place and the active harness/container processes were shut down.
- Follow-up Terminal-Bench runs now set the harness agent timeout to 7200
  seconds through `configs/benchmarks/terminal-bench.yaml`.
- Next ten-task diagnostic set:
  `fix-git,dna-insert,causal-inference-r,security-celery-redis-rce,schemelike-metacircular-eval,multi-source-data-merger,ode-solver-rk4,git-leak-recovery,cancel-async-tasks,feal-differential-cryptanalysis`.
  The last two are intentionally difficult for agent reasoning and trace
  analysis, but not primarily difficult because of GPU training, long builds,
  or large downloads.
- Alignment note from `fix-git`: recording has 43 iters while trace has 42
  `llm_call` actions. The raw OpenClaw trace, canonical attempt trace, and
  Terminal-Bench copied trace all agree: 43 `llm_call_start` events, 42
  `llm_call_end` events, 42 `llm_call` actions. The extra recording is
  `iter_0042`, corresponding to a final `llm_call_start` at the agent timeout
  boundary. This is not a collector copy bug; it is an interrupted final
  OpenClaw iteration where the HF sidecar finished and flushed recording, but
  OpenClaw did not emit the after-iteration trace action before harness
  shutdown.
- Local sync fix prepared: attempt finalization now writes `recordings/meta.json`
  after canonical `trace.jsonl` is available, keeps only trace-aligned calls in
  `meta.iters` when the trace `iteration`/`action_id` proves the mapping, and
  preserves unpaired flushed recordings under `meta.orphan_iters`.
- 2026-05-08: no-recording remote smoke for Terminal-Bench `fix-git` passed
  against the temporary HF/OpenAI-compatible proxy with `--max-iterations 50`.
- Run directory:
  `traces/terminal-bench/Qwen_Qwen3-Coder-30B-A3B-Instruct/remote-smoke-hf-norecord-fix-git-50iter-20260508T102747Z`
- Result: `success=true`, `exit_status=completed`, elapsed `237.98s`.
- 2026-05-08: recording smoke for `fix-git` ran with sampled-row attention and
  MoE routing enabled. The 50-iteration run completed without OOM and wrote 40
  recording iterations, but Terminal-Bench task success was false. A follow-up
  one-call sanity run passed `scripts/load_recording.py --call-idx 0` after
  normalizing `segment_mass` rows in the small aggregated tensor.

---

## YAGNI 准则（执行期持续生效，不是一次性检查）

实现这个特性的 agent 必须**主动**精简，而不是只在被指出后修：

- 冗余抽象立刻删——只有一个 impl 就不要 Protocol/Factory/Strategy
- 不写"启动期 dummy forward 校验"、"配置健全性兜底"这类防御代码——首个真实调用就会暴露问题，让它原地报错
- 实现过程中留下的 `# TODO: handle X case`、`# 暂时这样写`、`# 兼容性待定`、`# fallback`、`# Phase 1: ...` 这类阶段性注释，**收尾时全部回扫删除**
- 不预留"未来可能用得到"的参数 / 字段 / 文件 / config layer
- 测试只覆盖会改的契约（segment_mass 行和=1、shape、call_idx 对齐）；不要为每个内部函数都堆一个单测文件
- 配置默认值放 dataclass 里就够，**不预先**加 YAML config——出现第二个真实 use case 才加

如果发现自己在做以上任何一条，**就地删掉**，不要等 review。

---

## 调研结论（决定设计的关键点）

| 项 | 现状 | 引用 |
|---|---|---|
| Provider 抽象 | `LLMProvider(ABC)` 已存在，新 backend 实现它即可 | `src/agents/openclaw/providers/base.py:65` |
| Sidecar 落盘模板 | `ProcessStatsSampler` per-attempt 启停，写 `<attempt_dir>/resources.json` | `src/trace_collect/attempt_pipeline.py:307-336` |
| call_idx | 隐含在 `action_id`（"llm_1" 起 1-based 单调） | `src/agents/tongyi_deepresearch/trace.py:221-223` |
| In-process 模型先例 | `InProcessEngine` 用 vLLM 懒加载 + forward hook，profile-gpu 在用 | `src/serving/in_process_engine.py:12-71` |
| Keying | 没有 trace_id；用 `<attempt_dir>/recordings/` | `src/harness/trace_logger.py:11-29`, `attempt_layout.py:16-21` |

---

## 架构

Recording 关闭（默认）：CLI → `UnifiedProvider` → HTTP → vLLM。

Recording 开启（`--record-internals`）：CLI → `HFRecordingProvider` → in-process HF (`attn_implementation="sdpa"`) → `generate()` → `LayerCapturer`（attention hook 内只计算 sampled query rows + MoE 记录）→ `<attempt_dir>/recordings/iter_NNNN/{attention.npz,routing.npz,segments.json}`，外加一个 `recordings/meta.json`。

**强制 concurrency=1**：`--record-internals` + concurrency>1 在 CLI 解析时报错退出。

---

## 文件改动

### 新增 `src/serving/recording/`

| 文件 | 内容 |
|---|---|
| `__init__.py` | export `HFRecordingProvider`, `RecordingConfig` |
| `recording.py` | `RecordingConfig` dataclass + 纯函数：query 位置选择、segment_bucket、top_k、heavy_hitter、expert_load_per_segment、decode 环形缓冲（全部放一个文件，没必要拆） |
| `hooks.py` | `LayerCapturer`：attention forward hook（用 sampled Q rows 与 K cache 自算注意力并立即聚合）+ MoE 通过 `output_router_logits=True` 取 router_logits；提供 `recording_session(call_idx, segments)` 上下文管理器 + `flush()` 写盘 |
| `backend_hf.py` | `HFRecordingProvider(LLMProvider)`：加载 HF + capturer；`async def chat()` apply_chat_template → 分消息 tokenize 计算 segment offset → `asyncio.to_thread(generate)` → flush；`start_attempt(attempt_dir)` / `finish_attempt()` 生命周期；`__init__` 末尾把 model arch 摘要存进 capturer 给 meta.json 用 |

### 修改

| 文件 | 改动 |
|---|---|
| `src/trace_collect/cli.py` (`parse_collect_args`) | 加 `--record-internals` (action="store_true")。开启时若 concurrency>1 报错 |
| `src/trace_collect/collector.py:~673` | provider 装配处分支：`record_internals` → `HFRecordingProvider` else → `UnifiedProvider`；`record_internals` 写进 `run_config` 透传到 `trace_metadata` |
| `src/trace_collect/attempt_pipeline.py:~307` | 若 provider 是 `HFRecordingProvider`：`provider.start_attempt(attempt_dir/"recordings")` 在前，`provider.finish_attempt()` 在后（紧贴 `ProcessStatsSampler.start/stop` 的位置） |

`trace_metadata` 多一个 bool 字段 `record_internals`（**不存在/false 一律视为关**），不需要改 schema 类。

### 新增脚本与测试

- `scripts/load_recording.py`：`--attempt-dir <path> --call-idx <i>` 加载 npz + json，跑 sanity check（seg_mass 行和≈1.0、expert 索引范围、token_segment_id 长度=total_tokens、call_idx 与 trace.jsonl 对齐）
- `tests/test_recording_e2e.py`：5-step toy trace，断言所有 iter 文件存在 + shape + segment_id 与 `messages_in` 一致
- `README.md`：新章节"Recording internals"——怎么开、目录结构、读取脚本、性能开销提示

---

## 关键实现细节

### 1. attention 捕获必须避免 `output_attentions=True`

`output_attentions=True` 会让 HF 保存每层每 head 的完整 `(N,N)` attention。Qwen3-Coder-30B-A3B 是 48 层、32 query heads、BF16，完整 attention buffer 为 `48 * 32 * 2 * N^2` bytes：4K tokens 约 49GB，8K tokens 约 197GB，真实 agent trace 不可行。

每层 attention module `register_forward_hook(..., with_kwargs=True)`，hook 内从 `hidden_states` 重新投影 sampled query rows，并从 `past_key_values[layer_idx]` 读取已更新的 K cache；只计算 `(batch, query_heads, sampled_queries, key_len)`，随后立刻做 `segment_bucket` / `topk(K=32)` / `heavy_hitter` 并搬到 CPU。原始 `scores` / `attn` 张量不落盘、不跨层保存。

### 2. decode 环形缓冲

`generate()` 每个 decode step 触发一次 attention hook，Q=1 但 K 增长。用 maxlen=64 的环形缓冲只留最后 64 个 decode step（每层已降维记录）。

### 3. MoE 路由

`output_router_logits=True` 取 `outputs.router_logits`（每层 `(B*L, num_experts)`），外部 `softmax → topk(top_k_experts)`。`expert_choice / expert_weight` 全 token 全层都记，`expert_load_per_segment` 用 `scatter_add` 现场聚合。

### 4. segment 边界与 token offset

输入 `messages: list[dict]`（OpenAI 格式）。在 `_tokenize_with_segments` 里逐消息 `apply_chat_template([msg], tokenize=False, add_generation_prompt=False)` + `tokenizer(...).input_ids`，记 `(seg_type, token_start, token_end)`；末尾对整段渲染再 tokenize 一次，`assert sum(seg_lens) == len(full_ids)`。`tests/test_tokenize_segments.py` 用真实 openclaw message 列表覆盖。模板对不齐就在实现时直接修，不进运行期分支。

`assistant_call` 判定：`role=="assistant" and msg.get("tool_calls")`；同时有 content + tool_calls 时归 `assistant_call`，但在 segments.json 里 carry `has_content` / `has_tool_calls` 两个 bool 给下游。

### 5. trace 对接

`meta.json.iters[i].call_idx` 严格 = trace.jsonl 里 `action_type="llm_call"` 的第 i 次出现（**0-based**，对应 `action_id="llm_{i+1}"`），README 写清楚。

### 6. 设备 / 显存

Qwen3-Coder-30B-A3B BF16 ≈ 60GB，4×4090 24GB 用 `device_map="auto"` 自动分片；hook 内只保留 sampled-row 聚合结果，并用 `.detach().cpu()` 把数据搬回 CPU RAM。强制 concurrency=1 + `asyncio.to_thread()` 调 sync `generate`。

---

## 验证

```bash
# 单元
pytest tests/test_recording_e2e.py tests/test_tokenize_segments.py -xvs

# 集成（4×4090 服务器）
omc-collect --benchmark swe-rebench --sample 1 ...                        # 关闭 — 行为应与 main 完全一致
omc-collect --benchmark swe-rebench --sample 1 --record-internals ...     # 开启 — concurrency 自动锁 1
python scripts/load_recording.py --attempt-dir <run_dir>/<inst>/attempt_1 --call-idx 0
```

`load_recording.py` 跑通即视为采集层完成；下游分析下个 PR 再做。

---

## 不做

- ❌ phase 分类 / activity tag
- ❌ vLLM FlashAttn 上 dump attention（工程量大且需改 vLLM 内核）
- ❌ 改 v5 trace JSON 字段（只 append `record_internals` bool 到 trace_metadata）
- ❌ 完整 N×N attention matrix
- ❌ 多 agent 并发下的 HF 模型（强制 concurrency=1）
- ❌ 启动期 dummy forward 自检、配置兜底、运行期 fallback 分支
