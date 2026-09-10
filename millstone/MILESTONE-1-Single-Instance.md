# MILESTONE 1 — Single-instance LLM serving

日期：2026-09-06。状态：结束本阶段探索，进入 multi-instance。本文冻结当前 mixed28 四组结果；不启动额外 single-instance 实验。

## 1. 结论与证据范围

在当前持续补充负载下，Continuum public 相对同 fork 的 FCFS 大幅缩短原始任务的平均完成时间及整个原始队列的完成时间。收益同时伴随更少的未命中输入 token、更短的首次调度等待和更短的平均 decode span。CacheWise 和 ThunderAgent 的中位任务时间接近 Continuum，但少数长任务受到严重延迟；ThunderAgent 日志直接显示反复暂停并依赖 1800 秒强制恢复。

这是单 GPU、单次运行、固定工作负载的机制诊断结果，不是跨负载普遍优越性结论。FCFS/Continuum 是同服务 fork、同预算的主对照；另外两组为各自基线实现，不能把跨栈全部差异归因于某一个调度规则。

## 2. mixed28 与运行配置

- 工作负载：28 条 SWE-ReBench 原始 agent 轨迹，15 条 SQLGlot、13 条 PennyLane；具体顺序见文末任务表与 [manifest](../analysis/development/mixed28-l40s-closed-calibration-v1/manifest.yaml)。原始轨迹来自 gpt-5.6-sol；服务端重放模型为 Qwen/Qwen3-4B-Instruct-2507-FP8。
- 所有原始任务在 t=0 ready；最多 16 个活动任务/session，prep concurrency=8、workers=1。任务并发 16 与 vLLM max-num-seqs=8 是不同层次的限制。
- 每个任务按原始 LLM/tool 因果顺序重放。真实 GPU 生成，每次输出长度强制为原始请求长度；工具返回原始结果并按原始工具耗时的 1/4 等待，非重新执行原工具计算。replay-speed=1，trace-tool-replay-speed=4。
- 有持续补充任务：replacement delay 为均值 10 秒的指数分布，seed=42，session marker 开启；原始 28 个任务全部终止后停止补充并取消仍在运行的背景任务。不是纯 burst，也不是预定相同到达流的开放系统稳态实验；完成反馈决定了各组实际补充任务数量和运行窗口。
- 单张 NVIDIA L40S（48 GB 档），tensor parallel=1，GPU memory utilization=0.95，max model len=131072，max-num-seqs=8，prefix caching 开启，KV dtype=auto，enforce eager。
- vLLM CPU affinity=0–2；任务容器 CPU affinity=3–11，CPU quota=2。工具资源监控关闭；GPU utilization、CUPTI DRAM bandwidth、KV events 与服务指标开启。

| 方法 | 实现与预算 | 请求超时 |
|---|---|---:|
| FCFS | Continuum public 同 fork 的 serve-fcfs 控制；vLLM 0.10.2 serving overlay；batched token budget=2048，chunked prefill | 1800 s |
| Continuum | public fork `316a58794a6ff86b216e579b74fd56ed0c5a911f`；program-level FCFS + 固定 2 秒 KV pinning；budget=2048 | 1800 s |
| CacheWise | 官方 reproduction fork `16cc7d43d0e1a84f68f046e6caecfef21012f3fc`；官方 predictor；budget=512 | 7200 s |
| ThunderAgent | 官方 `7ddc8610270e56d3b109eed8796b3a4360fc67c9` proxy + host vLLM 0.11.2；router mode=tr，调度周期 5 s，acting token weight=1，decay 开启 | 7200 s |

Continuum 在本文始终指该 public 实现；它没有论文完整的 cost-model/empirical-CDF TTL estimator。CacheWise predictor 使用官方训练划分，未使用 mixed28 结果训练。ThunderAgent 的 7200 秒请求超时与内部 1800 秒等待后强制恢复是两个不同机制，后者保留官方语义。

## 3. 指标定义与失败处理

- JCT：从任务 ready 到 terminal 的 wall time，包含 admission、LLM 排队/生成及工具阶段。成功任务为完成时间。CacheWise 失败任务使用实际观测到的超时终止时间作为完成时间下界。
- CacheWise 的失败请求实测 7200.17 秒；计入请求耗时，但不捏造 TTFT、输出 token 或后续未执行请求。任务执行了 69/82 次调用尝试，其中最后一次失败；因此成功请求统计只覆盖 1221 条，失败尝试另列。
- Request latency/TTFT：客户端 shadow-generation 记录，ThunderAgent 包含 proxy 等待。TPOT=(latency−TTFT)/(output_tokens−1)，仅 output_tokens>1；去掉首 token 之前的 prefill/等待，但仍是请求 wall-time 间隔，不是纯 GPU kernel 时间。
- 请求指标采用请求等权均值和线性插值 P95；另外给出 token 加权 TPOT。任务指标采用任务等权。阶段 span 会跨请求重叠，不能把累计 span 当 GPU 执行秒数。
- prompt−cached 是未命中输入 token 的代理，不是精确 FLOPs 或包含所有抢占重算的计数。API cached usage 与 Prometheus prefix lookup hit ratio 的分母不同，不能混用。

## 4. 四组任务级表现

单位：分钟。CacheWise 的 mean/P95/max 带 ≥ 表示以失败终止时间代替未知完成时间得到的下界；其中位数不受该尾部失败值影响。

| 指标 | FCFS | Continuum | CacheWise | ThunderAgent |
|---|---|---|---|---|
| 平均 | 56.75 | 32.20 | ≥39.15 | 61.82 |
| P50 | 46.78 | 25.24 | 25.85 | 25.53 |
| P95 | 111.29 | 70.64 | ≥136.60 | 247.61 |
| 最大 | 145.41 | 76.72 | ≥166.54 | 568.27 |
| 成功 / 原始任务 | 28/28 | 28/28 | 27/28 | 28/28 |
| 比 FCFS 更快 / 更慢 | — | 27 / 1 | 24 / 4 | 23 / 5 |
| 补充任务成功数 | 17 | 11 | 101 | 406 |

背景完成数不能直接作为等时长吞吐排名：ThunderAgent 原始任务窗口长达 568 分钟，Continuum 仅 77 分钟。原始任务吞吐与全系统吞吐应分别报告。

下面仅报告各自 cohort 终止窗口的**平均成功完成率**，不将其称为共同工作负载的稳态吞吐：

| 窗口内完成率 (tasks/hour) | FCFS | Continuum | CacheWise | ThunderAgent |
|---|---:|---:|---:|---:|
| 原始任务成功数 / 观测窗口 | 11.55 | 21.90 | 9.73 | 2.96 |
| 原始 + 补充成功数 / 观测窗口 | 18.57 | 30.50 | 46.11 | 45.82 |

CacheWise 分子为 27 个原始成功任务，失败任务不计成功；分母是实际终止窗口，不假设失败任务已经完成。后两组较高的全窗口完成率与原始长任务表现差可以同时成立，但各窗口长度、完成反馈和成功任务组成不同，不能据此下结论“系统吞吐更高一定来自牺牲长任务”或直接作等负载吞吐排名。


## 5. 原始任务的请求与 token 指标

下表 latency/TTFT/TPOT 均只对成功返回的请求统计；CacheWise 的 7200.17 秒失败尝试单独计入“所有尝试平均 wall time”。

| 指标 | FCFS | Continuum | CacheWise | ThunderAgent |
|---|---|---|---|---|
| 成功请求数 | 1,235 | 1,235 | 1,221 | 1,235 |
| 输入 tokens | 36,103,360 | 36,103,360 | 34,689,636 | 36,103,360 |
| 缓存命中 tokens | 2,712,000 | 26,865,568 | 27,977,520 | 24,129,120 |
| 未命中输入 tokens | 33,391,360 | 9,237,792 | 6,712,116 | 11,974,240 |
| 输出 tokens | 228,197 | 228,197 | 225,241 | 228,197 |
| latency mean (s) | 56.171 | 30.325 | 34.732 | 69.852 |
| latency P95 (s) | 117.296 | 92.860 | 77.214 | 187.974 |
| TTFT mean (s) | 31.453 | 10.401 | 22.607 | 56.284 |
| TTFT P95 (s) | 64.410 | 34.237 | 50.612 | 172.492 |
| TPOT mean (ms/token) | 131.827 | 100.901 | 63.596 | 72.670 |
| TPOT P95 (ms/token) | 234.121 | 218.633 | 107.515 | 132.371 |
| token 加权 TPOT (ms/token) | 134.503 | 108.415 | 66.084 | 73.830 |
| 所有尝试平均 wall time (s) | 56.171 | 30.325 | 40.596 | 69.852 |

## 6. FCFS 与 Continuum 的服务端阶段分解

同一组 1235 个原始请求；每格为 mean / P95，单位秒。CacheWise 失败运行未生成同等完整阶段汇总；ThunderAgent 未采集匹配的服务端逐请求阶段，不能将其 proxy 等待等同于后端 queue span。

| 阶段 | FCFS | Continuum |
|---|---|---|
| queue_s | 26.781 / 53.742 | 8.794 / 28.432 |
| prefill_s | 4.489 / 16.189 | 1.456 / 7.731 |
| decode_s | 24.721 / 67.691 | 19.926 / 60.901 |
| preempted_wait_s | 0.184 / 1.160 | 4.082 / 7.121 |
| ttft_s | 31.391 / 64.313 | 10.339 / 34.118 |
| e2e_s | 56.113 / 117.229 | 30.267 / 92.818 |

## 7. GPU、显存带宽与全运行窗口

CacheWise 没有最终 Prometheus snapshot，表中不以中途快照冒充终态 counter delta。

GPU 指标覆盖各自原始 cohort 从 ready 到全部 terminal 的窗口，包含补充负载。CacheWise 使用重建的终止窗口（约 166.54 分钟）；其余使用保存的 `serving_metrics.json.window`。GB/s 使用十进制单位。

| 指标 | FCFS | Continuum | CacheWise | ThunderAgent |
|---|---:|---:|---:|---:|
| GPU utilization mean (%) | 98.11 | 96.80 | 99.79 | 97.65 |
| Memory activity mean (%) | 50.98 | 68.90 | 77.54 | 71.65 |
| DRAM read+write mean (GB/s) | 354.54 | 486.09 | 558.76 | 511.05 |
| DRAM read+write P95 (GB/s) | 724.53 | 735.33 | 761.72 | 736.85 |
| vLLM preemption counter delta | 215 | 215 | 未汇总 | 39 |
| Prometheus prefix lookup hit ratio (%) | 47.27 | 55.87 | 未汇总 | 74.42 |

Memory activity 是显存读写活跃时间比例，不是带宽利用率；CUPTI 是被采样 CUDA context 的实际读写速率。四组 GPU utilization 都接近满载，但任务平均/尾部差异很大：该指标只说明 GPU 经常有工作，不说明它在为哪个任务做多少有效工作。

Continuum 的平均 DRAM 带宽高于 FCFS，与请求执行组成改变一致；这本身不能证明 kernel 更快。CacheWise/ThunderAgent 也有较高带宽和较好的 TPOT，却没有保护好原始长任务，进一步说明硬件忙碌程度与任务完成质量不是同一个目标。ThunderAgent proxy 暂停次数也不能用 vLLM preemption counter 代替。

## 8. 为什么 Continuum 有收益

### 8.1 它给连续任务保留跨请求的调度位置，而不是保证所有请求都少等待

从本地固定版本的官方源码读取到，`ContinuumRequestQueue.peek_request` 先选择仍有 pinned KV 的 job；否则按 job 第一次进入请求队列的时间排序。任务进入 tool 阶段后，后续请求不用总是重新按“这一次请求刚到达”的身份排队。固定 2 秒 KV pinning 给较短工具间隙后的请求保留复用机会。这里的 job 首次进入服务端时间，不等于 manifest 的 t=0 ready 时间。

[本地基线入口](../scripts/baselines/continuum_public.sh) 固定了源码版本与 overlay 范围；相关源码为该版本的 `vllm/v1/core/sched/request_queue.py`、`scheduler.py` 和 `estimate_with_func.py`。

首请求与后续请求的测量直接显示了等待的重新分配：

| 原始请求 cohort | 数量 | FCFS latency mean (s) | Continuum latency mean (s) | FCFS TTFT mean (s) | Continuum TTFT mean (s) |
|---|---:|---:|---:|---:|---:|
| 每个任务的首请求 | 28 | 21.736 | 165.859 | 15.002 | 162.319 |
| 后续请求 | 1207 | 56.970 | 27.181 | 31.835 | 6.877 |

因此，**Continuum 允许新进入服务的任务等得更久，换取正在推进的任务后续调用更快**。首请求 TTFT 上升约 10.8 倍，后续 TTFT 下降约 78.4%；后续请求数量远多于首请求，最终原始任务平均 JCT 下降约 43.3%。这不是把后台任务物理分离到另一 GPU，也不是同时减少所有人的等待。

28 个任务中 27 个更快、1 个更慢。唯一更慢的 SQLGlot-4208 从 12.11 增至 14.75 分钟（约 +21.8%）。少数任务会付出代价，但不能据此把全部收益归结为牺牲这个任务：相同原始输入下，实际未命中输入量也大幅下降。

### 8.2 输入处理工作减少，同时排队与 decode span 改善

FCFS 与 Continuum 都完成 1235 个原始请求，输入总量均为 36,103,360 tokens，输出长度总量均为 228,197 tokens。未命中输入从 33,391,360 降至 9,237,792（**−72.3%**），API usage 层面的缓存命中比例从约 7.51% 升至 74.41%。这支持“相同原始工作负载需要新处理的输入 token 更少”。不能将该比例直接称为总 GPU 工作量下降 72.3%。

服务端平均每请求变化为：

- 首次调度前 queue span：26.781 → 8.794 秒，减少 17.987 秒。
- Prefill span：4.489 → 1.456 秒，减少 3.033 秒。
- Decode span：24.721 → 19.926 秒，减少 4.795 秒。
- E2E span：56.113 → 30.267 秒，减少 25.847 秒。

从 span 的差值看，约 69.6% 的 E2E 差值体现为更短的 queue，11.7% 为更短的 prefill，18.6% 为更短的 decode，其余为小量时间口径残差。**这是观测时间分解，不是三个独立因素的因果贡献分解**：减少 prefill 也可能改善其他请求的排队和 decode 干扰；当前实验没有单独打开/关闭 pinning 与调度的消融。

因此，不应声称“绝大多数收益只是直接省掉 prefill 的几秒”，也不应声称“排队收益与缓存完全无关”。证据支持的解释是：跨请求缓存复用减少输入处理，任务级调度减少重复排队，两者共同改变服务时间与干扰。

### 8.3 去掉 prefill 后，TPOT 仍然改善，但不能叫纯 decode kernel 加速

请求等权 TPOT 从 131.827 降至 100.901 ms/token（**−23.5%**）；token 加权值从 134.503 降至 108.415 ms/token（**−19.4%**）。所以收益确实不只发生在第一个 token 之前。

合理机制解释是：输入重算减少，prefill/decode 混合执行的组成以及 active batch、上下文长度和任务交错发生变化，生成间隔随之缩短。更高平均 DRAM 带宽与此一致。但本次没有直接隔离这些贡献，因此不声称 KV reuse 改变了单个 decode kernel 的实现或单位 token 固有成本。

同时保留负面证据：Continuum 的平均 preempted wait 从 0.184 升至 4.082 秒；服务端最大 queue 从 76.61 升至 624.10 秒，最大 decode span 从 167.82 升至 453.57 秒。该等待是阶段内部的诊断量，不能再加到 queue+prefill+decode 上。整体任务尾部改善，并不意味着请求极端尾部也改善。

## 9. CacheWise / ThunderAgent：中位数快，但长任务缺少持续推进保护

| 观测 | CacheWise | ThunderAgent |
|---|---|---|
| PennyLane-4161 ready→terminal | ≥165.28 min，失败 | 568.27 min，成功 |
| PennyLane-6049 ready→terminal | 166.54 min，成功 | 315.02 min，成功 |
| 4161 请求尝试 / 计划 | 69 / 82，最后一次 ReadTimeout | 82 / 82 |
| 4161 单次最长请求 | 7200.17 s，失败 | 1878.46 s |
| 6049 单次最长请求 | 3836.15 s | 1885.70 s |
| 4161 终止前，更晚启动且先完成的补充任务 | 100 | 406 |
| 6049 终止前，更晚启动且先完成的补充任务 | 101 | 212 |
| 原始 4161 / 6049 的 1800 s 强制恢复日志次数 | 不适用 | 15 / 5 |

ThunderAgent 的具体日志包含 `wait timeout after 1800.0s, forcing resume`，恢复后又出现同一任务被标记 pause 的记录。4161 累计 LLM 调用耗时约 545.06 分钟，占其 568.27 分钟 JCT 的约 95.9%；6049 对应 295.85 / 315.02 分钟。等待不是工具执行拖慢的假象。

**结论：ThunderAgent 出现反复的长时间调度饥饿，靠超时强制恢复获得有限进展；保护一次请求免于永久等待，不等于保护整个任务持续推进。** 它最终完成，因此不是数学意义上已证明的无限等待。CacheWise 有请求超时和后续任务反复越过的结果证据，支持保护不足；未取得同等直接的暂停原因分解，不能把 ThunderAgent 的具体内部机制照搬到 CacheWise。

所有原始任务均在 t=0 ready，所以问题应表述为“持续补充负载下原始长任务的推进保护”，不是笼统声称凡早到任务都更慢。CacheWise/ThunderAgent 有 24/23 个原始任务快于 FCFS，恶化集中于部分任务。它们的更好 TPOT、缓存命中率或系统繁忙程度不能抵消这一尾部问题。

CacheWise 的成功请求只有 1221 条；较低 token 总量和更好成功请求 TPOT 不能作为完成相同全部工作的证明。失败带来的 censoring 必须随表保留。即使将 7200 秒失败尝试计入平均请求耗时，仍不能补出其 13 次未执行后续请求及失败调用本应生成的输出。

## 10. 本阶段结束，带入 multi-instance 的问题

本阶段已回答：GPU utilization 不能解释任务完成速度；public Continuum 的跨请求保留与任务级顺序在本配置中有明确收益；较好的缓存与 TPOT 并不自动提供长任务公平性。早期 legacy FCFS/Continuum 与旧超时批次不再作为当前主对照，本文只使用文末两份完整结果包。

下一阶段的研究问题是：**多个实例之间如何放置和调度有工具间隙的长任务，同时保留 prefix locality、利用可用容量，并给任务持续推进的机会？** “能使用多个实例”本身不是研究贡献。

延续的评估要求：

1. 保留 task ready→completion、首请求/后续请求 TTFT、TPOT、缓存/未命中输入量、原始/补充 cohort 及超时记录。不要仅用全窗口 token throughput 取代任务完成目标。
2. 多实例增加 routing/locality、跨实例负载分布和可用容量差异；只有实际发生 migration/KV transfer 时才测量并计入这些成本，不预设必须有迁移机制。
3. 与实际复用的基线明确同硬件预算、同请求长度与工具时间语义；区分调度/路由收益与新增 GPU 容量收益。若比较系统吞吐，使用明确的共同观测窗口或相同外生到达流。
4. 记录长任务被越过和长期无进展的情况，不能仅凭完成数奖励持续服务更短的新任务。保留失败/未完成任务，不做只看完成者的排名。

这是下一阶段的问题与指标承接，不在本里程碑中预定新的大规模实验矩阵，也不要求额外 single-instance 重跑。

## 11. 逐任务结果

单位：分钟；顺序与 manifest 一致。失败任务标记 ≥。

| 原始任务 | FCFS | Continuum | CacheWise | ThunderAgent |
|---|---|---|---|---|
| tobymao__sqlglot-2450 | 17.85 | 8.94 | 10.48 | 11.12 |
| tobymao__sqlglot-2521 | 37.40 | 14.44 | 16.56 | 23.12 |
| tobymao__sqlglot-4151 | 24.13 | 8.30 | 13.72 | 15.06 |
| PennyLaneAI__pennylane-6049 | 109.69 | 52.78 | 166.54 | 315.02 |
| tobymao__sqlglot-4448 | 24.51 | 23.29 | 14.19 | 17.68 |
| tobymao__sqlglot-4545 | 32.30 | 9.93 | 29.39 | 26.34 |
| tobymao__sqlglot-4208 | 12.11 | 14.75 | 7.31 | 8.54 |
| tobymao__sqlglot-2754 | 32.26 | 9.89 | 11.77 | 13.66 |
| PennyLaneAI__pennylane-3386 | 53.70 | 32.01 | 68.22 | 66.18 |
| tobymao__sqlglot-3230 | 28.90 | 10.33 | 13.57 | 15.43 |
| PennyLaneAI__pennylane-1357 | 31.45 | 14.96 | 17.34 | 18.88 |
| PennyLaneAI__pennylane-2603 | 71.98 | 39.10 | 83.34 | 95.18 |
| tobymao__sqlglot-2296 | 39.87 | 15.28 | 26.74 | 24.73 |
| tobymao__sqlglot-3549 | 30.77 | 22.71 | 15.46 | 18.40 |
| tobymao__sqlglot-3426 | 19.87 | 18.31 | 9.62 | 10.14 |
| PennyLaneAI__pennylane-5926 | 30.80 | 24.48 | 15.83 | 13.44 |
| tobymao__sqlglot-4148 | 37.75 | 19.07 | 14.17 | 17.21 |
| PennyLaneAI__pennylane-5860 | 70.45 | 36.30 | 30.90 | 34.14 |
| PennyLaneAI__pennylane-4697 | 111.76 | 49.50 | 74.19 | 122.41 |
| PennyLaneAI__pennylane-2601 | 75.06 | 37.66 | 33.47 | 34.76 |
| PennyLaneAI__pennylane-4343 | 79.67 | 47.51 | 52.78 | 38.54 |
| tobymao__sqlglot-3954 | 54.40 | 26.00 | 20.24 | 23.44 |
| PennyLaneAI__pennylane-889 | 79.28 | 46.98 | 32.44 | 34.06 |
| PennyLaneAI__pennylane-4161 | 145.41 | 72.37 | ≥165.28 | 568.27 |
| tobymao__sqlglot-3734 | 61.98 | 49.48 | 25.43 | 28.05 |
| PennyLaneAI__pennylane-5831 | 92.90 | 67.44 | 48.10 | 44.63 |
| PennyLaneAI__pennylane-5857 | 110.42 | 76.72 | 53.00 | 59.09 |
| tobymao__sqlglot-4040 | 72.44 | 53.19 | 26.27 | 33.44 |

## 12. 原始结果与可追溯性

- FCFS: [results/mixed28-l40s-continuum-fcfs-b8-t2048-trace4x-20260905/continuum-fcfs-r1](../results/mixed28-l40s-continuum-fcfs-b8-t2048-trace4x-20260905/continuum-fcfs-r1)。
- Continuum: [results/mixed28-l40s-continuum-public-trace4x-20260905-rerun/continuum-public-r1](../results/mixed28-l40s-continuum-public-trace4x-20260905-rerun/continuum-public-r1)。
- CacheWise: [results/mixed28-timeout7200-20260905/cachewise/cachewise-r1](../results/mixed28-timeout7200-20260905/cachewise/cachewise-r1)。
- ThunderAgent: [results/mixed28-timeout7200-20260905/thunderagent/thunderagent-r1](../results/mixed28-timeout7200-20260905/thunderagent/thunderagent-r1)。
- [FCFS/Continuum 完整包](../results/mixed28-fcfs-continuum-complete-pair-20260905.tar.gz)。
- [CacheWise/ThunderAgent 完整包](../results/mixed28-timeout7200-20260905-complete.tar.gz)，2,286,897,635 bytes、5,214 files；包含失败任务原始记录。传输 SHA-256 与 gzip 完整性验证通过。
- [7200 秒运行入口](../results/mixed28-timeout7200-20260905/run.sh)、[源码 bundle](../results/mixed28-timeout7200-20260905/run-source.bundle)、[实例释放确认](../results/mixed28-timeout7200-20260905/release-verified.json)。GPU 已释放；本地临时 API key 文件已删除。
- 读取位置：任务 `output/throughput_summary.json`、`openclaw_host_replay_status.json`；请求 `request_metrics.jsonl` / `openclaw_host_replay.jsonl`；阶段与 GPU `serving_metrics.json`；饥饿机制 `proxy.log`。CacheWise 无最终 throughput summary，JCT 由 run-id 的开始时间 `2026-09-05T18:33:03.945Z` 与任务最后事件重建，与其他组终态收尾可能相差数秒。
- [请求指标实现](../scripts/evaluation/summarize_serving_metrics.py) 定义 TTFT、TPOT 与 cohort。CacheWise 缺失的最终派生表不是原始记录丢失；完整包保留其失败现场。
