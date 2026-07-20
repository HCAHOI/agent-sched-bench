# Tool-Time 决策方法与当前结果

日期：2026-07-14

## 结论先行

当前证据不支持部署 `offline_gated_robust_clock`。它确实减少了提前触发和短调用误触发，但没有稳定提高最终 utility；在新的 SWE-ReBench 100-task confirmation 上，10 个 cost 点的 simultaneous confidence interval 全部跨 0。

现在可以部署的策略是 **Survival-Conditioned Deadline Anchor**：不在调用开始时预测 long/short，而是在 action 的 break-even deadline 到达时重新检查调用是否仍存活；仍存活且资源压力要求执行时才触发 action。

下一版研究方法应是 **Deadline-Anchored Certified Deviation (DACD)**：deadline 是默认行为，只有当离线 task-level 置信下界证明某个 early trigger 相对 deadline 有正收益时，才安装这条偏移规则。没有足够证据就自动回退 deadline。

## 为什么不再做 long/short 二分类

短 `read/edit` 数量多并不是需要用 class reweighting 修复的标签问题。对 trigger-time policy 而言，调用在候选触发时刻之前结束后，就不会进入决策集合。真正需要判断的是：

> 在调用已经存活到时刻 `k` 的条件下，现在触发 action 是否比继续等待更有价值？

设 action cost 为 `c_a`，安全余量为 `g_a`，则默认 deadline 为：

```text
T_a = c_a + g_a
```

在当前 utility accounting 下，deadline 对足够长的调用已经取得完整收益；预测器能改善的主要是 `L in (T_a, T_a + c_a)` 的窄边界带。预测器同时必须避免把 `L <= T_a` 的调用误判为值得提前触发，因此这个边界带天然是最困难的区域。

## 可用方法：DACD

### 1. 默认 anchor

每个 action 声明自己的 cost、guard、资源收益和容量约束。运行时默认在 `T_a` 重新检查：

```text
call ends before T_a  -> no action
call survives to T_a  -> execute only if resource pressure requires it
```

对于 KV swap，这就是当前的 `deadline_only`。它不会因为短调用占多数而持续预测 short，也不会在短调用上产生 early-trigger penalty。

### 2. 离线认证 early deviation

离线 probe 只使用推理时可得的 causal context，例如 tool name、command prefix、已发生的 sequence history 和当前资源状态；禁止使用 dataset name 或调用最终时长。

对每个 context group、action 和候选触发时刻 `k < T_a`：

1. 按完整 task 划分 profile/evaluation folds。
2. 计算每个 task 上相对 deadline 的 utility 差 `Delta_j(k)`。
3. 用 task-cluster bootstrap 计算 simultaneous lower confidence bound。
4. 仅当 lower bound 严格大于 0 时安装该 early rule。
5. 任何样本不足、分布不稳定或 interval 跨 0 的情况都回退到 `T_a`。

候选 `k` 来自经验 utility 的有限 knots，不额外搜索连续超参数。多个 action 共享 causal context，但分别计算 action utility；运行时再按 CPU、HBM、host memory、disk 和带宽容量做 admission。

### 3. 能保证什么

- 短调用多数不会直接支配训练目标，因为优化对象是 utility delta，不是 long/short accuracy。
- 未认证规则不会改变 deadline 行为，策略 fail closed。
- 认证是 deployment-distribution 下的期望收益保证，不是逐调用保证。
- 方法不包含 benchmark-specific threshold；不同 workload 的规则只能由其 offline probe 样本产生。

## 当前实验结果

### 数据

| Corpus | Logical tasks used | Tool-latency samples | Role |
|---|---:|---:|---|
| SWE-ReBench development | 50 | 2,347 | method development |
| Terminal-Bench | 83 | 2,959 | method development |
| ScientificAgentBench Verified | 102 | 1,730 | method development |
| SWE-ReBench confirmation | 100 | 4,640 | frozen confirmation |

Terminal-Bench 有 100 个 trace，其中 83 个 task 产生了可提取的 tool-latency sample。

### `robust_clock` 相对 deadline 的 point estimate

单位为跨 workload 的累计秒数；正值表示优于 deadline。符号在 corpus 和 cost 之间明显变化。

| Corpus | 500 ms | 2,000 ms | 3,000 ms | 5,000 ms |
|---|---:|---:|---:|---:|
| SWE dev 50 | +65.5 | +0.5 | +70.1 | +199.8 |
| Terminal | -2.7 | -16.3 | -25.1 | +3.4 |
| ScienceAgentBench | +3.6 | -10.7 | +22.8 | +35.0 |
| SWE confirmation 100 | +1.2 | +18.9 | -11.1 | +85.0 |

因此不能挑一个全局 cost 或依据 corpus 名称选择 policy。

### Frozen confirmation：gated 相对 robust

正值表示 gated 更好。所有 simultaneous intervals 均跨 0，因此 10/10 都是 `inconclusive`。

| Cost | Paired delta | Simultaneous 95% interval | Early fires | Early short fires |
|---:|---:|---:|---:|---:|
| 500 ms | -2.3 s | [-7.3, +3.2] s | 616 -> 247 | 20 -> 10 |
| 1,000 ms | -9.2 s | [-26.0, +6.9] s | 475 -> 254 | 43 -> 26 |
| 1,500 ms | +0.3 s | [-6.4, +9.4] s | 383 -> 200 | 12 -> 7 |
| 2,000 ms | +0.9 s | [-5.3, +11.2] s | 290 -> 110 | 11 -> 8 |
| 2,500 ms | +3.0 s | [-33.6, +70.9] s | 262 -> 116 | 20 -> 7 |
| 3,000 ms | +8.3 s | [-4.5, +26.7] s | 241 -> 94 | 14 -> 9 |
| 3,500 ms | -8.8 s | [-27.8, +11.3] s | 213 -> 104 | 10 -> 8 |
| 4,000 ms | +5.7 s | [-32.4, +64.6] s | 210 -> 114 | 15 -> 8 |
| 4,500 ms | +6.2 s | [-9.8, +31.0] s | 192 -> 106 | 4 -> 1 |
| 5,000 ms | -26.5 s | [-65.5, +8.1] s | 196 -> 92 | 11 -> 6 |

结果说明 current gate 解决了“触发过多”，但没有解决“哪些触发真正产生 utility”。它过度抑制了部分有价值的 long calls，所以不能替代 `robust_clock`，更不能作为稳定优于 deadline 的方法。

## 当前建议

1. **立即可用**：部署 survival-conditioned deadline anchor，并将 action cost 和 resource-pressure gate 作为系统测量量。
2. **停止使用**：不继续调整 current global gated guard，也不从 confirmation 的 10 个 cost 中挑选有利点。
3. **下一实现**：实现 DACD 的 per-context task-level LCB certification；比较对象必须是 deadline，而不是另一个不稳定 predictor。
4. **验证方式**：现有 SWE confirmation 已经被本轮结果使用，之后再基于它设计 DACD 时只能算 development。下一次正式 confirmation 使用未来自然收集的 prequential traces，不需要立刻寻找第五个 benchmark，但必须在 trace 到达前冻结方法。

## 结果位置

- Confirmation summary: `analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/summary.md`
- Bootstrap result: `analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results/paired_task_cluster_uncertainty.json`
- Pooled policy result: `analysis/tool-time-offline-gated-robust-confirmation-swe-rebench-100-20260713/results/cv/pooled_results.json`
