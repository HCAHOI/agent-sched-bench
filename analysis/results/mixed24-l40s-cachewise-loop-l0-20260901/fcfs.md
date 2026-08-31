# L0 FCFS

## 结果

L0 的 exact-fork FCFS 已完成：24/24 个任务成功，平均吞吐为
**6.917 Task/h**，makespan 为 **12,490.135 s**。同一 workload 下，原始
CacheWise 需要达到 **8.993 Task/h**，即 FCFS 的 1.30 倍；等价的 makespan
上限为 **9,607.796 s**。

## 配置与产物

- GPU：单卡 NVIDIA L40S；模型：`Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`。
- Workload：12 个 SQLGlot 与 12 个 PennyLane 任务；Poisson 到达率
  `0.004390159 task/s`，最后一次到达为 `5,890.426913 s`。
- vLLM：并发任务上限 24、batch 上限 8、最大上下文 131,072、prefix
  caching 开启、eager 模式、GPU memory utilization 0.95。
- CPU：vLLM 使用 cores 0–2；工具容器使用 cores 3–11，每个容器最多
  2 cores。工具命令在物理容器中执行。
- 所有任务镜像在实验计时前完成拉取并保留；容器按计划到达时间创建。
- 运行代码 commit：`8412950e2e1bf88fc45cc15f8c69633477d34c07`。
- 运行时间：2026-08-31 15:25:07–19:08:15 UTC；首个任务到达时间为
  15:39:51.500 UTC。
- 运行目录：`/home/Ubuntu/agent-sched-bench/results/mixed24-l40s-real-tools-fcfs-preloaded-20260831`。
- 配置记录：`protocol.json`；汇总：
  `cachewise-disabled-r1/output/throughput_summary.json`；主 trace：
  `cachewise-disabled-r1/output/simulate_cloud_model_c24_20260831T153951499.jsonl`。
- 工具资源记录、GPU 采样、vLLM 指标及日志均保存在
  `cachewise-disabled-r1` 下。

## 完整性检查

- 24/24 个任务成功，失败任务与失败 action 均为 0；共执行 1,127 次 LLM
  请求、1,103 次工具调用、2,230 个 action。
- 每个任务的运行时 action 类型、ID 与源 trace 逐项一致；任务标签、顺序、
  到达时间和源 trace 与 manifest 一致。
- 1,127 次请求的请求输出 token 数均与返回 token 数一致；全部 vLLM chat
  请求返回 HTTP 200。
- 所有请求 priority 均为 0，调度策略更新次数为 0；服务启动参数关闭了
  CacheWise 调度。
- vLLM 日志未出现 5xx、CUDA、XID、OOM 或 traceback；主流程与单元格退出码
  均为 0。
- 24 个镜像均在首个任务到达前完成预取和预建，并在运行结束后仍可读取。
- 工具资源采样正常结束且无采样错误；GPU 采样覆盖实验区间的 95% 以上，
  最大采样间隔为 1.075 s，所有数值均有限。

## 主要指标

| 指标 | 数值 |
|---|---:|
| Task/h | 6.917 |
| Makespan | 12,490.135 s（3.469 h） |
| 平均 / 中位任务 JCT | 3,568.677 / 3,253.341 s |
| p95 / 最大任务 JCT | 7,267.511 / 8,043.188 s |
| p50 / p95 / p99 TTFT | 11.905 / 103.760 / 123.466 s |
| 最大 TTFT | 145.456 s |
| Prompt / completion tokens | 35,673,752 / 211,699 |
| 工具容器累计运行时间 | 8.804 h |
| vLLM waiting requests，平均 / 最大 | 3.470 / 14 |
| KV 使用率峰值 | 99.9% |
| Prefix cache hit rate，结束 / 最小 / 最大 | 20.2% / 0.3% / 95.1% |
| GPU 利用率均值 | 54.37% |
| GPU memory activity 均值 | 32.54% |
| GPU 显存峰值 | 43,745 MiB |
| GPU 平均功率 / 能耗 | 202.46 W / 0.702 kWh |

## 解释与下一步

该节点使用 CacheWise 的 vLLM fork，并关闭 CacheWise 调度，因此它提供同一
serving stack 下的 FCFS 基准。L0 使用物理工具执行与计划到达，后续调度器可
直接复用相同 workload、模型和资源配置进行成对比较。

下一步运行原始 CacheWise L0。达到 8.993 Task/h 后进入 Oracle Length；若低于
该值，则按预先固定的负载序列运行 L1 FCFS 与 L1 CacheWise，并继续使用 1.30
倍 Task/h 作为要求。
