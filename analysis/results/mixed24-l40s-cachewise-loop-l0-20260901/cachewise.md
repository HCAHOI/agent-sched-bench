# L0 Original CacheWise

## 测量事实

L0 的原始 CacheWise 已完成：24/24 个任务成功，平均吞吐为
**8.368 Task/h**，makespan 为 **10,324.710 s**。同一 workload 的 FCFS 为
6.917 Task/h，因此 CacheWise 达到 FCFS 的 **1.210 倍**，Task/h 提高
**20.97%**。预先固定的要求为 1.30 倍（8.993 Task/h），本节点低于该要求。

Makespan 按 24 个任务中最大的 `arrival_s + ready_to_terminal_s` 计算；实验结束
时的清理时间不计入。相应的 1.30 倍 makespan 上限为 9,607.796 s，本节点高出
716.914 s。

## 配置与产物

- GPU：单卡 NVIDIA L40S；模型：`Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`。
- Workload：12 个 SQLGlot 与 12 个 PennyLane 任务；Poisson 到达率
  `0.004390159 task/s`，最后一次到达为 `5,890.426913 s`。
- vLLM：batch 上限 8、最大上下文 131,072、prefix caching 开启、eager 模式、
  GPU memory utilization 0.95；启用 CacheWise KV 回收与 waiting-request 策略。
- CPU：vLLM 使用 cores 0–2；工具容器使用 cores 3–11，每个容器最多 2 cores；
  工具命令在物理容器中执行。
- 所有任务镜像在首个任务到达前完成预取和预建；容器按计划到达时间创建，
  24 个源镜像在运行结束后仍可读取。
- 运行代码 commit：`6e57b4722ac493ddeab80dc0e4ebf96dfc50cfad`。
- 运行时间：2026-08-31 19:27:22–22:21:37 UTC；首个任务到达时间为
  19:29:18.469 UTC。
- 运行目录：
  `/home/Ubuntu/agent-sched-bench/results/mixed24-l40s-real-tools-cachewise-l0-20260901`。
- 配置记录：`protocol.json`；汇总：
  `cachewise-r1/output/throughput_summary.json`；主 trace：
  `cachewise-r1/output/simulate_cloud_model_c24_20260831T192918468.jsonl`。
- 工具资源记录、GPU 采样、vLLM 指标及日志均保存在 `cachewise-r1` 下。

## 完整性检查

- 24/24 个任务成功，失败任务与失败 action 均为 0；共执行 1,127 次 LLM
  请求、1,103 次工具调用、2,230 个 action；主流程与单元格退出码均为 0。
- 每个任务的运行时 action 类型、ID 与源 trace 逐项一致；任务标签、顺序、
  到达时间和源 trace 与 manifest 一致。
- 1,127 次请求的请求输出 token 数均与返回 token 数一致，finish reason 均为
  `length`；全部 chat 请求返回 HTTP 200。
- 每次 LLM 返回后均成功发送一次 CacheWise 策略更新，共 1,127 次，全部返回
  HTTP 200；所有请求 priority 均为 0。
- 协议允许物理工具的成功标记与源 trace 不同；本节点的 source-false /
  replay-false / newly-false 计数为 26 / 35 / 12，FCFS 为 26 / 32 / 11。
  两次运行的工具状态存在小幅波动；LLM action 序列、请求规模与返回长度保持
  固定。
- vLLM 与回放日志未出现 5xx、CUDA OOM、XID、ERROR 或 traceback。
- 工具资源采样正常结束且无采样错误，共 25,284 条；GPU 采样覆盖 makespan 的
  99.99%，最大采样间隔为 1.098 s，所有数值均有限。

## 主要指标

| 指标 | 数值 |
|---|---:|
| Task/h | 8.368 |
| Makespan | 10,324.710 s（2.868 h） |
| 相对 FCFS Task/h | 1.210×（+20.97%） |
| 平均 / 中位任务 JCT | 2,126.340 / 1,516.118 s |
| p95 / 最大任务 JCT | 5,577.135 / 5,877.763 s |
| p50 / p95 / p99 TTFT | 0.481 / 24.689 / 76.708 s |
| 最大 TTFT | 428.311 s |
| Prompt / completion tokens | 35,673,752 / 211,699 |
| 工具容器累计运行时间 | 8.920 h |
| CacheWise 更新耗时，累计 / 中位 / p95 / 最大 | 53.670 s / 47.559 ms / 118.368 ms / 245.277 ms |
| vLLM waiting requests，平均 / 最大 | 0.556 / 5 |
| KV 使用率峰值 | 99.0% |
| Prefix cache hit rate，结束 / 最小 / 最大 | 64.2% / 0.6% / 94.8% |
| GPU 利用率均值 | 41.77% |
| GPU memory activity 均值 | 28.60% |
| GPU 显存峰值 | 43,745 MiB |
| GPU 平均功率 / 能耗 | 165.31 W / 0.474 kWh |

## 解释与决定

本次原始 CacheWise 在 L0 上带来 20.97% 的 Task/h 增量，结果方向为正；单次
探索性运行提供描述性证据。该结果未达到预先固定的 30% 要求，因此按照负载
序列进入 L1：先运行 L1 FCFS，再在同一 L1 workload 上运行原始 CacheWise，
继续以 Task/h 至少达到 FCFS 的 1.30 倍作为要求。
