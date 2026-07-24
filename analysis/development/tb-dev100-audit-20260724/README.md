# Terminal-Bench dev-100 迁移审计与仲裁诊断(2026-07-24,development-only)

**暴露边界**:本目录所有内容只读取 TB 的 **dev-100 划分**(既有边界:manifest
任务列表按字典序,`random.Random(42).shuffle` 后取前 100;`tb-dev100-split.json`
为复现清单,方法字段内含生成方式)。其余 **139 个确认任务的命令、标签、误差
一律未读**,保持 untouched,仅用于最终一次性确认读数。历史注记:
`analysis/tool-time-tb-all/`(0173c40)是一次失败的聚合分析,几乎无有效结论,
如实记录其存在,但不构成可用的 TB 先验,也不改变剩余确认集的状态。

## 内容

- `tb-dev100-split.json` — dev-100 任务清单(seed=42,可复现)。
- `prefix_transfer_audit_output.txt` — 冻结 SWE-100 词表(818 前缀节点 /
  39 bin)在 TB dev-100 的 label-free 迁移审计 + fresh-277 同口径对照。
  头条数字:深度-4 前缀命中 **3.9% vs 81.4%**(count≥1);bashlex 原生解析
  失败 **44.5% vs 1.4%**(heredoc 占 TB 行的 44.7%、循环 38.4%);bin 命中
  (pop≥20)**48.7% vs 98.3%**,OOV 集中在 shell 内建(printf/set/test/…)。
  结论:fresh-277 的深前缀高覆盖是 SWE agent 模板记忆,不是可迁移结构。
- `k_arbitration_diag.py` / `k_arbitration_diag_output.txt` — k=1(现冠军
  规则)vs k=5(拟议 R1)的 exploratory 对比,任务聚类 CI,**非确认读数**:
  latency skill −0.013 [−0.093,+0.056],mem −0.074 [−0.275,+0.068],
  cpu-peak **−0.195 [−0.419,+0.001]**。位置分桶显示机制:TB 每任务中位仅
  10 个调用,k=5 使 pos4–5 的 repo 使用率从 ~55% 降到 0%、pos6+ 从 76% 降到
  ~30%,把行按在尺度失配的 SWE public 层上,损失多数变差。
  **结论:固定样本数门槛不是跨域机制**(SWE 上薄 repo 节点是损失主源,TB 上
  public 失配使早采纳任务内观测反而正确)。
- 解析兼容性量化(R1.5,仅覆盖率主张):heredoc 构造替换为占位重定向后
  bashlex 解析率 55.5% → **98.5%**(非 heredoc 失败仅 0.6%);另一路
  `shell_command_segments` 段头回退可把 bin 可提取率推到 99.6%。两者均为
  通用 shell 语义处理,不含任何 TB 特定 token/规则。

## 本次消费记录

fresh-277(早已 dev 暴露)与 TB dev-100(既有 dev 划分)各新增若干次
exploratory 读数;TB 确认集零消费。
