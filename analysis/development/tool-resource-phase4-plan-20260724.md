# 工具资源预测 · 下一阶段计划(2026-07-24)

**状态:development 计划。** 本文取代口头讨论,是下一轮注册实验的唯一计划;
所有历史数字均为 fresh-277(已开发暴露)上的 development 级读数。
terminal-bench 仍然封存,只留给最终冠军集合的一次性确认读数。

---

## 一、冻结边界与现行方法(截至 2026-07-23 campaign,权威记录见
`analysis/results/tool-resource-20260723/tool-resource-findings-2026-07-23.md`)

**三个预测目标**:tool latency(ms)、峰值 CPU(cores)、峰值容器内存(MB,
整容器采样峰值)。两种形态:q90/q95 分位数;部署阈值二分类(长/短、重/轻)。

**现任冠军**:

| 格子 | 冠军 | 数字(dev) |
|---|---|---|
| latency q90 静态 | 命令前缀分层 ECDF(已认证) | 844.5 ms p90 pinball |
| latency q90 在线 | 两层格 {repo,public}×{前缀 trie,tool} | 724.0 ms,skill +0.143 |
| latency 长/短 | 特征 MLP | BA 0.72/0.74 |
| mem q90 静态 | ambient 锚定按-tool 残差 ECDF | 43.3 MB |
| mem q90 在线 | 两层格 | 40.2 MB,skill +0.464 |
| mem 重/轻@500MB | 锚定生存函数 | BA 0.867 |
| cpu-peak q90 | 按-tool ECDF(最弱格,覆盖率 0.704 欠校准) | 0.457 cores |
| cpu 重/轻@2c | 最近观测 prequential 混合 | BA 0.684(冷)/0.73–0.78(稳态) |

**两层格机制(代码核实)**:public 层 = SWE-100 拟合语料的 ECDF,节点词表
= 前缀 trie(深 4→浅 1)+ tool + global,评估期冻结;repo 层 = 同词表按仓库
分键,评估流内按"已完成调用"(ts_end < 当前 ts_start)因果累积;选择 = 硬回
退,由深到浅,同键 repo 有 ≥1 样本即优先,无样本量仲裁、无混合。

**已关死的门**:命令前缀之于内存(显著负);BERT 文本编码(冻结+微调均无
稳健增益,代码已删,ecefd2c);core-seconds 目标(与时长 r=0.963,退役);
纯 call 级特征之于 cpu-heavy 冷启动(信息天花板 ≈BA 0.68)。

**per-binary 资产**:pacct 记账成本表(163 个二进制,SWE-100 20× replay,
提交于 `analysis/results/tool-resource-20260723/per_binary_cost_table*.json`);
fresh-277 的 per-bin replay 亦存在,**仅可作 dev 诊断,禁作特征/拟合输入**
(评估侧信息,且为 replay 测量,作 oracle 亦偏乐观)。特征管道
(`src/tool_resource/features.py`)已含 bashlex 子句解析 + 成本表特征,该
信息通道在注册对比 #3(MLP vs 前缀 ECDF,q90)已测过且**未过线**。

---

## 二、导师提出的三个想法(2026-07-24 讨论输入)

- **P1**:按统计支撑重组两层——public 层改为 bin 级(跨仓库更泛),
  workspace 层保留 command detail + 前缀 trie(特异)。依据:跨仓库前缀难
  重合,bin 特征更通用。
- **P2**:动态双节奏更新——第二层(workspace)per-call 更新,第一层
  (public)per-task 更新,动态补充信息。
- **P3**:组合式预测——bashlex 把复杂调用拆成多个 bin 子句,分别预测再按
  结构聚合,以高效处理 for 循环、复杂嵌套、管道。

## 三、讨论结论(Fable 5 独立讨论代理,三个 dev 探针裁决;
探针脚本可复现,均用冠军机制原码计算)

**探针 1(损失质量 vs 覆盖率)**:latency 上,public 层缺深度-4 前缀节点的
行只承载 **7.4%** 的基线 pinball 损失——92.6% 的损失落在**已有**精确深前缀
节点的行上(该桶均值 1436ms,是最难的桶);缺失行反而更容易(均值 292ms)。
mem 缺失质量 24.4%、cpu ~17%。行覆盖率(89.4%)与损失质量(92.6% 已覆盖)
之间的差异是本轮所有判断的枢纽:**该按损失质量分配实验预算,不是按行数**。

**探针 2(结构类别)**:循环行在 13,396 个 latency 行中只有 **3 个**——循
环乘数轴当场死亡。管道行的残差损失 73% 落在**已有精确深前缀节点**的行上
(pytest 类管道的条件尾宽问题),掉到浅节点的管道行只占 5.3%——"组合更擅
长管道"瞄准的行,恰是回退原则下组合永远不该接管的行。

**探针 3(选中节点样本量)**:两层格 latency 残差损失的 **79.6%** 落在
"repo 节点样本 n<20 却压过 public 节点"的行上(n=1–4 占 39.3%)——现行
"repo 有 1 个样本就赢"的硬规则是当前最大的可修损失来源。mem 64%、cpu 51%
同型。(注意:薄 repo 节点的行可能本身更难,集中性是必要非充分证据——所
以 R1 仍需注册对比裁决。)

**对三个提案的裁决**:

- **P1 强形式否决,保留弱形式。**"深前缀跨仓库不迁移"经验上不成立(agent
  命令高度刻板,81.5% 的 exec 评估行在 public 层有深度-4 节点,承载 93% 损
  失);把 public 层粗化为 bin 会在唯一需要 public 层的冷行上丢弃精确匹配。
  **保留:bin 作为共享词表中新增的一档回退**(前缀→bin→tool→global),
  约 20 行改动即获得位置无关性(命令中段的 pytest 也能命中)。
- **P2 砍掉/搁置。** 引作依据的 07-21 实验实际反对它:per-call vs per-task
  差异恰为 0.000,且 3500ms 点上 276/277 任务零变化;机制上 repo 层
  count≥1 优先使 public 更新只在"本仓库从未见过该键"时生效,而落点是本就肥
  的刻板节点。两个真实代价:评估流内更新 public 使损失跨仓库耦合,**破坏
  repo 聚类 bootstrap 依赖的可交换性**(丢掉不确定性工具);丢掉
  "public 臂 ≡ 静态冠军"的复核恒等式。近零收益换两个代价,不做。
- **P3 砍掉(本轮)。** 三重理由:① 探针 1 给出上界——完美组合预测器在其
  主战场(深前缀缺失行)最多挪动 ~7% 总损失,小于现有 CI 宽度;② 该信息通
  道(bashlex 子句 + 成本表特征)已由 MLP 在注册对比 #3 测过且 q90 未过线,
  手写代数需先解释它为何能过学习组合器没过的线;③ 技术缺陷:**分位数不可
  加**——q90(A+B) ≠ q90(A)+q90(B),正确组合需独立卷积,而同一命令内子句共
  享工作区与输入规模、正相关,独立假设恰在 q90 计费的尾部系统性低估。
  组合通道在 q-error/分类形态(MLP 获胜处)仍然在役,不是全盘否定。

## 四、TB dev-100 迁移审计改变了序列(2026-07-24 增补;
证据见 `analysis/development/tb-dev100-audit-20260724/`)

**TB 数据边界(既有,非新切)**:manifest 字典序任务列表经 seed=42 shuffle
的前 100 个 attempt = development-exposed(可细查);其余 139 个 =
untouched confirmation,只用于最终一次性确认。`tool-time-tb-all`(0173c40)
是一次失败的聚合分析,如实记录、不构成可用先验、不降级确认集。

**审计头条**:冻结 SWE-100 词表在 TB dev-100 上深度-4 前缀命中仅 **3.9%**
(fresh-277:81.4%),bashlex 原生解析失败 44.5%(heredoc 44.7%、循环
38.4% 的行),bin 命中 48.7%——fresh-277 的深前缀高覆盖是 SWE 模板记忆。
k=1 vs k=5 的 exploratory 诊断进一步显示:**固定样本数门槛不是跨域机制**
(TB 上 k=5 三目标点估计全负,cpu-peak −0.195 [−0.419,+0.001];TB 任务中位
仅 10 个调用,k=5 把早期调用按在尺度失配的 public 层上)。

**修订后的序列**:

**R1.5 — 通用解析兼容(实现暂停,待 parser bake-off 裁决;导师 2026-07-24
批准)。** bashlex+heredoc 替换 workaround(55.5%→98.5% 解析率、段头回退
99.6% bin 提取)降为四候选之一,与 mvdan.cc/sh/v3 v3.13.1、brush-parser
0.4.0、tree-sitter-bash 0.25.1 在 TB dev-100 上公平对决(行级+去重命令级
双口径;bash -n 仅作语法有效性参照;tree-sitter 含 ERROR/MISSING 不算严格
成功,部分恢复单列;固定通用 microcase 套件;构造级分歧矩阵;冷/热耗时与
内存;临时工具链只进 /tmp 不入库)。裁决产出主 parser + malformed 回退的
推荐后,才实现进 `features.py`。harness 与结果归档于
`analysis/development/tb-dev100-audit-20260724/parser-bakeoff/`。
注:prefix trie 仍走现有 shlex 归一化,bake-off 只针对结构/bin 特征
parser,不声称修复 prefix 迁移。

**R1′ — 仲裁重设计(取代固定 k;两个先验固定的候选,各在 SWE-dev 与
TB-dev-100 上一次性读数,不扫参数)**:
(a) 连续收缩:repo⊕public 按 `w = n_repo/(n_repo+1)` 混合(公式先验固定);
(b) 任务内尺度校准:public 分位数乘以本任务已完成调用的
observed/public-p50 中位比(无参数、尺度无关)。
两个 dev 读数后选一个进入确认候选;固定 k=5 版本退役。

**R2 — bin 档词表(降级为 SWE 域内问题)**:TB 上 SWE 拟合的 bin 词表对
shell 内建大面积 OOV,预期迁移增益有限;仅当 R1′ 定稿后仍有预算再做。

**明确不做**:组合式预测器、循环乘数(SWE 域内探针已否;TB 上循环占 38%,
但其处理归 R1.5 解析 + 在线层,不立独立轴)、public 层 per-task 更新、
bin-first 词表、oracle 组合研究、固定 k 门槛注册读数。

**确认读数的问题重述**:TB 一次性确认检验的是尺度无关机制组
(在线任务内层 + ambient 锚定 + R1′ 仲裁)的迁移,不是 SWE 绝对分位数的
迁移;public 深前缀覆盖明示为 SWE 域内性质。前置动作已完成(tool_args
格式已核验:`{command,timeout,working_dir}`,可提取率 100%)。

## 五、执行安排

实现走 codex(tmux),审查走独立 Claude reviewer(非作者新上下文):
1. codex 改 `evaluate_two_layer_resource.py` 加 `--min-repo-count k` 仲裁
   + 单测(含"k=1 复现现冠军数字"的恒等回归);
2. reviewer 过关(重点:仲裁只改选择、不动累积;恒等回归;判据无漂移);
3. 跑 R1 注册读数(分钟级)→ 汇报 → 视结果起 R2;
4. cpu-peak 校准(分位数加宽/保形化)记为独立候选轴,R1 之后按需立项。

已知风险:探针为单代理 dev 诊断,若 R1 注册读数与探针 3 的损失集中性矛盾,
先停下诊断分歧再继续;fresh-277 每多读一次,dev 过拟合风险递增——R1/R2 之
后应停手,把剩余问题带去 terminal-bench 确认。
