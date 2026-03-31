# Agent Scheduling Benchmark Environment Spec

**Project**: Agentic AI Scheduling Optimization  
**Owner**: YU (Hao)  
**Last Updated**: 2026-03-31  
**Hardware**: 1×A100 40GB (cloud instance)  
**Dev Pipeline**: Mac (local dev) → GitHub sync → Cloud Server  

---

## 0. Repo Structure

```
agent-sched-bench/
├── README.md
├── pyproject.toml                 # uv/pip project config
├── Makefile                       # 常用命令快捷方式
│
├── configs/                       # 所有实验配置 YAML
│   ├── systems/
│   │   ├── vllm_baseline.yaml     # raw vLLM launch args
│   │   ├── continuum.yaml         # Continuum-specific config
│   │   └── thunderagent.yaml      # ThunderAgent proxy config
│   ├── workloads/
│   │   ├── code_agent.yaml        # SWE-bench 参数
│   │   ├── data_agent.yaml        # NL2SQL 参数
│   │   └── research_agent.yaml    # Deep research 参数
│   └── sweep.yaml                 # 实验矩阵定义 (N × workload × system)
│
├── src/
│   ├── agents/                    # Agent 实现
│   │   ├── base.py                # AgentBase ABC: step(), get_trace()
│   │   ├── code_agent.py          # mini-SWEAgent (SWE-bench Lite)
│   │   ├── data_agent.py          # NL2SQL on BIRD
│   │   └── research_agent.py      # Multi-step search + synthesis
│   │
│   ├── harness/                   # Benchmark 基础设施
│   │   ├── runner.py              # 核心: 并发调度 N 个 agent，发请求到 serving engine
│   │   ├── metrics.py             # Metric 收集器 (pull from vLLM /metrics endpoint)
│   │   ├── trace_logger.py        # Per-step trace logger (JSON lines)
│   │   └── scheduler_hooks.py     # vLLM scheduler instrumentation (日志注入)
│   │
│   ├── serving/                   # Serving engine 管理
│   │   ├── engine_launcher.py     # 统一启动接口: launch_vllm(), launch_continuum(), ...
│   │   └── health_check.py        # 等待 engine ready + warmup
│   │
│   └── analysis/                  # 分析 & 可视化
│       ├── parse_traces.py        # 从 JSON lines → DataFrame
│       ├── plots.py               # Throughput vs N, latency breakdown, etc.
│       └── inefficiency_detector.py  # 找 thrashing / bubble / idle patterns
│
├── scripts/
│   ├── setup_server.sh            # 一键服务器环境配置
│   ├── download_model.sh          # 下载 Llama-3.1-8B-Instruct
│   ├── run_sweep.sh               # 跑完整实验矩阵
│   └── collect_results.sh         # 汇总所有结果到 results/
│
├── data/
│   ├── swebench_lite/             # SWE-bench Lite tasks (git submodule or download)
│   ├── bird_sql/                  # BIRD NL2SQL 数据集
│   └── research_queries/          # Deep research query set (手写 20-30 个)
│
├── results/                       # 实验输出 (gitignore, 只 sync summary)
│   ├── raw/                       # 原始 trace JSON lines
│   ├── processed/                 # 聚合后的 CSV/parquet
│   └── figures/                   # 生成的图
│
└── tests/                         # 单元测试 & smoke tests
    ├── test_agent_basic.py        # 单个 agent 能跑完 1 个 task
    ├── test_harness_n2.py         # N=2 并发 smoke test
    └── test_metrics_collection.py # metrics endpoint 能正常拉数据
```

---

## 1. Environment Setup

### Checkpoint ENV-1: Server Base Environment

```bash
# scripts/setup_server.sh 的核心内容

# 1. System packages
sudo apt update && sudo apt install -y git tmux htop nvtop jq

# 2. Python (推荐 uv 管理)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11

# 3. CUDA 验证
nvidia-smi  # 确认 A100 40GB 可见
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**TODO**:
- [ ] 确认 cloud instance 的 CUDA driver 版本 (需要 >= 12.1 for latest vLLM)
- [ ] 确认 instance 有足够的 CPU RAM (推荐 >= 64GB，用于 CPU offloading 实验)
- [ ] 确认磁盘空间 (Llama-8B fp16 ≈ 16GB + vLLM cache + traces ≈ 至少 100GB)
- [ ] 设置 SSH key 用于 GitHub sync

**验收标准**: `nvidia-smi` 显示 A100-SXM-40GB，`torch.cuda.is_available()` 返回 True

---

### Checkpoint ENV-2: Model Download

```bash
# scripts/download_model.sh
# 方案 A: HuggingFace (需要 token)
DOWNLOAD_BACKEND=huggingface ./scripts/download_model.sh

# 方案 B: 如果 HF 下载太慢，用 modelscope
DOWNLOAD_BACKEND=modelscope ./scripts/download_model.sh
```

**TODO**:
- [ ] 选择模型下载方式 (HuggingFace 需要 Meta license approval，modelscope 可能更快)
- [ ] 验证模型完整性: `python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('/data/models/Llama-3.1-8B-Instruct'); print(m.config)"`
- [ ] 记录模型路径到 `.env` 文件
- [ ] 保存模型校验报告到 `results/processed/model_report.json`

**验收标准**: 能 load 模型，config 显示 hidden_size=4096, num_layers=32

---

### Checkpoint ENV-3: Serving Engine Installation

#### 3a. Raw vLLM (Baseline)

```bash
uv pip install vllm  # latest stable (截至 2026-03 应该是 0.8.x+)

# 启动测试
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --port 8000
```

**TODO**:
- [ ] 确认 vLLM 版本 (记录到 requirements.txt)
- [ ] 跑通 `curl http://localhost:8000/v1/models` 
- [ ] 跑通一个简单的 chat completion 请求
- [ ] 记录 `--max-model-len` 的选择依据 (32768 tokens × 0.5MB/token ≈ 16GB KV cache pool，加上 model weights 16GB ≈ 32GB < 40GB)
- [ ] 确认 `/metrics` endpoint 可访问 (Prometheus 格式)

**验收标准**: 能通过 OpenAI-compatible API 发送 chat completion 请求并收到回复

#### 3b. Continuum (vLLM Fork)

```bash
# Continuum 基于 vLLM 0.10.x + LMCache
# 参考: https://github.com/Hanchenli/vllm-continuum

# 可能需要单独的 venv 避免版本冲突
uv venv .venv-continuum
source .venv-continuum/bin/activate

git clone https://github.com/Hanchenli/vllm-continuum.git
cd vllm-continuum
pip install -e .
pip install lmcache==0.3.7  # 如果需要 CPU offloading

# 启动 (需要额外参数 enable TTL)
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --port 8001
    # + Continuum-specific flags (查看 repo README)
```

**TODO**:
- [ ] Fork Continuum repo，确认支持的 vLLM 版本 (论文用的 0.10.2)
- [ ] 如果版本冲突严重，**先跳过 Continuum**，用他们论文 Table 8B/A100 的数据做 reference
- [ ] 理解 Continuum 的 `program_id` 机制——agent 发请求时需要附带这个字段
- [ ] 测试 TTL 机制是否正常工作 (发两轮 multi-turn 请求，check KV cache 是否被 pin)

**验收标准**: Continuum 启动无报错，multi-turn 请求 KV cache hit rate > 0

**⚠️ 风险**: Continuum 的 vLLM 版本可能和 latest stable 不兼容。如果花 > 4 小时仍无法 build，**果断放弃本地部署，改用 trace replay 模式**。

#### 3c. ThunderAgent (Middleware) ✅ 已确认开源

```bash
# ThunderAgent: 中间件 proxy，不改 vLLM 内核
# GitHub: https://github.com/ThunderAgent-org/ThunderAgent (254 stars, MIT license)
# 论文: arxiv:2602.13692

# 安装
git clone git@github.com:HaoKang-Timmy/ThunderAgent.git
cd ThunderAgent
pip install -e .

# 使用流程:
# Step 1: 先正常启动 vLLM (port 8000)
vllm serve /data/models/Llama-3.1-8B-Instruct --port 8000

# Step 2: 启动 ThunderAgent proxy (port 9000)
thunderagent \
    --backend-type vllm \
    --backends http://localhost:8000 \
    --port 9000 \
    --metrics \
    --profile

# Step 3: Agent 发请求到 9000 而非 8000，附带 program_id
```

**Agent 侧改动** (非常小——只需加一个 `extra_body`):
```python
# 原始 OpenAI API 调用
client.chat.completions.create(model=model, messages=messages)

# ThunderAgent 调用：只多一个 program_id
client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"program_id": agent_id}  # ← 唯一改动
)
```

**TODO**:
- [ ] `pip install -e .` 验证安装成功
- [ ] 启动 ThunderAgent proxy，验证 `curl http://localhost:9000/v1/models` 返回正确
- [ ] 发一个带 `program_id` 的 multi-turn 请求，确认 proxy 日志显示 program tracking
- [ ] 确认 `--metrics` 暴露的 metrics endpoint URL 和字段
- [ ] 确认 `--profile` 的 profiling 数据输出位置
- [ ] 理解单节点模式下的行为: pause Acting program + shortest-first eviction + exponential time decay

**验收标准**: ThunderAgent proxy 启动，multi-turn 请求被正确跟踪为同一个 program

---

### Checkpoint ENV-4: GitHub Sync Pipeline

```bash
# 本地 Mac 端
git init agent-sched-bench
cd agent-sched-bench
git remote add origin git@github.com:YU_USERNAME/agent-sched-bench.git

# .gitignore
cat > .gitignore << 'EOF'
results/raw/
results/figures/
__pycache__/
.venv*/
*.pyc
data/swebench_lite/repos/  # SWE-bench 的 git repos 很大
/data/models/               # 模型不上传
.env
EOF

# 开发流程
# Mac: 写代码 → git push
# Server: git pull → 跑实验
# 或者用更自动化的方式:
# Mac: git push → server 用 cron/webhook 自动 pull
```

**TODO**:
- [ ] 创建 GitHub repo (private)
- [ ] 配置 server 的 SSH key → GitHub
- [ ] 写 Makefile 快捷命令: `make pull`, `make run-smoke`, `make run-sweep`
- [ ] 考虑是否用 `rsync` 同步 results/ 回 Mac 做分析 (results 文件可能较大)

**验收标准**: Mac 上 `git push` 后，server 上 `git pull` 能拿到最新代码

---

### Checkpoint ENV-5: vLLM Preemption & KV Cache 配置 (⚠️ 关键)

之前 probe 实验发现 "preserve always wins"——根因是 vLLM 的 preemption 充当了隐式安全阀，让 preserve 永远没有 downside。为了让 benchmark 结果有意义，**必须控制 vLLM 的 preemption 行为**，否则所有 system 的对比都会被这个 confound 污染。

#### 需要关注的 vLLM 配置参数

```bash
# vLLM 的 preemption 相关配置 (截至 v0.8.x)
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    # ↓↓↓ 以下是需要特别关注的 ↓↓↓
    --preemption-mode recompute \     # V1 默认 recompute (非 swap)
    --max-num-seqs 256 \              # 最大并发 sequences, 影响何时触发 preemption
    --port 8000
```

#### 实验需要的三种 vLLM 配置

| Config | 用途 | 关键设置 |
|:---|:---|:---|
| **vllm-baseline** | 纯 raw vLLM，end-of-turn eviction | 默认设置，不做任何 multi-turn 感知 |
| **vllm-preserve** | 手动 preserve (模拟 naive preserve) | 通过 keep-alive 或 dummy padding 保持 KV cache 存活 |
| **vllm-no-preempt** | 禁用 preemption (复现 Continuum 的 deadlock 场景) | hack scheduler 或设 `max_num_seqs` 极低 |

#### 为什么这很关键

你之前的实验发现：**在所有 200+ 配置下 preserve 都赢 recompute**。这是因为 vLLM 的 preemption = "soft preserve"——preserve 了也不怕，挤不下了系统自动 evict。但这意味着：

1. **vLLM baseline 不是真正的 "no preserve"**——即使你不显式 preserve，如果两轮请求到达间隔足够短（< vLLM 的 eviction latency），KV cache 可能还在。需要确认这个窗口有多大。

2. **Continuum 的 TTL pin 和 ThunderAgent 的 pause 的价值，只有在 preemption 可能发生的压力下才能看到**。如果 N 太小（比如 N=1），所有 system 表现一样因为没有资源竞争。

3. **你需要找到 "preemption starts to matter" 的临界并发度**——这就是之前说的 cliff point。预期在 N=4-8 附近（1×A100 40GB，48K tokens KV pool，SWE-bench agent avg 70K tokens/program）。

#### 具体怎么控制

**方案 A: 通过并发度自然触发**（推荐先做这个）

不改 vLLM 配置，靠推高 N 让系统自然进入 memory pressure：
- N=1-2: 所有 system 表现接近（KV cache 够用）
- N=4-6: 开始出现 eviction，Continuum/ThunderAgent 的优势应该显现
- N=8-16: 严重 thrashing，system 间差异最大

**方案 B: Hack vLLM scheduler 日志**（Phase 2 做）

在 vLLM 的 `LLMEngine._process_model_outputs()` 或 `Scheduler.schedule()` 里加 hook：
```python
# 在 scheduler 做 eviction 决策时记录
logger.info(f"EVICT seq_id={seq.seq_id} tokens={seq.get_len()} "
            f"reason={reason} gpu_usage={self.block_manager.gpu_utilization}")
```

这样你就能精确知道每次 eviction 是什么时候发生的、evict 了谁。

**方案 C: 对比 "有 preemption" vs "无 preemption"**（Optional，但对论文很有说服力）

如果时间允许，跑一组对比实验：
- Config X: 正常 vLLM（有 preemption）+ preserve
- Config Y: 禁用 preemption + preserve（模拟 InferCept 的 hard pin）

这会直接复现 "preserve always wins 是因为 preemption 兜底" 的发现，作为论文的 motivation figure 非常有力。

**TODO**:
- [ ] 查 vLLM latest stable 的 preemption 配置参数（`--preemption-mode` 的可选值）
- [ ] 确认 vLLM V1 (PagedAttention v2) 是否还支持 swap mode 或只有 recompute
- [ ] 写一个 smoke test: N=4 跑 code agent，检查 vLLM log 里是否出现 preemption
- [ ] 记录每个 baseline system 在不同 N 下的 preemption 频率

**验收标准**: 能确认在某个 N 值下 vLLM 开始频繁 preempt（`vllm:num_preemptions_total > 0`）

---

## 2. Agent Implementation

### Checkpoint AGENT-1: Base Agent Interface

```python
# src/agents/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class StepRecord:
    """一个 ReAct step 的完整记录"""
    step_idx: int
    phase: str                    # "reasoning" | "acting"
    
    # LLM 侧
    prompt_tokens: int
    completion_tokens: int
    llm_latency_ms: float         # 从发请求到收到回复
    
    # Tool 侧 (acting phase only)
    tool_name: Optional[str] = None
    tool_args: Optional[str] = None
    tool_result: Optional[str] = None
    tool_duration_ms: Optional[float] = None
    tool_success: Optional[bool] = None
    
    # Timestamps
    ts_start: float = 0.0
    ts_end: float = 0.0
    
    # 额外信号 (为未来 hidden state hook 预留)
    extra: dict = field(default_factory=dict)


class AgentBase(ABC):
    """所有 agent 的基类"""
    
    def __init__(self, agent_id: str, api_base: str, model: str):
        self.agent_id = agent_id   # 同时作为 ThunderAgent/Continuum 的 program_id
        self.api_base = api_base
        self.model = model
        self.trace: list[StepRecord] = []
        self.task_id: str = ""
        self.task_success: Optional[bool] = None
    
    async def _call_llm(self, messages: list[dict]) -> str:
        """统一的 LLM 调用，自动附带 program_id"""
        # ThunderAgent 和 Continuum 都需要 program_id
        # 对 raw vLLM 这个字段会被忽略
        extra_body = {"program_id": self.agent_id}
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
        )
        return response.choices[0].message.content
    
    @abstractmethod
    async def run(self, task: dict) -> bool:
        """运行 agent 完成一个 task，返回 success/failure"""
        ...
    
    def get_trace(self) -> list[dict]:
        """导出 trace 为 JSON-serializable 的 list"""
        return [vars(r) for r in self.trace]
    
    def summary(self) -> dict:
        """聚合统计"""
        total_llm_ms = sum(r.llm_latency_ms for r in self.trace)
        total_tool_ms = sum(r.tool_duration_ms or 0 for r in self.trace)
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "n_steps": len(self.trace),
            "total_llm_ms": total_llm_ms,
            "total_tool_ms": total_tool_ms,
            "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in self.trace),
            "success": self.task_success,
        }
```

**TODO**:
- [ ] 实现 `AgentBase`
- [ ] 确认 `StepRecord` 的字段覆盖了 Section 4 的所有 diagnostic metrics
- [ ] 决定是否在 `StepRecord` 里加 `program_id` 字段 (Continuum/ThunderAgent 需要)

**验收标准**: 能 import `AgentBase`，字段设计 review 通过

---

### Checkpoint AGENT-2: Coding Agent (SWE-bench)

这是最重要的 workload，也是最成熟的 benchmark。

**实现策略**: 复用 Continuum 论文中用的 mini-swe-agent 架构，但自己重写以控制 trace granularity。

```python
# src/agents/code_agent.py (核心逻辑骨架)

SYSTEM_PROMPT = """You are a software engineer. You will be given a GitHub issue 
and the relevant repository. Fix the bug by modifying the code.

Available tools:
- bash(command): Execute a bash command in the repository
- submit(patch): Submit your fix as a unified diff

Always think step by step. Use grep/find to locate relevant files first,
then read the code, then make targeted edits, then run tests."""

class CodeAgent(AgentBase):
    """SWE-bench Lite coding agent"""
    
    MAX_STEPS = 40  # 防止无限循环，同时保留更丰富的 trace
    
    async def run(self, task: dict) -> bool:
        """
        task = {
            "instance_id": "astropy__astropy-12345",
            "problem_statement": "...",
            "repo_path": "/tmp/swebench/astropy",
            "test_cmd": "pytest tests/test_foo.py",
        }
        """
        self.task_id = task["instance_id"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._format_issue(task)},
        ]
        
        for step_idx in range(self.MAX_STEPS):
            # Reasoning step: 调用 LLM
            record = StepRecord(step_idx=step_idx, phase="reasoning", ...)
            response = await self._call_llm(messages)
            # ... 记录 tokens, latency
            
            # 解析 tool call
            tool_call = self._parse_tool_call(response)
            if tool_call is None:
                break  # agent 认为完成了
            
            if tool_call.name == "submit":
                self.task_success = await self._apply_and_test(
                    tool_call.args, task
                )
                break
            
            # Acting step: 执行 tool
            record.phase = "acting"
            record.tool_name = tool_call.name
            t0 = time.monotonic()
            result = await self._execute_tool(tool_call, task)
            record.tool_duration_ms = (time.monotonic() - t0) * 1000
            record.tool_success = not result.startswith("ERROR")
            
            # 把 tool result 追加到 messages
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool output:\n{result}"})
            
            self.trace.append(record)
        
        return self.task_success or False
```

**TODO**:
- [ ] 从 SWE-bench Lite 下载任务数据 (`datasets` library 或直接 `git clone`)
- [ ] 设置简化版 sandbox: 为每个 task clone 对应 repo 到 `/tmp/swebench/{instance_id}/`，直接 `subprocess.run()` 跑 bash，不用 Docker
  - 加 per-command timeout (30s) 和 per-task timeout (5min)
  - 每个 task 结束后 cleanup temp dir
  - 注意: 不做进程隔离意味着 agent 可能 rm -rf 或写坏环境——用 `tempfile.mkdtemp()` 限制影响范围
- [ ] 实现 tool parser: 从 LLM output 中提取 `bash(...)` 或 `submit(...)` 调用
- [ ] 实现 bash tool executor: `subprocess.run()` in temp dir, 加 timeout (30s)
- [ ] 选择 SWE-bench Lite tasks 子集 (20-30 个，优先选 median difficulty)
- [ ] 测试: 单个 agent 能跑完一个 task (不管 pass/fail)

**验收标准**: `CodeAgent` 能对一个 SWE-bench task 运行 5+ 步，产出 `StepRecord` trace

**关键设计决策**:

| 决策点 | 选项 A | 选项 B | 决定 |
|:---|:---|:---|:---|
| LLM output 格式 | function calling API | 自由文本 + regex | **B**: 更贴近真实 SWE agent 行为，且 function calling 和 Llama-8B 的兼容性可能有问题 |
| Sandbox | Docker per task | 简化版 (temp dir + subprocess) | **B**: 牺牲隔离性换开发速度，加 timeout + cleanup 控制风险 |
| Task 选择 | 随机 30 个 | 按 tool-call 轮数分层采样 | **按轮数分层**: 确保覆盖 short/medium/long workflows |

---

### Checkpoint AGENT-3: Data Agent (NL2SQL)

```python
# src/agents/data_agent.py (核心逻辑骨架)

SYSTEM_PROMPT = """You are a data analyst. Given a natural language question 
and a database schema, write SQL to answer the question.

Available tools:
- sql_execute(query): Execute SQL on the database and return results
- schema_inspect(table): Show column names and types for a table

If your SQL has errors, read the error message and try again."""

class DataAgent(AgentBase):
    """NL2SQL agent on BIRD dataset"""
    MAX_STEPS = 20
    
    async def run(self, task: dict) -> bool:
        """
        task = {
            "question": "How many employees ...",
            "db_path": "/data/bird_sql/databases/employee.sqlite",
            "gold_sql": "SELECT COUNT(*) FROM ...",
            "evidence": "..."
        }
        """
        # ReAct loop: schema_inspect → sql_execute → (retry if error) → done
        ...
```

**调度特征** (为什么选这个 workload):
- Tool duration 有较大方差: `schema_inspect` 是 < 1ms, `sql_execute` 从 1ms 到数秒 (复杂 JOIN)
- 天然有 retry pattern: SQL 语法错误 → LLM 读错误信息 → 改写 SQL → 再执行
- 8B 模型的 SQL 质量不高，预期大量 retry → 覆盖更多 FSM 状态

**TODO**:
- [ ] 下载 BIRD dataset (https://bird-bench.github.io/)
- [ ] 设置 SQLite 数据库文件
- [ ] 实现 `sql_execute` tool: `sqlite3.connect()` + timeout
- [ ] 实现 `schema_inspect` tool: `PRAGMA table_info()`
- [ ] 选择 50-100 个 questions (按 difficulty 分层: simple/moderate/challenging)
- [ ] 测试: 单个 agent 能跑完一个 question (包含 retry)

**验收标准**: `DataAgent` 能完成一轮 question，trace 包含至少一次 `sql_execute`

---

### Checkpoint AGENT-4: Deep Research Agent

```python
# src/agents/research_agent.py

SYSTEM_PROMPT = """You are a research assistant. Given a question, search 
the web for information and synthesize a comprehensive answer.

Available tools:
- web_search(query): Search the web, returns top-5 snippets
- page_read(url): Read full content of a web page

Use multiple searches to gather diverse perspectives. Synthesize 
information from multiple sources into a coherent answer."""

class ResearchAgent(AgentBase):
    """Multi-step search + synthesis agent"""
    MAX_STEPS = 30
    
    async def run(self, task: dict) -> bool:
        """
        task = {
            "question": "What are the latest developments in ...",
            "reference_answer": "..."  # optional, for evaluation
        }
        """
        # ReAct loop: web_search → page_read → ... → synthesize
        ...
```

**调度特征**:
- Tool duration 最大且最不可预测: web_search API 从 100ms 到 10s+
- 可能出现 fork-like pattern: "搜 A 方面" + "搜 B 方面" 的 context 会 share prefix
- Context 增长最快: 每次 `page_read` 可能往 context 加几千 tokens

**TODO**:
- [ ] 选择 search API: DuckDuckGo (无需 key，但有 rate limit) 或 SerpAPI (需要 key)
- [ ] 实现 `web_search` tool: HTTP call + parse results
- [ ] 实现 `page_read` tool: `httpx.get()` + HTML → text extraction (trafilatura/readability)
- [ ] 手写 20-30 个 research questions (覆盖不同领域避免 API cache 命中)
- [ ] 设置 rate limit handler (DuckDuckGo 大概 1 req/s)
- [ ] 测试: 单个 agent 能完成一轮 research

**验收标准**: `ResearchAgent` 运行 5+ 步，trace 中有 `web_search` 和 `page_read` 调用

---

## 3. Benchmark Harness

### Checkpoint HARNESS-1: Concurrent Runner

```python
# src/harness/runner.py

import asyncio
from typing import Type

class BenchmarkRunner:
    """并发运行 N 个 agent instances"""
    
    def __init__(
        self,
        agent_cls: Type[AgentBase],
        api_base: str,
        model: str,
        concurrency: int,
        tasks: list[dict],
    ):
        self.agent_cls = agent_cls
        self.api_base = api_base
        self.model = model
        self.concurrency = concurrency
        self.tasks = tasks
    
    async def run(self) -> list[dict]:
        """
        用 asyncio.Semaphore 控制并发度，
        所有 tasks 放进 queue，N 个 worker 并发消费。
        """
        sem = asyncio.Semaphore(self.concurrency)
        results = []
        
        async def worker(task, idx):
            async with sem:
                agent = self.agent_cls(
                    agent_id=f"agent-{idx:04d}",
                    api_base=self.api_base,
                    model=self.model,
                )
                success = await agent.run(task)
                results.append({
                    "summary": agent.summary(),
                    "trace": agent.get_trace(),
                })
        
        await asyncio.gather(*[
            worker(task, i) for i, task in enumerate(self.tasks)
        ])
        return results
```

**TODO**:
- [ ] 实现 `BenchmarkRunner`
- [ ] 添加 Poisson arrival 模式 (Continuum 论文用的) vs. closed-loop (所有 tasks 一起发)
  - **建议先用 closed-loop** (更简单，Continuum Fig 8 也有 closed-loop 的结果)
- [ ] 添加 progress bar (tqdm)
- [ ] 添加 timeout per task (防止 agent 卡死)
- [ ] 添加 graceful shutdown (Ctrl+C 能保存已跑完的 trace)

**验收标准**: N=2 并发跑 4 个 SWE-bench tasks，所有 trace 正确保存

---

### Checkpoint HARNESS-2: Metrics Collection

```python
# src/harness/metrics.py

import httpx
import re

class VLLMMetricsCollector:
    """定期拉 vLLM Prometheus metrics"""
    
    METRICS_OF_INTEREST = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting", 
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:avg_prompt_throughput_toks_per_s",
        "vllm:avg_generation_throughput_toks_per_s",
        "vllm:e2e_request_latency_seconds",  # histogram
        "vllm:time_to_first_token_seconds",   # histogram
    ]
    
    def __init__(self, metrics_url: str = "http://localhost:8000/metrics"):
        self.url = metrics_url
        self.snapshots: list[dict] = []
    
    async def poll(self, interval_s: float = 1.0):
        """每 interval_s 秒拉一次 metrics，直到被 cancel"""
        async with httpx.AsyncClient() as client:
            while True:
                resp = await client.get(self.url)
                snapshot = self._parse_prometheus(resp.text)
                snapshot["timestamp"] = time.time()
                self.snapshots.append(snapshot)
                await asyncio.sleep(interval_s)
```

**TODO**:
- [ ] 确认 vLLM 的 metrics endpoint URL (默认是 `/metrics`)
- [ ] 确认 Continuum 和 ThunderAgent 是否暴露相同/额外的 metrics
- [ ] 添加 GPU utilization 采集: `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1` → 写到文件
- [ ] 实现 metrics → JSON 的 dump

**验收标准**: 在跑 benchmark 期间，每秒能采集到 GPU util + vLLM metrics

---

### Checkpoint HARNESS-3: Trace Logger

```python
# src/harness/trace_logger.py

import json
import pathlib

class TraceLogger:
    """JSONL 格式的 trace 日志"""
    
    def __init__(self, output_dir: str, run_id: str):
        self.path = pathlib.Path(output_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")
    
    def log_step(self, agent_id: str, record: StepRecord):
        entry = {"agent_id": agent_id, **vars(record)}
        self._f.write(json.dumps(entry) + "\n")
        self._f.flush()
    
    def log_summary(self, agent_id: str, summary: dict):
        entry = {"type": "summary", "agent_id": agent_id, **summary}
        self._f.write(json.dumps(entry) + "\n")
        self._f.flush()
```

**TODO**:
- [ ] 实现 `TraceLogger`
- [ ] 定义 `run_id` 命名规范: `{system}_{workload}_{N}_{timestamp}`
- [ ] 确保 trace 足够 granular——每个 StepRecord 都有精确的 timestamp

**验收标准**: 跑完一组实验后，JSONL 文件可以被 `pandas.read_json(lines=True)` 正确读取

---

## 4. Diagnostic Metrics Spec

上次讨论定的四组 metrics，这里明确 **每个 metric 的数据源和采集方法**：

### Group A: System-Level (从 vLLM /metrics 端点拉)

| Metric | 定义 | 采集方式 |
|:---|:---|:---|
| Throughput (steps/min) | 每分钟完成的 ReAct step 数 | 从 trace timestamp 计算 |
| Avg Job Completion Time | 一个 agent 从开始到结束的 wall-clock 时间 | trace 首末 timestamp 差 |
| P90 / P95 JCT | 分位数 | 从所有 agents 的 JCT 集合计算 |
| GPU Utilization | SM 利用率 | nvidia-smi 每秒采样 |
| KV Cache Usage | GPU KV cache 占用百分比 | `vllm:gpu_cache_usage_perc` |

### Group B: Inefficiency Diagnostic (需要 vLLM scheduler hook 或 trace 分析)

| Metric | 定义 | 采集方式 |
|:---|:---|:---|
| Re-prefill Count | 一个 agent workflow 被 re-prefill 的次数 | vLLM scheduler log (需要加 hook) |
| Re-prefill Tokens | 每次 re-prefill 的 token 数 | vLLM scheduler log |
| Tool-Wait Idle Memory | agent 等 tool 时 KV cache 占用 (MB×s) | KV size × tool duration |
| Queue Wait Time | LLM request 在 waiting queue 的等待时间 | vLLM scheduler log |
| Eviction Log | 谁被 evict、什么时候、多少 tokens | vLLM scheduler log |

**TODO**:
- [ ] 确认 vLLM latest stable 是否已有 re-prefill count 的日志 (可能需要 patch)
- [ ] 如果 vLLM 没有足够的 scheduler logging，写一个 monkey-patch 在 `src/harness/scheduler_hooks.py`
- [ ] Group B 的 metrics 可以在 **Phase 2** 再完善——Phase 1 先跑 Group A 确认基本趋势

---

## 5. Trace Replay 模式 (Plan B)

如果 Continuum/ThunderAgent 部署困难，或 SWE-bench 的 Docker sandbox 太重，可以退化到 **trace replay** 模式。

**原理**: 不运行真实 agent，而是用 Continuum 论文的 collected traces (或自己收集的 traces) 来模拟 agent 请求模式。

```python
# src/harness/trace_replayer.py

class TraceReplayer:
    """Replay collected traces against a serving engine"""
    
    async def replay(self, trace_file: str, api_base: str, concurrency: int):
        """
        读取 trace 文件，按照原始 timing 向 engine 发请求。
        每个 "program" 按顺序发多轮请求，
        两轮之间等待 tool_duration_ms 模拟 tool call。
        """
        ...
```

**好处**:
- 不需要真实的 tool 环境 (SWE-bench Docker, SQLite, search API)
- 可以精确控制 arrival pattern (Poisson, closed-loop, bursty)
- 可以使用 Continuum 自己的 SWE-bench traces 作为 baseline

**TODO**:
- [ ] 确认 Continuum 是否 release 了 trace 文件 (论文说 "we will open-source traces")
- [ ] 设计 trace 文件格式: JSONL, 每行一个 {program_id, step_idx, prompt_tokens, completion_tokens, tool_name, tool_duration_ms}
- [ ] 实现 `TraceReplayer`

---

## 6. Fallback & Risk Registry

| 风险 | 概率 | 触发条件 | Fallback |
|:---|:---|:---|:---|
| Continuum 无法 build | 中 | vLLM 版本冲突 > 4h | 引用论文数据 + trace replay |
| ThunderAgent 安装问题 | 低 | 依赖冲突 | 已确认开源可用，风险低 |
| SWE-bench 简化 sandbox 不稳定 | 中 | subprocess crash / 资源泄露 | 加 timeout + temp dir cleanup；极端情况退化到 trace replay |
| A100 40GB KV cache 太小 | 低 | 即使 N=2 也 OOM | 降低 `max-model-len` 到 16384 |
| vLLM metrics endpoint 不够 | 中 | 没有 re-prefill count | 写 monkey-patch or scheduler hook |
| vLLM preemption 掩盖 system 差异 | 高 | N 太小时所有 system 表现一样 | 推高 N 到 cliff point；对比有/无 preemption 配置 |
| 模型下载慢/blocked | 低 | 中国网络问题 | 用 modelscope 镜像 |

---

## 7. Execution Timeline

### Phase 1: Foundation (Day 1-3)

```
Day 1 (ENV):
  □ ENV-1: Server base environment
  □ ENV-2: Model download
  □ ENV-3a: Raw vLLM 启动并验证
  □ ENV-4: GitHub sync pipeline

Day 2 (AGENT):
  □ AGENT-1: Base agent interface
  □ AGENT-2: Code agent (至少能跑 1 个 task)
  □ HARNESS-3: Trace logger

Day 3 (HARNESS):
  □ HARNESS-1: Concurrent runner (N=2 smoke test)
  □ HARNESS-2: Metrics collection
  □ First benchmark run: vLLM + code agent, N=[1,2,4]
```

**Phase 1 Milestone**: vLLM baseline + code agent + metrics 能 end-to-end 跑通，trace 落盘。

### Phase 2: Expand (Day 4-6)

```
Day 4 (MORE AGENTS):
  □ AGENT-3: Data agent
  □ AGENT-4: Research agent
  □ vLLM baseline: 三种 workload × N=[1,2,4,6,8]

Day 5 (BASELINE SYSTEMS):
  □ ENV-3c: ThunderAgent (如果开源了)
  □ ENV-3b: Continuum (如果 build 成功)
  □ 或: trace replay 模式 (Plan B)

Day 6 (FULL SWEEP):
  □ 完整实验矩阵: 3 systems × 3 workloads × N=[1,2,4,6,8,12,16]
  □ 收集所有 Group A metrics
```

**Phase 2 Milestone**: 完整实验矩阵跑完，有 Throughput vs N 的原始数据。

### Phase 3: Analysis (Day 7-8)

```
Day 7 (ANALYSIS):
  □ Throughput vs N 曲线 (per system, per workload)
  □ Latency breakdown: prefill / decode / queue wait / tool wait
  □ KV cache hit rate vs N
  □ 找 cliff point

Day 8 (DEEP DIVE):
  □ Group B diagnostic metrics (如果 scheduler hook 可用)
  □ Inefficiency pattern 分析
  □ 写 findings summary → 确定优化靶点
```

**Phase 3 Milestone**: 有清晰的 motivation figure——"在这种 workload 下，这个 system 因为这个原因丢了 X% 性能"。

---

## 8. Quick Reference: 关键命令

```bash
# 启动 vLLM serving
make serve-vllm

# Smoke test (单个 agent, 1 task)
make smoke-code
make smoke-data  
make smoke-research

# 跑一组实验
python -m src.harness.runner \
    --agent code \
    --system vllm \
    --concurrency 4 \
    --tasks 20 \
    --output results/raw/

# 分析
python -m src.analysis.parse_traces results/raw/
python -m src.analysis.plots results/processed/

# 从 Mac 同步结果回来
rsync -avz server:/home/user/agent-sched-bench/results/ ./results/
```

---

## 9. Resolved Design Decisions

| # | 问题 | 决定 | 备注 |
|:--|:---|:---|:---|
| 1 | Arrival pattern | **Phase 1 用 closed-loop，Phase 2 加 Poisson** | Continuum 用 Poisson，但 closed-loop 更简单也更容易控制变量 |
| 2 | SWE-bench 环境 | **简化版 sandbox** | 不用 Docker per task，直接在 cloned repo 里 `subprocess.run()` 跑 bash。牺牲隔离性换开发速度——反正要的是 trace pattern 不是 pass rate |
| 3 | ThunderAgent | **✅ 已开源**，`pip install -e .` | GitHub 254 stars, MIT license。作为 proxy 运行在 vLLM 前面，agent 侧只需加 `extra_body={"program_id": ...}` |
| 4 | Hidden state hook | **Phase 3 再加** | 先跑 baseline benchmark 确认哪些 inefficiency 是真的，再用 hidden state 去 address |
| 5 | 并发度范围 | **N=[1,2,4,6,8,12,16]** | 1 是 no-contention baseline，2-8 是预期 cliff 区间，12-16 是 heavy thrashing |
| 6 | vLLM preemption | **必须控制** (详见 ENV-5) | 之前实验 "preserve always wins" 的 confound——preemption 作为安全阀让 preserve 没有 downside。需要在 benchmark 中显式监控 preemption 频率 |

### 关于 vLLM preemption 的补充说明

**这是整个 benchmark 设计中最重要的 confound control。**

之前 probe 实验的核心发现：在 200+ 配置下 preserve always wins。事后调查发现根因是 vLLM 的 LIFO preemption 充当了 implicit safety valve——preserve 太多了系统自动 evict，所以 preserve 永远没有 downside。

对 benchmark 的影响：

- **vLLM baseline**: 标准 vLLM 是 end-of-turn eviction——完成 decoding 就扔 KV cache。如果两轮请求间隔 < eviction 延迟，可能出现 "意外的 KV cache hit"。需要确认并记录这个行为。
- **ThunderAgent**: 通过 proxy 层管理 program lifecycle，可以显式 pause (保留 KV) 和 resume。它的 eviction policy 是 shortest-first + exponential time decay。**需要确认 ThunderAgent 是否改变了底层 vLLM 的 preemption 行为**。
- **Continuum**: TTL pin 是在 vLLM 内核里做的，TTL 到期后自动释放。跟 vLLM 的 preemption 机制有交互——TTL 到期前即使 preemption 触发，pinned 的 KV cache 也不会被 evict。

**实验中需要监控的信号**:
- `vllm:num_preemptions_total`: 每个 N 值下的 preemption 总次数
- `vllm:gpu_cache_usage_perc`: GPU KV cache 利用率时序曲线
- 如果 preemption = 0 但 throughput 已经在下降 → 说明瓶颈不在 KV cache 而在 compute
- 如果 preemption > 0 且持续增长 → 说明进入了 thrashing / death spiral
