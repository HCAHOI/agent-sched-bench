# sparse_attention — 插件层 schema

`src/serving/sparse_attention/` 与 `src/serving/kv_policies/` 在结构上**对
称、运行时互斥**：

| 子系统 | 改什么 | 何时介入 | artifact |
|--------|--------|---------|----------|
| `kv_policies` | K/V cache 内容（物理 drop） | `Cache.update()` | `kv_eviction.npz` |
| `sparse_attention` | per-query attention mask（保留 K/V） | self_attn pre-forward hook | `sparse_attention.npz` |

CLI 层 `validate_attention_method_exclusivity()` 在 config 解析完之后、模型
加载之前抛 `ValueError`；Provider `__init__` 再兜一层 assert。

---

## 协议（`base.py`）

```python
class BaseSparseAttention(Protocol):
    name: str
    def build_additive_mask(*, layer_idx, query_len, key_len, phase,
                            device, dtype) -> torch.Tensor: ...
    def record_metadata(*, layer_idx, phase, decode_step) -> dict: ...
```

- 返回 4D additive mask，broadcast 到 `[B, H, Q, K]`。`0` = 允许，`-inf` =
  屏蔽。
- HF SDPA 路径上游已经叠了 causal mask，**method 自己不再加 causal**。
- `record_metadata` 返回的 dict 直接进 recorder 的 `extras_json` 列。

---

## sliding（`sliding.py`）

```
k ∈ [0, sink_size)                     → 0
k ∈ [key_len - recent_window, key_len) → 0
其余                                     → -inf
```

每个 query row 共用同一份 key mask（mask shape `[1,1,1,K]`，靠 broadcast）。

约束（构造时校验）：

- `sink_size >= 0`、`recent_window >= 0`
- `sink_size + recent_window > 0`（否则全屏蔽，无意义）

当 `sink_size + recent_window >= key_len`，两段已覆盖全长度，等效 dense
attention。

---

## SparseAttentionConfig（frozen dataclass）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `name` | `"none" \| "sliding"` | — | 必填 |
| `record` | bool | `False` | 是否写 `sparse_attention.npz` |
| `sink_size` | int | `4` | sliding 专属；其他 method 忽略 |
| `recent_window` | int | `256` | sliding 专属 |

---

## CLI / YAML

镜像 `kv_policies` 同款 overlay 顺序：

1. `--sparse-attn-config PATH` 提供 base map
2. 显式 `--sparse-attn-*` 覆盖 YAML
3. `--sparse-attn none`（argparse 默认）**不会**清掉 YAML 提供的 `name`

```yaml
# configs/sparse_attention/sliding_b260.yaml
name: sliding
sink_size: 4
recent_window: 256
record: true
```

---

## `sparse_attention.npz` schema（per-call 写盘）

每行对应一次 `(call_idx, layer, phase, decode_step)` 的 mask 决策。

| 字段 | 形状 | dtype | 含义 |
|------|------|-------|------|
| `call_idx` | scalar | i32 | 写盘时的 chat 编号 |
| `method_name` | scalar | U16 | `sliding`（未来扩展时多值） |
| `record_step` | `[R]` | i32 | append 时的 row index（运行内全局） |
| `record_layer` | `[R]` | i32 | 层号 |
| `record_phase` | `[R]` | U7 | `prefill` / `decode` |
| `record_decode_step` | `[R]` | i32 | decode 阶段步号；prefill = -1 |
| `query_len` | `[R]` | i32 | 该 forward 的 query 行数 |
| `key_len` | `[R]` | i32 | mask 作用的 key 序列长度 |
| `kept_count` | `[R]` | i32 | 未被屏蔽的 key 位置数。sliding（key-uniform）下精确；future per-query method（Quest / MInference 等）写 query rows 的 MEAN kept count |
| `density` | `[R]` | fp16 | `kept_count / key_len`（key_len=0 时 = 0）。sliding 下精确；per-query method 下是 row-mean keep fraction |
| `extras_json` | `[R]` | object/U | method 自定义元数据的 JSON 字符串 |

`extras_json` 例（sliding）—— sink_size / recent_window 已在 attempt 级
`meta.json` 的 `sparse_attention` block，行级不再重复：

```json
{}
```

future per-query method 若要 per-row density / kept-count 等扩展信息，走
`extras_json`，不动顶层 schema。Loader 端可懒解码。

---

## meta.json 增量

attempt 级与 `kv_policy` block 并列新增：

```json
"sparse_attention": {
  "method": "sliding",
  "sink_size": 4,
  "recent_window": 256,
  "record": true
}
```

iter 级 `recording_integrity` 内追加：

```json
"sparse_attention_records": <int>
```

---

## 不做（明确边界）

- 不实现 Quest / MInference / DuoAttention / NSA（留下一轮 PR）
- 不和 `kv_policies` 组合（互斥强制）
- 不写完整 mask 到 npz（density 等统计足够；如需要 raw mask 再单开 flag）
- 不动 `LayerCapturer` 的 POST-softmax hook
