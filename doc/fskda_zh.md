# FSKDA（Fast/Slow Kimi Delta Attention）中文说明与变更记录

本文档说明本仓库中引入 **FSKDA（Fast/Slow Kimi Delta Attention）** 的全部相关变更：它如何在 **保持 KDA chunk 并行** 的前提下，引入 **双时间尺度（fast/slow）两份记忆**，并用轻量、token-local 的特征生成：
- 两套写入强度：`β_fast` / `β_slow`
- 一套输出融合系数：`λ`
- 两套 decay 的 raw 输入：`g_fast` / `g_slow`（用于 KDA gate 计算 `α_t`）

> 重要工程约束：所有 gate/feature 都 **不依赖 `S_{t-1}`**，从而不会破坏 `chunk_kda` 的并行前提。

---

## 1. 动机：双时间尺度记忆（fast/slow）

单份 KDA 记忆很容易同时承担：
- 近期局部模式（短期、需要快速适配）
- 长期统计/主题信息（长期、需要稳定累积）

FSKDA 维护两份独立的递推状态：
- **fast memory**：更新更“激进”，更快适配新 token
- **slow memory**：更新更“保守”，更慢变化、更稳定

并在输出端用 `λ` 做融合，使得模型能对不同 token 的难度/不确定性自动选择更偏向 fast 或 slow。

---

## 2. 公式（单头写法，忽略 batch/head 下标）

对 fast/slow 两份记忆分别进行 KDA 型更新（形式保持一致）：

`S_t^(x) = (I - β_t^(x) k_t k_t^T) Diag(α_t^(x)) S_{t-1}^(x) + β_t^(x) k_t v_t^T,  x∈{f,s}`

其中：
- `β_t^(f)` / `β_t^(s)` 为每 token/head 的标量写入强度（由 gate 生成）
- `α_t^(f)` / `α_t^(s)` 仍由 KDA gate 内核根据 `g_raw^(x), A_log, dt_bias` 计算得到（`use_gate_in_kernel=True`）

输出融合采用“融合输出”方式（实现更直接）：

`o_t = λ_t o_t^(f) + (1-λ_t) o_t^(s),  λ_t∈(0,1)`

其中：
- `o_t^(f) = S_t^(f)^T q_t`
- `o_t^(s) = S_t^(s)^T q_t`

---

## 3. Surprise-aware gate（token-local 特征）

FSKDA 的 gate 只使用 token-local 可并行计算的统计量（与 SKDA 同一原则）：
- 误差强度：`log(1+||e||_2)`，其中 `e = W_p k - v`（proxy，不依赖 `S`）
- 不确定性：用轻量辅助 logits `Linear(hidden -> C)` 的熵（可选 margin）
- key 尺度：`log(1+||k||_2)`
- 结构信息：归一化位置 `t/T`
- head embedding：每个 head 的小 embedding（默认不训练）

通过一个小 MLP 输出：
- `β_fast, β_slow`（Sigmoid 到 `(0,1)`）
- `λ`（Sigmoid 到 `(0,1)`）
- `amp_fast, amp_slow`（标量幅度，用于构造 `g_raw`）

并构造 decay 的 raw 输入（方向来自 `k`）：

`g_raw^(x) = amp_x * normalize(k),  x∈{f,s}`

---

## 4. 可选：NKDA 风格 β 归一化（scale-invariant）

FSKDA 支持可选的 `β` 归一化（默认开启，和 NKDA 同一思想）：

`β^(x) = β^(x) / (||k||^2 + beta_norm_eps),  x∈{f,s}`

这能让同一 `β` 在不同 `||k||` 尺度下的数值行为更一致。

---

## 5. 并行性说明（为什么仍然能 chunk 并行）

FSKDA 仍然满足 chunk 并行前提：
- 对每个 token，`β_fast/β_slow/λ` 与 `g_raw_fast/g_raw_slow` 仅依赖 `(q,k,v,hidden_states,position,head_id)` 的可并行计算量；
- 不依赖 `S_{t-1}`，不会引入“状态依赖 gate”导致递推变非线性；
- fast/slow 两条递推都保持 KDA 的 DPLR 结构，因此可以直接复用 `chunk_kda`。

实现上：
- 训练/长序列：调用 `chunk_kda` **两次**（fast/slow 各一次）
- 推理短序列：可选走 `fused_recurrent_kda`（同样两次）

---

## 6. 关键注意事项

### 6.1 不要同时开启 `use_beta_norm=True` 与 kernel 内 `q/k` L2Norm

若 `use_qk_l2norm_in_kernel=True`，`||k||≈1`，则 `β/(||k||^2+eps)` 归一化会退化，FSKDA 的 `β` 归一化意义变弱。

因此实现中直接禁止：
- `use_beta_norm=True` 且 `use_qk_l2norm_in_kernel=True`（会报错）

### 6.2 KV cache 结构

FSKDA 每层会缓存两份 recurrent state：
- `recurrent_state=(recurrent_fast, recurrent_slow)`

### 6.3 参数共享与显存开销（要点）

- fast/slow **共享**同一套 `q/k/v` 投影与输出投影（参数增量主要来自 gate 的小 MLP/heads，而不是两套注意力投影）。
- fast/slow 的运行时开销来自：
  - 需要维护两份 recurrent state（KV cache/训练时的状态显存翻倍）
  - 训练/长序列会调用 `chunk_kda` 两次（fast/slow 各一次）

### 6.4 Ablation 开关

- `fix_lambda`：把融合系数 `λ` 固定为常数（例如 0.5），用于隔离“是否需要自适应融合”。
- `share_decay_gate`：fast/slow 共享同一条 decay gate（让 `α^(f)=α^(s)`），用于对比“仅双 β/双记忆”与“同时双衰减”的差异。

---

## 7. 代码变更（文件级）

新增：
- `fla/layers/fskda.py`
- `fla/models/fskda/__init__.py`
- `fla/models/fskda/configuration_fskda.py`
- `fla/models/fskda/modeling_fskda.py`
- `examples/train_fskda_wikitext103.py`
- `examples/eval_fskda_wikitext103.py`
- `doc/fskda_zh.md`

修改：
- `fla/layers/__init__.py`：导出 `FastSlowKimiDeltaAttention`
- `fla/models/__init__.py`：导出 `FSKDAConfig/FSKDAModel/FSKDAForCausalLM`

---

## 8. 训练与评测（WikiText-103）

训练示例（单卡）：

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/train_fskda_wikitext103.py \
  --output_dir exp/fskda-wt103 \
  --seq_len 1024 \
  --max_steps 2000 \
  --tokenized_cache data/wt103_tok \
  --fp16 \
  --preflight_compile
```

评测示例：

```bash
python examples/eval_fskda_wikitext103.py \
  --model exp/fskda-wt103 \
  --split validation \
  --seq_len 1024 \
  --tokenized_cache data/wt103_tok \
  --dtype fp16 \
  --generate
```
