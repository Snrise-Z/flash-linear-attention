# SNKDA（Surprise-aware Normalized Kimi Delta Attention）中文说明与变更记录

本文档说明本仓库中引入 **SNKDA（Surprise-aware Normalized Kimi Delta Attention）** 的全部相关变更：它如何把 SKDA（surprise 自适应）与 NKDA（尺度归一化）结合，并在工程上保持 chunk 并行可用。

---

## 1. 动机：SKDA 与 NKDA 的正交性

- NKDA：解决 “同一个 `β` 在不同 `||k||` 下数值行为不一致”的问题，本质是尺度/几何预条件。
- SKDA：解决 “这一步该不该多写/多忘” 的自适应问题，本质是数据依赖的门控（surprise-aware）。

SNKDA 的做法是：
1. 用 SKDA 的特征工程得到 `β_sk`（语义上“想走多大步”）
2. 再做 NKDA 风格的归一化：`β = β_sk / (||k||^2 + eps)`（几何上“单位换算 + 安全气囊”）

---

## 2. 公式（单头写法）

SKDA 输出：
- `β_sk ∈ (0, 1)`（每 token/head 标量）
- `g_raw = amp * normalize(k)`（每 token/head 的 `d_k` 向量，用于 KDA gate 计算 decay）

NKDA 归一化：

`β = β_sk / (||k||^2 + beta_norm_eps)`

最后依然交给 KDA 内核（chunk 并行）：

`S_t = (I - β k k^T) Diag(α_t) S_{t-1} + β k v^T`

其中 `α_t` 仍由 KDA gate 内核根据 `g_raw, A_log, dt_bias` 计算得到（`use_gate_in_kernel=True`）。

---

## 3. 并行性说明

SNKDA 仍然满足 chunk 并行前提：
- `β` 与 `g_raw` 只依赖当前 token 的 `(k, v, hidden_states, position, head_id)`；
- 不依赖 `S_{t-1}`，不会把递推变成非线性；
- 因此可以继续复用 `chunk_kda` 的 chunk-parallel kernel。

---

## 4. 关键注意事项

### 4.1 不要同时开启 kernel 内 `q/k` L2Norm

如果 `use_qk_l2norm_in_kernel=True`，`||k||≈1`，则 `β/(||k||^2+eps)` 会退化成近似常数缩放，SNKDA 的核心归一化意义大幅减弱。

因此：
- SNKDA 默认 `use_qk_l2norm_in_kernel=False`
- 并在实现里禁止 `use_beta_norm=True` 与 `use_qk_l2norm_in_kernel=True` 同时开启（直接报错）。

### 4.2 `g_raw` 与 `β` 的“用力点”不同

- `g_raw = amp * normalize(k)`：提供 decay 的方向/强度输入（用于 `α_t`），不与 `β` 的归一化冲突。
- `β` 归一化：只影响写入/投影强度的尺度一致性。

---

## 5. 代码变更（文件级）

新增：
- `fla/layers/snkda.py`
- `fla/models/snkda/__init__.py`
- `fla/models/snkda/configuration_snkda.py`
- `fla/models/snkda/modeling_snkda.py`
- `examples/train_snkda_wikitext103.py`
- `examples/eval_snkda_wikitext103.py`
- `doc/snkda_zh.md`

修改：
- `fla/layers/__init__.py`：导出 `SurpriseNormalizedKimiDeltaAttention`
- `fla/models/__init__.py`：导出 `SNKDAConfig/SNKDAModel/SNKDAForCausalLM`

---

## 6. 训练与评测（WikiText-103）

训练示例（单卡）：

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/train_snkda_wikitext103.py \
  --output_dir exp/snkda-wt103 \
  --seq_len 1024 \
  --max_steps 2000 \
  --tokenized_cache data/wt103_tok \
  --fp16 \
  --preflight_compile
```

评测示例：

```bash
python examples/eval_snkda_wikitext103.py \
  --model exp/snkda-wt103 \
  --split validation \
  --seq_len 1024 \
  --tokenized_cache data/wt103_tok \
  --dtype fp16 \
  --generate
```

