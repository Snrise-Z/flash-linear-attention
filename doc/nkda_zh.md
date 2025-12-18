# NKDA（Normalized KDA）中文说明与变更记录

本文档说明在本仓库中为 **NKDA（Normalized Kimi Delta Attention）** 引入的全部相关变更：算法动机、实现位置、配置项、并行性保证、以及训练/评测脚本的用法与 ablation 开关。

> 目标：在 **不修改原 KDA 行为** 的前提下，复制出一份新的注意力实现与模型类型（`nkda`），并保持 **chunk 并行训练** 路径可用。

---

## 1. 背景：KDA 的“在线优化”视角

KDA 可理解为“在线最小二乘 + 记忆衰减/遗忘”的一类 Delta Rule 更新。NKDA 的改动集中在 **写入步长（learning rate / beta）** 的尺度稳定性：当 key 的范数变化较大时，固定的 `beta_t` 会造成更新不稳定。

NKDA 采用一种非常便宜的一阶预条件/归一化近似：

**归一化写入步长：**

`beta'_t = beta_t / (||k_t||^2 + eps)`

然后沿用原本的并行 chunk 内核（`chunk_kda`），只是在调用内核时传入 `beta'`。

直觉：
- `||k||` 大 → 写入自动变小，避免“冲刷”记忆；
- `||k||` 小 → 写入自动变大，避免“写不进去”。

---

## 2. 代码变更总览（NKDA 相关）

### 2.1 新增：NKDA layer

- 新文件：`fla/layers/nkda.py`
- 新类：`NormalizedKimiDeltaAttention`

关键实现点：
1. 与 `KimiDeltaAttention` 基本同构（投影层、可选 short conv、gate、输出门等），以保证工程可复用与性能一致。
2. 在得到 `k`（形状 `[..., H, D]`）和 `beta`（形状 `[..., H]`）之后计算：
   - `k_norm_sq = sum(k^2, dim=-1)`（用 `float32` 计算以提升稳定性）
   - `beta = beta / clamp_min(k_norm_sq, beta_norm_eps)`
3. 继续调用并行内核：
   - 训练：`chunk_kda(..., beta=beta, ...)`
   - 推理短序列：`fused_recurrent_kda(..., beta=beta, ...)`

### 2.2 新增：NKDA 模型（HF Transformers 兼容）

新增目录：`fla/models/nkda/`

包含：
- `fla/models/nkda/configuration_nkda.py`
  - 新配置类：`NKDAConfig`
  - `model_type = "nkda"`
  - 新增 NKDA 专属配置项：
    - `beta_norm_eps: float = 1e-6`
    - `use_qk_l2norm_in_kernel: bool = False`
- `fla/models/nkda/modeling_nkda.py`
  - 新模型：`NKDAModel` / `NKDAForCausalLM`
  - Block 内部使用 `NormalizedKimiDeltaAttention`
- `fla/models/nkda/__init__.py`
  - 注册 `AutoConfig/AutoModel/AutoModelForCausalLM`

并将新类型导出到聚合入口：
- 更新：`fla/models/__init__.py`（导出 `NKDAConfig/NKDAModel/NKDAForCausalLM`）

### 2.3 新增：layers 导出

- 更新：`fla/layers/__init__.py`
  - 导出 `NormalizedKimiDeltaAttention`

### 2.4 新增：WikiText-103 训练/评测脚本（NKDA 版）

> 要求：不改动 KDA 原脚本，复制一份 NKDA 脚本并加 ablation 开关。

新增：
- `examples/train_nkda_wikitext103.py`
  - 与 `examples/train_kda_wikitext103.py` 同结构
  - 使用 `NKDAConfig`
  - 新增参数：
    - `--use_qk_l2norm_in_kernel`：传入 `NKDAConfig(use_qk_l2norm_in_kernel=...)`
- `examples/eval_nkda_wikitext103.py`
  - 与 `examples/eval_kda_wikitext103.py` 同结构
  - 新增参数：
    - `--use_qk_l2norm_in_kernel`：在 **加载 checkpoint 后**，对模型中所有含 `use_qk_l2norm_in_kernel` 属性的模块进行运行时覆盖（便于不重训做推理 ablation）

---

## 3. 并行性保证（训练可并行）

NKDA 的实现策略是“**不改内核，只改输入的 beta**”：

- 训练仍然走 `chunk_kda` 路径（与 KDA 相同），因此保留原本 chunk-wise 并行算法；
- 归一化只引入按 token/head 的逐元素计算：`k_norm_sq` 与 `beta / k_norm_sq`，不会破坏 chunk 并行可用性；
- 不引入新的全序列依赖（没有把计算变成串行 RLS）。

---

## 4. ablation：`use_qk_l2norm_in_kernel` 的意义与注意事项

NKDA 的归一化写入步长依赖 `||k||^2`。

- 如果 kernel 内开启 `use_qk_l2norm_in_kernel=True`，通常意味着 `k` 会在内核里做 L2 normalize（`||k||≈1`），此时 `beta/(||k||^2+eps)` 近似退化回 `beta`，NKDA 的核心改动效果会被显著削弱。
- 因此：
  - NKDA 默认 `use_qk_l2norm_in_kernel=False`；
  - 训练脚本与评测脚本提供 `--use_qk_l2norm_in_kernel` 方便做对照实验（ablation）。

建议的 ablation 组合：
- NKDA（默认）：`--use_qk_l2norm_in_kernel` 关闭
- NKDA + kernel L2Norm：打开 `--use_qk_l2norm_in_kernel`

---

## 5. 训练与评测命令示例（WikiText-103）

### 5.1 训练（单卡示例）

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/train_nkda_wikitext103.py \
  --output_dir exp/nkda-wt103 \
  --seq_len 1024 \
  --max_steps 2000 \
  --tokenized_cache data/wt103_tok \
  --fp16 \
  --preflight_compile
```

做 kernel L2Norm ablation：

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/train_nkda_wikitext103.py \
  --output_dir exp/nkda-wt103-l2norm \
  --seq_len 1024 \
  --max_steps 2000 \
  --tokenized_cache data/wt103_tok \
  --fp16 \
  --use_qk_l2norm_in_kernel
```

### 5.2 评测（loss / perplexity）

```bash
python examples/eval_nkda_wikitext103.py \
  --model exp/nkda-wt103 \
  --split validation \
  --seq_len 1024 \
  --tokenized_cache data/wt103_tok \
  --dtype fp16
```

推理 ablation（不重训，仅覆盖 runtime 开关）：

```bash
python examples/eval_nkda_wikitext103.py \
  --model exp/nkda-wt103 \
  --split validation \
  --seq_len 1024 \
  --tokenized_cache data/wt103_tok \
  --dtype fp16 \
  --use_qk_l2norm_in_kernel
```

---

## 6. 与 HF `generate()` 的 attention_mask 兼容性

HF 的 `generate()` 在部分版本/路径下会传入 3D/4D 的“扩展因果 mask”。KDA/NKDA 的实现只需要 2D padding mask（KDA 的因果性由结构保证，不需要显式 causal mask）。

因此：
- `NormalizedKimiDeltaAttention` 在 `forward()` 内将 3D/4D mask 自动降维恢复为 2D padding mask。

> 这使得使用 `model.generate()` 时不再因 attention_mask 维度不匹配而报错。

---

## 7. 变更清单（文件级）

新增：
- `fla/layers/nkda.py`
- `fla/models/nkda/__init__.py`
- `fla/models/nkda/configuration_nkda.py`
- `fla/models/nkda/modeling_nkda.py`
- `examples/train_nkda_wikitext103.py`
- `examples/eval_nkda_wikitext103.py`
- `doc/nkda_zh.md`（本文档）

修改：
- `fla/models/__init__.py`：导出 NKDA
- `fla/layers/__init__.py`：导出 NKDA layer

