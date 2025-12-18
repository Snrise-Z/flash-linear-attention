# SKDA（Surprise-aware KDA）中文说明与变更记录

本文档说明本仓库中为 **SKDA（Surprise-aware Kimi Delta Attention）** 引入的全部相关变更：设计动机、并行性约束下的实现取舍、代码位置、配置项、以及如何使用/对照实验。

> 目标：在 **不修改原 KDA/NKDA 行为** 的前提下，复制出一份新的注意力实现与模型类型（`skda`），并保持 **chunk 并行训练** 路径可用（继续复用 `chunk_kda` 内核）。

---

## 1. 背景：你提出的“误差驱动门控”

你给出的思路是让门控依赖“本步学得好不好”的误差：

- 误差：`e_t = S_{t-1}^T k_t - v_t`
- 门控：`α_t = σ(W_α e_t)`, `β_t = σ(w_β^T e_t)`

这在在线优化/自适应优化器视角下很自然：误差大就更强更新，误差小就更少更新。

---

## 2. 关键工程约束：必须保持 chunk 并行

现有 KDA 的 chunk 并行（DPLR/WY/UT）依赖于如下性质：
- 状态递推对 `S_{t-1}` **保持仿射线性**：`S_t = A_t S_{t-1} + B_t`
- 且 `A_t` 的结构满足内核可处理（对角+低秩形式）

如果直接使用 `e_t = S_{t-1}^T k_t - v_t` 来生成 `α_t(e_t), β_t(e_t)`，那么 `A_t` 会通过 `α_t,β_t` **反过来依赖 `S_{t-1}`**，递推变成非线性：
- `S_t = F(S_{t-1})`（不再是线性仿射）

这会破坏现有 chunk 并行算法的前提，意味着要么放弃并行（串行递推），要么重写一套更复杂的并行算法。

---

## 3. SKDA 的实现取舍：使用“并行友好的误差代理”

为保证并行，SKDA 不使用真实的 `S_{t-1}` 来计算误差，而使用 **只依赖当前 token 投影的误差代理（proxy error）**：

1. 先从当前 token 的 key 预测一个 value：
   - `v_hat = P(k)`
2. 误差代理：
   - `e = v_hat - v`
3. 用 `e` 生成门控：
   - `β = sigmoid(w_beta(e))`（标量/每 head）
   - `g_raw = W_alpha(e)`（每 head 的 `d_k` 维向量，作为 KDA 的 decay raw input）
4. 仍然调用原 KDA 内核（并行不变）：
   - 训练：`chunk_kda(..., g=g_raw, beta=β, use_gate_in_kernel=True, ...)`
   - 推理短序列：`fused_recurrent_kda(..., g=fused_kda_gate(g_raw), beta=β, ...)`

注意：
- 在本实现里，**α 仍由原 KDA 的 gate 机制产生**（`use_gate_in_kernel=True`），SKDA 实际上是“误差驱动的 raw gate + 误差驱动的 beta”。
- 这种设计保持了递推对 `S` 的线性仿射结构，因此 chunk 并行可继续复用现有内核。

---

## 3.1 进一步稳定：把误差代理压缩成“统计量”

直接把 `e ∈ R^{d_v}` 的整向量喂给 gate，实践中容易被某些维度的噪声/偶然模式影响。SKDA 增强版本将误差代理先压缩为几个标量统计量，再喂给非常小的 MLP：

- `s^(L2) = ||e||_2`
- `s^(L1) = ||e||_1`
- `s^(cos) = 1 - cos(v_hat, v)`

拼成 `s = [s^(L2), s^(L1), s^(cos)]`，用于产生门控：

- `beta = sigmoid(MLP_beta(s))`
- `amp = MLP_alpha(s) / surprise_gate_logit_normalizer`

其中 `amp` 是标量幅度。为了让 `g_raw` 仍具备 `d_k` 维度，我们再从 `k` 构造一个不依赖 `S_{t-1}` 的参考向量：

- `r = normalize(k)`（逐 token/head 的 `d_k` 向量）
- `g_raw = amp * r`（对 `d_k` 维广播）

这样做的好处：
- gate 决策只依赖误差的“大小/方向偏差”这类 summary statistic，更稳、更不易过拟合某些 value 维度；
- 所有输入都只依赖当前 token 的 `(k, v)`，不包含 `S_{t-1}`，因此不破坏 chunk 并行。

---

## 3.2 极简但靠谱的 3+2 特征工程（当前实现）

为了避免一上来塞太多统计量，SKDA 当前实现采用一个 “3 + 2” 的极简特征组合：

3 个核心统计量：
1. 误差强度：`log(1 + ||e||_2)`，其中 `e = v_hat - v`
2. 模型不确定性：来自 **层内辅助 logits** 的熵（可选再加 margin）
3. key 尺度：`log(1 + ||k||_2)`

2 个结构信息：
4. 归一化位置：`t/T`（实现上使用 `position_ids/max_position_embeddings` 或由 mask/cuseqlens 推导的相对位置）
5. head embedding：每个 head 的固定向量（默认不训练）

门控形式：
- 先拼接成 `s = [s_err, s_unc, s_k, s_pos, head_emb]`
- 经过小 MLP 得到隐藏表征 `h`
- `beta = sigmoid(w_beta^T h + b)`
- `g_raw = amp * normalize(k)`，其中 `amp = (w_amp^T h + b)/normalizer`

说明：
- “不确定性”不是用最终 LM head 的 vocab logits（attention 层拿不到），而是用一个很轻量的 **辅助 logits 投影** `Linear(hidden_size -> C)` 来近似不确定性，然后计算其熵（可选 margin）。这满足“只依赖当前 token”的并行约束，且工程上更容易落地。

---

## 4. 代码变更总览（SKDA 相关）

### 4.1 新增：SKDA layer

- 新文件：`fla/layers/skda.py`
- 新类：`SurpriseKimiDeltaAttention`

实现要点：
- 结构上参考 `KimiDeltaAttention`，保留 short conv、输出门、以及 `A_log/dt_bias` 的 decay 参数化。
- 新增“surprise proxy”子网络（逐 head）：
  - `surprise_v_proj: Linear(d_k -> d_v)`：得到 `v_hat`
  - 从 `e=v_hat-v` 提取误差强度 `log1p(||e||2)`
  - `surprise_uncertainty_proj: Linear(hidden_size -> C)`：产生辅助 logits，计算熵（可选 margin）
  - 计算 key 尺度 `log1p(||k||2)`、位置 `t/T`、head embedding
  - `surprise_feat_mlp`：对特征向量做小 MLP
  - `surprise_beta_head`：输出 `β`
  - `surprise_amp_head`：输出幅度 `amp`
  - `r = normalize(k)`，`g_raw = amp * r`

新增/更新的配置项：
- `surprise_gate_logit_normalizer`：控制 `amp` 缩放
- `surprise_stat_eps`：统计量计算的数值稳定项（L2/cos 分母）
- `surprise_mlp_hidden_dim`：上述 MLP 的隐藏维度
- `surprise_head_embed_dim`：head embedding 维度
- `surprise_trainable_head_embed`：head embedding 是否参与训练（默认否）
- `surprise_uncertainty_bins`：辅助 logits 的类别数 `C`
- `surprise_include_margin`：是否把 margin 作为额外不确定性特征

### 4.2 新增：SKDA HF 模型（Transformers 兼容）

新增目录：`fla/models/skda/`

包含：
- `fla/models/skda/configuration_skda.py`
  - 新配置类：`SKDAConfig`
  - `model_type = "skda"`
  - 新增 SKDA 专属配置项：
    - `surprise_gate_logit_normalizer: float = 1.0`
    - `surprise_stat_eps: float = 1e-6`
    - `surprise_mlp_hidden_dim: int = 32`
    - `surprise_head_embed_dim: int = 4`
    - `surprise_trainable_head_embed: bool = False`
    - `surprise_uncertainty_bins: int = 64`
    - `surprise_include_margin: bool = False`
    - `use_qk_l2norm_in_kernel: bool = True`
- `fla/models/skda/modeling_skda.py`
  - 新模型：`SKDAModel` / `SKDAForCausalLM`
  - Block 内部使用 `SurpriseKimiDeltaAttention`
- `fla/models/skda/__init__.py`
  - 注册 `AutoConfig/AutoModel/AutoModelForCausalLM`

并将新类型导出到聚合入口：
- 修改：`fla/models/__init__.py`（导出 `SKDAConfig/SKDAModel/SKDAForCausalLM`）

### 4.3 layers 导出

- 修改：`fla/layers/__init__.py`
  - 导出 `SurpriseKimiDeltaAttention`

---

## 5. 并行性说明（为什么仍可并行）

SKDA 在“并行性”上的关键点是：
- `β_t` 与 `g_raw_t` 只依赖当前 token 的 `(k_t, v_t)`（通过 `v_hat=P(k_t)` 得到代理误差）
- 因此 `A_t`（由 `β_t` 与 `k_t` 构造、以及 gate 内核产生的 decay）不需要访问 `S_{t-1}` 的内容
- 更新仍然是 `S_t = A_t S_{t-1} + B_t`，保持 chunk 并行所需的仿射结构

如果未来要实现“真实误差驱动”（依赖 `S_{t-1}^T k_t`），需要重新设计并行算法或做分块近似；当前 SKDA 明确选择了 proxy 误差以保证 chunk 并行。

---

## 6. 如何使用（最小示例）

```python
from fla.models import SKDAConfig
from transformers import AutoModelForCausalLM

cfg = SKDAConfig(
    hidden_size=512,
    num_hidden_layers=6,
    num_heads=8,
    head_dim=64,
    max_position_embeddings=1024,
    vocab_size=32000,
    use_short_conv=False,
    surprise_gate_logit_normalizer=1.0,
    use_qk_l2norm_in_kernel=True,
)
model = AutoModelForCausalLM.from_config(cfg)
```

---

## 7. 变更清单（文件级）

新增：
- `fla/layers/skda.py`
- `fla/models/skda/__init__.py`
- `fla/models/skda/configuration_skda.py`
- `fla/models/skda/modeling_skda.py`
- `doc/skda_zh.md`（本文档）

修改：
- `fla/layers/__init__.py`：导出 `SurpriseKimiDeltaAttention`
- `fla/models/__init__.py`：导出 SKDA
