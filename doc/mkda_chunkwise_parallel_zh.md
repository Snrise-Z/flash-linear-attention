# MKDA（Multi‑Key Delta Attention）Chunkwise 并行版：实现说明（可直接落地）

本文档对应仓库里“真正可并行化的 chunkwise MKDA”实现（不是 micro-step 近似）：

- 算子：`fla/ops/mkda/chunkwise.py:mkda_chunkwise_parallel`
- 参考真值：`fla/ops/mkda/recurrent.py:mkda_recurrent`
- Attention 层：`fla/layers/mkda_chunkwise.py:ChunkwiseMultiKeyDeltaAttention`
- HF 选择开关：`MKDAConfig.mkda_impl="chunkwise"`（默认仍是 `"microstep"`）

该实现完全使用 PyTorch（核心为 `torch.linalg.solve_triangular` + `torch.cumsum`），**chunk 内没有 Python time-loop**，chunk 之间仍携带状态 `S`（与 KDA/DeltaRule 家族一致）。

---

## 1. 形状约定

- `q`: `[B, T, H, K]`
- `k`: `[B, T, H, R, K]`
- `v`: `[B, T, H, R, V]`
- `log_alpha`: `[B, T, H, K]`（每步 log 遗忘门，通常 ≤ 0，且不是 cumulative）
- `beta`: `[B, T, H, R]`
- 状态 `S_t`: `[B, H, K, V]`（实现里强制 float32）
- 输出 `o`: `[B, T, H, V]`

其中 `R` 是每个 token 的多 key/value 对数量（建议 2/4）。

---

## 2. 逐步递推真值（reference）

在每个时间步 `t`：

1) 遗忘（逐通道）：

$$
\bar S_{t-1} = \mathrm{Diag}(\alpha_t)\,S_{t-1},\quad \alpha_t=\exp(\log\alpha_t)
$$

2) 同时写入 `R` 个 key：

把 `K_t` 看成 `[K, R]`，`V_t` 看成 `[R, V]`，`β_t` 看成 `[R]`：

$$
E_t^\top = V_t^\top - K_t^\top \bar S_{t-1}
$$

$$
S_t = \bar S_{t-1} + K_t\,\mathrm{Diag}(\beta_t)\,E_t^\top
$$

3) 读出：

$$
o_t = S_t^\top (q_t\cdot \mathrm{scale})
$$

对应实现：`fla/ops/mkda/recurrent.py:mkda_recurrent`。

---

## 3. chunkwise 并行的关键：块下三角系统 + cumsum

对一个 chunk（长度 `C`）内部，把所有时间步的残差 `E_t` 堆成一个大矩阵：

- `E ∈ R^{(C·R)×V}`

可以构造一个对角为 `I_R` 的**块下三角矩阵** `A ∈ R^{(C·R)×(C·R)}`，满足：

$$
A\,E = \mathrm{RHS}
$$

其中 RHS 会包含：

- 当前 chunk 的 `V_t`
- chunk 开始状态 `S_0`
- 以及所有跨时间的耦合项（由 key 与遗忘门引入）

**一旦解出 `E`**，每步写入矩阵

$$
W_t = K_t\,\mathrm{Diag}(\beta_t)\,E_t^\top
$$

就全部得到，然后状态序列可用一次 `cumsum` 构造出来（在“衰减归一化空间”里）。

---

## 4. 遗忘门的处理：K_plus / K_minus

在 chunk 内定义 cumulative gate：

$$
g_{\mathrm{cum}}[t] = \sum_{j=0}^{t}\log\alpha_j,\quad
F_t = \exp(g_{\mathrm{cum}}[t]),\quad F_t^{-1}=\exp(-g_{\mathrm{cum}}[t])
$$

对 key 做逐维缩放：

$$
K_t^{+} = F_t \odot K_t,\quad
K_t^{-} = F_t^{-1} \odot K_t
$$

并把 `β_t` 视作对 `R` 方向的对角缩放：

$$
K_{t,\beta}^{-} = K_t^{-}\,\mathrm{Diag}(\beta_t)
$$

这样块矩阵的严格下三角块可写为：

$$
\text{block}(s,i) = (K_s^{+})^\top K_{i,\beta}^{-},\quad s>i
$$

对角块是 `I_R`，因此 `A` 是 unit-lower-triangular，可用 `solve_triangular(..., unitriangular=True)` 解。

对应实现：`fla/ops/mkda/chunkwise.py:mkda_chunkwise_parallel` 中的

- `K_plus, K_minus, K_minus_beta`
- `blocks = einsum(K_plus, K_minus_beta)`
- `Aflat = reshape(blocks + I)`

---

## 5. 状态恢复：在归一化空间里做 cumsum

先构造写入矩阵：

$$
W_t = K_t\,\mathrm{Diag}(\beta_t)\,E_t^\top
$$

在归一化空间累加：

$$
U_t = \sum_{i=0}^{t} F_i^{-1}\odot W_i
$$

恢复真实状态：

$$
S_t = F_t\odot (S_0 + U_t)
$$

对应代码：

- `U = cumsum(W * F_inv, dim=time)`
- `S = (S0 + U) * F`

---

## 6. 接入模型：如何切换到 chunkwise MKDA

当前仓库里 MKDA 仍保留 micro-step 实现（`mkda_impl="microstep"`），chunkwise 通过配置开关启用：

- `MKDAConfig.mkda_impl="chunkwise"`
- `MKDAConfig.chunk_size=64`（可调）

并在 `fla/models/mkda/modeling_mkda.py` 中按 `mkda_impl` 选择：

- micro-step：`fla/layers/mkda.py:MicrostepKimiDeltaAttention`
- chunkwise：`fla/layers/mkda_chunkwise.py:ChunkwiseMultiKeyDeltaAttention`

训练/评测脚本示例：

- 训练：`examples/train_mkda_chunkwise_wikitext103.py`
- 评测：`examples/eval_mkda_chunkwise_wikitext103.py`

