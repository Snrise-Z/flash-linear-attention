# MKDA（Micro-step KDA）当前 rank‑r 实现：极度详细、逐步推导与代码对齐

本文档目标是**极度详细**地描述当前仓库里 MKDA（micro-step 近似的 rank‑r KDA）是如何实现的、为什么这样实现、每一处张量 shape 如何变化、以及每一步数学公式与代码如何对应。

本文覆盖的“当前实现”指：

- **写入（state update）**：把每个 token 的 rank‑r 更新，近似为 token 内 `r` 次**串行 rank‑1 micro-step**（Approximation B）。
- **读出（readout）**：不再只读取最后一个 micro-step，而是对 token 内 `r` 个 micro-step 的输出做一个**可学习的线性混合**（`γ` mixing）。
- **Triton kernel 不改**：仍复用现有 rank‑1 `chunk_kda` / `fused_recurrent_kda` kernel；所有 rank‑r 行为由 PyTorch 侧“展开 + reshape + 混合”实现。

对应代码入口：

- micro-step 展开与 wrapper：`fla/ops/kda/microstep.py`
- MKDA attention 层：`fla/layers/mkda.py`（`MicrostepKimiDeltaAttention`）
- MKDA HF config/model：`fla/models/mkda/configuration_mkda.py`、`fla/models/mkda/modeling_mkda.py`
- 训练脚本：`examples/train_mkda_wikitext103.py`
- 评测脚本：`examples/eval_mkda_wikitext103.py`
- 单元测试：`tests/ops/test_kda_microstep.py`

---

## 0. 记号、维度与核心对象（先把“是什么”说清楚）

为避免下标爆炸，推导部分先以**单个 batch、单个 head**为主（省略 batch/head 维），但实现里所有维度都按 batch/head 并行。

### 0.1 维度

- 原始 token 序列长度：`T`
- micro-rank（rank‑r）：`r`（在代码中通常叫 `micro_rank`）
- key/query 维度：`K`（每个 head 的 key/query dim）
- value 维度：`V`（每个 head 的 value dim）
- 记忆（状态）矩阵：`S ∈ R^{K×V}`

### 0.2 rank‑1 KDA 的基本输入（单 head 视角）

在 rank‑1 KDA 里，每个时间步 `t`（token）对应：

- `q_t ∈ R^K`（query）
- `k_t ∈ R^K`（key）
- `v_t ∈ R^V`（value）
- `g_t ∈ R^K`（gate / decay，按维度逐元素作用在 `K` 上）
- `β_t ∈ R`（步长/强度，标量）

在当前实现中，MKDA 统一采用 **PyTorch 侧预先把 raw gate 转成 log-decay** 的路径：

- 先通过 `fused_kda_gate(g_raw, A_log, dt_bias)` 得到 `g`（log-decay）；
- micro-step 展开时，严格令 token 内 `a>0` 的 micro-step gate 为 **精确的 0**（表示“无衰减”）；
- 然后调用“接受 log-decay 的版本” kernel（`chunk_kda`/`fused_recurrent_kda` 以 `use_gate_in_kernel=False` 方式使用）。

这样在数学上 `g` 的非首 micro-step 真正等于 0，不再依赖“极小 raw 值穿过 kernel-side 非线性后的近似结果”。

本文推导会在数学上把 decay 写成 `exp(g)`（自然指数），代码为了数值与性能往往在内部用 `exp2` 等价实现，这不改变数学含义。

---

## 1. Rank‑1 KDA：从“一个 token”到“一个可并行的算子”

本节的目的：把 rank‑1 KDA 的每一步写成**无跳跃**的数学式，作为后续“把一个 token 展开成 r 个 micro-step”的基石。

### 1.1 定义逐维衰减矩阵

令

$$
D_t \triangleq \mathrm{Diag}(\exp(g_t)) \in \mathbb{R}^{K\times K}.
$$

注意：`g_t` 是长度为 `K` 的向量，`D_t` 是 `K×K` 的对角矩阵。

### 1.2 rank‑1 KDA 的一次更新（串行递推）

给定上一时刻状态 `S_{t-1} ∈ R^{K×V}`，rank‑1 KDA 的更新由两部分组成：

**(1) 衰减：**

$$
S_{t^-} = D_t S_{t-1}.
$$

这里 `t^-` 表示“在写入之前”的状态（先衰减）。

**(2) 计算 key 在状态上的投影：**

$$
p_t = k_t^\top S_{t^-} \in \mathbb{R}^{V}.
$$

解释：`k_t^\top` 是 `1×K`，`S_{t^-}` 是 `K×V`，所以结果是 `1×V`，即 `V` 维向量。

**(3) 计算残差：**

$$
e_t = v_t - p_t \in \mathbb{R}^{V}.
$$

**(4) 外积写入（delta-rule）：**

$$
S_{t^+} = S_{t^-} + \beta_t \, k_t e_t^\top.
$$

其中 `k_t e_t^\top` 是 `K×V` 的 rank‑1 外积矩阵。

**(5) 输出（读出）：**

$$
o_t = q_t^\top S_{t^+} \in \mathbb{R}^{V}.
$$

### 1.3 把更新写成“对旧状态的线性变换 + 注入项”（关键结构）

从 1.2(4) 的式子开始，代入 `e_t = v_t - k_t^\top S_{t^-}`：

$$
\begin{aligned}
S_{t^+}
&= S_{t^-} + \beta_t k_t (v_t - k_t^\top S_{t^-})^\top \\
&= S_{t^-} + \beta_t k_t v_t^\top - \beta_t k_t (k_t^\top S_{t^-}) \\
&= S_{t^-} + \beta_t k_t v_t^\top - \beta_t (k_t k_t^\top) S_{t^-} \\
&= (I - \beta_t k_t k_t^\top) S_{t^-} + \beta_t k_t v_t^\top \\
&= (I - \beta_t k_t k_t^\top) D_t S_{t-1} + \beta_t k_t v_t^\top.
\end{aligned}
$$

这一步非常重要，因为它把“更新”分成了：

- **线性部分**：`(I - β k k^T) D` 作用在旧状态 `S_{t-1}` 上；
- **注入部分**：`β k v^T` 与旧状态无关。

后面我们会把 rank‑r 写入变成 token 内多次这样的线性/注入叠加。

---

## 2. rank‑r（micro-rank）写入的目标：一个 token 内有 r 个方向

现在进入 rank‑r 的设定：对同一个 token `t`，我们希望有 `r` 个“写入方向”，直观上你可以把它看成：

- `u_{t,a} ∈ R^K`：第 `a` 个写入方向（key-like）
- `y_{t,a} ∈ R^V`：第 `a` 个写入内容（value-like）
- `β_{t,a} ∈ R`：第 `a` 个写入步长

其中 `a = 0,1,...,r-1`。

如果我们直接定义“精确 rank‑r KDA”，就需要在一个 token 内“同时”写入 `r` 个 rank‑1 外积，并且还要处理它们之间的耦合（高阶项）。这会要求改 kernel。

而当前实现选择的是一个**近似但可落地**的路线：

> 把“同一个 token 的 r 次写入”放到时间轴上串行执行：把一个 token 展开成 `r` 个 micro-step（每个 micro-step 是一个 rank‑1 KDA）。

这就是 Approximation B（串行 rank‑1 micro-step）。

---

## 3. micro-step 时间轴：把 token 展开成 r 个“子时刻”

### 3.1 micro-step 索引与展开后的长度

定义 micro-step 的全局索引 `s`：

$$
s = t\cdot r + a,\quad t\in\{0,\dots,T-1\},\ a\in\{0,\dots,r-1\}.
$$

因此：

- 原序列长度 `T`
- 展开后序列长度 `T' = T·r`

### 3.2 展开后的 rank‑1 输入（核心想法）

我们希望构造一条长度为 `T'` 的 rank‑1 序列：

- `k'_s ∈ R^K`
- `v'_s ∈ R^V`
- `β'_s ∈ R`
- `g'_s ∈ R^K`
- `q'_s ∈ R^K`

并把它们喂给已有的 rank‑1 KDA kernel（`chunk_kda` 或 `fused_recurrent_kda`），得到 micro-step 输出 `o'_s` 与最终状态。

### 3.3 当前实现的两条关键“micro 规则”（写入侧）

#### 规则 A：衰减只在每个 token 的第一个 micro-step 发生一次（严格为 0）

数学上我们想要：

$$
g'_{t,0} = g_t,\quad g'_{t,a>0}=0.
$$

原因（非常重要）：如果你让每个 micro-step 都应用同一个 `g_t`，那么一个 token 会被衰减 `r` 次，相当于把 decay 变成 `D_t^r`，状态会被过度抹除。

在代码里这个规则由 `fla/ops/kda/microstep.py:_expand_to_microsteps` 实现：

- 它先做 `g_rep = g.repeat_interleave(R, dim=1)` 得到 `[B, T*R, H, K]`
- 再构造 mask `is_first = (micro_rank == 0)`
- 然后对非 first micro-step（`a>0`）直接填 `0.0`（严格意义上的“无衰减”，因为 `exp(0)=1`）。

注意：当前 MKDA 的 chunk/fused_recurrent 两条路径都在 PyTorch 侧先用 `fused_kda_gate` 得到 log-decay `g`，因此这里填 0 是完全严格的。

#### 规则 B：token 内的 r 次写入是串行的 rank‑1 更新

把 token `t` 的 `u_{t,a}, y_{t,a}, β_{t,a}` 依序映射到 micro-step：

$$
k'_{t,a} = u_{t,a},\quad
v'_{t,a} = y_{t,a},\quad
\beta'_{t,a} = \beta_{t,a}.
$$

代码中：

- `k` 输入 shape 是 `[B,T,H,R,K]`（或最后两维可互换），通过 permute+reshape 变成：
  - `k_micro`：`[B, T*R, H, K]`
- `v` 同理得到 `v_micro`：`[B, T*R, H, V]`
- `beta` 得到 `beta_micro`：`[B, T*R, H]`

---

## 4. micro-step 串行写入：严格逐步写出 token 内状态演化

本节只看固定 token `t`，从上一个 token 结束的状态开始，写出 `r` 次 micro-step 的状态演化，完全不跳步。

为了清晰，我们把 token 内第 `a` 次 micro-step 的状态记为：

- `S_{t,a^-}`：第 `a` 次写入前（先衰减）
- `S_{t,a^+}`：第 `a` 次写入后

并且 `S_{t-1,r-1^+}` 是上一个 token 写完 r 次之后的最终状态。

### 4.1 定义 token 内的 rank‑1 线性算子与注入项

对每个 micro-step `(t,a)`，定义：

**线性算子**（作用在 `K×V` 矩阵上）：

$$
L_{t,a}(X) \triangleq (I - \beta_{t,a} u_{t,a} u_{t,a}^\top)\,X.
$$

**注入项**：

$$
R_{t,a} \triangleq \beta_{t,a} u_{t,a} y_{t,a}^\top \in \mathbb{R}^{K\times V}.
$$

### 4.2 micro-step a=0：先衰减，再写入一次

`a=0` 时衰减矩阵是 `D_t = Diag(exp(g_t))`：

$$
S_{t,0^-} = D_t S_{t-1,r-1^+}.
$$

更新后：

$$
\begin{aligned}
S_{t,0^+}
&= (I - \beta_{t,0} u_{t,0}u_{t,0}^\top)\,S_{t,0^-} + \beta_{t,0} u_{t,0} y_{t,0}^\top \\
&= L_{t,0}(S_{t,0^-}) + R_{t,0} \\
&= L_{t,0}(D_t S_{t-1,r-1^+}) + R_{t,0}.
\end{aligned}
$$

### 4.3 micro-step a=1：不再衰减，继续写入

`a=1` 时我们期望“无额外衰减”，因此 `D_{t,1}=I`（对应 `g'_{t,1}=0` 或 raw gate 的 fill 近似）。

因此：

$$
S_{t,1^-} = S_{t,0^+}.
$$

更新后：

$$
\begin{aligned}
S_{t,1^+}
&= L_{t,1}(S_{t,1^-}) + R_{t,1} \\
&= L_{t,1}(S_{t,0^+}) + R_{t,1} \\
&= L_{t,1}(L_{t,0}(D_t S_{t-1,r-1^+}) + R_{t,0}) + R_{t,1} \\
&= L_{t,1}L_{t,0}(D_t S_{t-1,r-1^+}) + L_{t,1}R_{t,0} + R_{t,1}.
\end{aligned}
$$

### 4.4 micro-step a=2：继续展开（显式写出 pattern）

同理：

$$
\begin{aligned}
S_{t,2^+}
&= L_{t,2}(S_{t,1^+}) + R_{t,2} \\
&= L_{t,2}\Big( L_{t,1}L_{t,0}(D_t S_{t-1,r-1^+}) + L_{t,1}R_{t,0} + R_{t,1}\Big) + R_{t,2} \\
&= L_{t,2}L_{t,1}L_{t,0}(D_t S_{t-1,r-1^+})
 + L_{t,2}L_{t,1}R_{t,0}
 + L_{t,2}R_{t,1}
 + R_{t,2}.
\end{aligned}
$$

### 4.5 一般形式（严格归纳得到）

对任意 `a`，有：

$$
S_{t,a^+}
= \Big(\prod_{b=0}^{a} L_{t,b}\Big)\,D_t S_{t-1,r-1^+}
 \sum_{j=0}^{a}\Big(\prod_{b=j+1}^{a} L_{t,b}\Big)\,R_{t,j}.
$$

这表达式说明：

- 旧状态经过 `D_t` 只衰减一次，然后经过 `a+1` 个 rank‑1 线性扰动的乘积；
- 每个注入项 `R_{t,j}` 在注入后，还会被后续的 `L_{t,j+1},...,L_{t,a}` 再“滤波”一遍。

这就是“串行 rank‑1 micro-step 近似 rank‑r”的精确数学含义。

---

## 5. 读出（readout）：从“只读最后一步”到“混合 r 个 micro-step”

当前实现包含一个关键增强：**混合 micro-step 输出**。同时，也保留了一个可选的“只读最后 micro-step”的模式用于对照实验。

配置项（在 `MKDAConfig` / 启动脚本中暴露）：

- `micro_readout_mode="mix"`（默认）：对 `r` 个 micro-step 输出做可学习混合；
- `micro_readout_mode="last"`：只使用最后一个 micro-step 的输出（与旧版读出行为一致）。

### 5.1 旧行为（历史版本）：只读最后 micro-step

旧行为相当于只取：

$$
o_t^{\mathrm{last}} = q_t^\top S_{t,r-1^+}.
$$

这对应“只看深度为 `r` 的那条路径”，token 内中间状态全部不参与输出。

### 5.2 新行为（当前实现）：读出整个 token 内轨迹并加权混合

当前实现让每个 micro-step 都读一次：

$$
o_{t,a} = q_t^\top S_{t,a^+},\quad a=0,\dots,r-1.
$$

然后用权重 `\gamma_a` 做线性混合：

$$
o_t^{\mathrm{mix}}
 = \sum_{a=0}^{r-1} \gamma_a\, o_{t,a}
 = q_t^\top\Big(\sum_{a=0}^{r-1}\gamma_a S_{t,a^+}\Big).
$$

在代码里，`γ` 是**每层、每 head**的一组可学习参数，通过 softmax 归一化：

$$
\gamma_{h,:} = \mathrm{softmax}(\ell_{h,:}),\quad \ell\in\mathbb{R}^{H\times r}.
$$

这样 `γ` 非负且和为 1，训练更稳定。

### 5.3 设计直觉（为什么混合读出值得做）

把 token 内 `r` 次写入想象成 `r` 个“微小残差块”串起来：

- 只读最后一步：你只能看到“经过 r 次残差后的最终特征”。
- 混合读出：你同时看到了中间每个深度的特征，再由 `γ` 学会怎么组合。

这等价于从“固定深度的单一路径”变成“多深度的 ensemble”：

- 浅层（小 `a`）更偏向“刚写入、还没被后续扰动”的信息；
- 深层（大 `a`）更偏向“经过更多 rank‑1 扰动后、更加组合化”的信息。

在实践中，当不同 `a` 学到不同类型的写入（局部 vs 长程、语义 vs 形状）时，混合读出能显著增加灵活性。

---

## 6. 代码如何实现（逐行对齐，按数据流走一遍）

本节用“从 `hidden_states` 到最终 `o`”的顺序，把 MKDA 的实现精确对齐到每个张量变换。

### 6.1 MKDA attention 层：`fla/layers/mkda.py:MicrostepKimiDeltaAttention`

#### 6.1.1 线性投影与 shape（核心：k/v/b 投影乘了 r）

构造：

- `q_proj: hidden_size -> key_dim`
- `k_proj: hidden_size -> key_dim * r`
- `v_proj: hidden_size -> value_dim * r`
- `b_proj: hidden_size -> num_heads * r`
- `f_proj: hidden_size -> key_dim`（gate 的 raw 输出，**不乘 r**）

其中：

- `key_dim = num_heads * head_k_dim`
- `value_dim = num_v_heads * head_v_dim`

#### 6.1.2 把“扁平维度”还原成 `[T,H,R,D]`

代码使用 `einops.rearrange`：

- `q`：`[..., (h d)] -> [..., h, d]` 得到 `[B,T,H,K]`
- `k`：`[..., (h r d)] -> [..., h, r, d]` 得到 `[B,T,H,R,K]`
- `v`：`[..., (h r d)] -> [..., h, r, d]` 得到 `[B,T,H,R,V]`
- `beta`：`[..., (h r)] -> [..., h, r]` 得到 `[B,T,H,R]`
- `g_raw`：`[..., (h d)] -> [..., h, d]` 得到 `[B,T,H,K]`

#### 6.1.3 调用 micro-step wrapper 获取 `o_micro`

当前实现强制请求“all micro-step outputs”：

- chunk 模式：先 `fused_kda_gate` 得到 log-decay，再 `chunk_kda_rank_r_microstep(..., micro_readout="all", use_gate_in_kernel=False)`
- fused_recurrent 模式：先 `fused_kda_gate` 得到 log-decay，再 `fused_recurrent_kda_rank_r_microstep(..., micro_readout="all")`

因此 `o_micro` 的 shape 是：

$$
o_{\mathrm{micro}} \in \mathbb{R}^{B\times T\times r\times H\times V}.
$$

#### 6.1.4 混合读出（这一步在 PyTorch 侧做）

参数：

- `micro_readout_logits ∈ R^{H×r}`
- `gamma = softmax(micro_readout_logits, dim=-1) ∈ R^{H×r}`

混合：

$$
o[b,t,h,:] = \sum_{a=0}^{r-1} \gamma[h,a]\;o_{\mathrm{micro}}[b,t,a,h,:].
$$

代码等价于：

```python
gamma = softmax(logits, dim=-1)  # [H,R]
o = (o_micro * gamma[None,None,:,None,:]).sum(dim=2)
```

（实现里 reshape 的广播维度写法略有差异，但数学等价。）

#### 6.1.5 初始化：尽量保持“接近只读最后一步”

`micro_readout_logits` 的初始化：

- 前 `r-1` 个位置是 `-8`
- 最后一个位置是 `0`

因此 `softmax` 后：

- `γ_{last}` 接近 1
- 其他 `γ_a` 很小

这使得模型一开始行为接近旧方案（只读最后 micro-step），再在训练中逐渐学会如何利用中间 micro-step。

### 6.2 micro-step wrapper：`fla/ops/kda/microstep.py`

核心函数 `_expand_to_microsteps` 接收：

- `q: [B,T,H,K]`
- `k: [B,T,H,R,K]`
- `v: [B,T,H,R,V]`
- `g: [B,T,H,K]`（raw 或 log-decay，取决于调用者）
- `beta: [B,T,H,R]`

并输出：

- `q_micro: [B,T*R,H,K]`（每个 token 的 `q_t` 复制 `R` 次）
- `k_micro: [B,T*R,H,K]`（把 rank 维挪到 time 维）
- `v_micro: [B,T*R,H,V]`
- `g_micro: [B,T*R,H,K]`（只在 token 的第一个 micro-step 保留 gate，其余填 0 或极小 raw）
- `beta_micro: [B,T*R,H]`

随后 wrapper 调用 rank‑1 kernel：

- `chunk_kda(q_micro,k_micro,v_micro,g_micro,beta_micro,...)`
- 或 `fused_recurrent_kda(...)`

最后根据 `micro_readout` 决定返回：

- `"last"`：`o_micro.reshape(B,T,R,H,V)[:, :, -1]` → `[B,T,H,V]`
- `"all"`：`o_micro.reshape(B,T,R,H,V)` → `[B,T,R,H,V]`

---

## 7. 训练/评测与统计导出：当前实现如何“确认生效”

### 7.1 训练脚本（HF Trainer）：`examples/train_mkda_wikitext103.py`

关键参数：

- `--micro_rank r`：决定 `k_proj/v_proj/b_proj` 的输出维度扩张，以及 micro-step 展开的 `R`
- `--micro_fill_g_raw`：gate-in-kernel 路径中，非 first micro-step 的 raw gate 填充值（默认 `-1e4`）
- `--print_microstep_stats`：导出 micro-step 统计到 `--output_dir`

统计导出文件：

- 预热导出：`mkda_microstep_stats_preflight.json`
- 训练中周期导出：`mkda_microstep_stats_stepXXXXXX.json`
  - 触发条件：`interesting_steps = {0,100,300,1000}` 或 `global_step % 1000 == 0`

统计内容包含：

- `k` 的 Gram 非对角均值（反映 rank 方向的耦合/正交性趋势）
- `beta` 的 min/max/mean/rms（反映步长大小是否过大，过大会使串行近似的高阶项影响变强）

### 7.2 评测脚本：`examples/eval_mkda_wikitext103.py`

`--print_microstep_stats` 会导出：

- `mkda_microstep_stats_eval_<split>.json`（写到 `--model` 目录）

---

## 8. 性能/显存直觉：为什么 micro_rank 变大但显存可能看起来不怎么涨

这是实现层面的重要直觉，帮助你解释“micro_rank 生效了但显存变化不明显”的现象：

1) **参数增量**主要来自 `k_proj/v_proj/b_proj` 这三处线性层按 `r` 扩张；但在完整 Transformer 里，embedding/MLP/optimizer state 可能才是显存主项，所以总显存不一定线性增长。

2) 当前实现不会把整段序列显式保存成 `[B,T*r,...]` 再贯穿所有层；micro-step 展开只发生在 attention 算子内部，并且输出最终会被 reshape/混合回 `[B,T,...]`。

3) 计算量（尤其是 KDA kernel 内的 work）通常更接近 `r` 倍，因此更显著的变化往往是 **step time** 而非显存。

---

## 9. 小结（当前 rank‑r 实现的“精确一句话”）

当前 MKDA 的 rank‑r 实现可以用一句严格的话概括：

> 对每个 token `t`，先按 `g_t` 对状态 `S` 做一次逐维衰减，然后用 `r` 个 `(u_{t,a}, y_{t,a}, β_{t,a})` 串行执行 `r` 次 rank‑1 KDA 更新；同时在每个 micro-step 上都用同一个 `q_t` 读出一个 `o_{t,a}`，最后用每层每 head 的 `γ`（softmax 参数化）对 `o_{t,0..r-1}` 做线性混合得到最终输出 `o_t`。整个过程通过把序列从 `T` 展开到 `T*r` 来复用 rank‑1 kernel，不修改 Triton kernel。
