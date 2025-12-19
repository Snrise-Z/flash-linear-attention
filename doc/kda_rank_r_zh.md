# KDA 的 rank-1 修正推广到 rank-r：完整公式推导 + FLA KDA kernel 改造点

本文档目标：
- 用**不跳步骤**的数学推导，把当前 FLA 的 KDA（rank-1 修正）写成“chunk 内解一个下三角线性系统”的形式；
- 在此基础上，给出一个**严格的 rank-r 修正定义**（仍保持对状态 `S` 的仿射线性，从而仍可 chunk-parallel）；
- 把 rank-r 的推导逐项对齐到现有代码（`chunk_kda` / `chunk_kda_fwd_intra` / `chunk_gated_delta_rule_*` / `chunk_gla_*`）；
- 最后列出：为了做“精确 rank-r”，Triton kernel 需要改哪些地方；以及几个“合理近似、尽量复用现有 kernel”的落地路线。

备注：本文用自然指数 `exp(·)` 做推导。代码里为了性能大量使用 `exp2(·)`，通过 `RCP_LN2 = 1/ln(2)` 把 log-space gate 转换到 `log2` 空间：

- 若 `x` 是自然对数域的 log gate，则 `exp(x) = exp2(x / ln 2)`。

---

## 0. 记号、shape 与代码变量

下文主要以 **单个 batch、单个 head** 来写公式（避免下标爆炸），但所有公式都可逐 batch/head 并行计算。

- 序列长度：`T`
- chunk 长度：`BT=64`（`chunk_kda` 默认 `chunk_size=64`）
- key/query 维度：`K`（限制 `K<=256`，见 `fla/ops/kda/chunk_intra.py` / `fla/ops/common/chunk_delta_h.py`）
- value 维度：`V`
- rank：`r`（本文的新增维度）

### 0.1 现有 rank-1 KDA 输入（与代码一致）

对每个时间步 `t`：
- `q_t ∈ R^K`（`chunk_kda` 内部会乘 `scale=1/sqrt(K)`）
- `k_t ∈ R^K`
- `v_t ∈ R^V`
- `g_t ∈ R^K`（log gate；`chunk_kda` docstring：`g` is in log space）
- `β_t ∈ R`（标量；代码参数名 `beta`，shape `[B,T,H]`）

状态（记忆矩阵）：
- `S_t ∈ R^{K×V}`

### 0.2 现有 kernel 里的“中间量”对应

rank-1 路径里这些名字很关键：

- `g_cum`：chunk 内对 `g` 做 prefix-cumsum 后的 gate（代码：`g = chunk_local_cumsum(g, ...)`）
- `Akk`：`chunk_kda_fwd_intra` 输出的下三角矩阵（语义是 `A=(I+L)^{-1}`，见第 2 节）
- `Aqk`：`chunk_kda_fwd_intra` 输出的 query-key 相关系数（用于输出）
- `w,u`：`recompute_w_u_fwd` 根据 `Akk` 计算的两个张量（用于构造创新量）
- `kg`：把 key 乘上“到 chunk 末尾的衰减因子”的版本（用于更新 chunk 末状态）
- `h`：`chunk_gated_delta_rule_fwd_h` 输出的每个 chunk 的中间状态（供 `chunk_gla_fwd_o_gk` 输出用）
- `v_new`：`chunk_gated_delta_rule_fwd_h` 输出的新 value（本质是“创新量/残差”，见第 2 节）

相关代码入口：
- 串行参考：`fla/ops/kda/naive.py:naive_recurrent_kda`
- chunk 参考：`fla/ops/kda/naive.py:naive_chunk_kda`
- Triton 主入口：`fla/ops/kda/chunk.py:chunk_kda`

---

## 1. Rank-1 KDA 串行递推：从代码到数学公式（不跳步）

直接对齐 `fla/ops/kda/naive.py:naive_recurrent_kda`。

### 1.1 定义

把 `g_t` 写成对 `K` 维逐元素的衰减：

$$
D_t = \operatorname{diag}(\exp(g_t)) \in R^{K\times K}
$$

### 1.2 串行递推（与 naive_recurrent_kda 完全一致）

代码等价于：

1) 旧状态衰减：

$$
\hat S_t = D_t S_{t-1}
$$

2) rank-1 修正（outer product）：

$$
S_t = \hat S_t + \beta_t\, k_t\,(v_t^\top - k_t^\top \hat S_t)
$$

3) 输出：

$$
o_t = q_t^\top S_t
$$

### 1.3 写成对 `S_{t-1}` 的仿射线性

展开第 1.2 的更新（不省略）：

$$
\begin{aligned}
S_t
&= \hat S_t + \beta_t k_t v_t^\top - \beta_t k_t (k_t^\top \hat S_t) \\
&= \hat S_t + \beta_t k_t v_t^\top - \beta_t (k_t k_t^\top)\hat S_t \\
&= (I - \beta_t k_t k_t^\top)\,\hat S_t + \beta_t k_t v_t^\top \\
&= (I - \beta_t k_t k_t^\top)\,D_t\, S_{t-1} + \beta_t k_t v_t^\top
\end{aligned}
$$

要点：递推对 `S_{t-1}` 仍是线性的（仿射线性）。这也是 `chunk_kda` 能并行化的根本前提。

---

## 2. Rank-1 KDA 的 chunk 化推导：为什么会出现 `Akk/Aqk/w/u/v_new`

本节把第 1 节的串行递推，推导成“chunk 内解下三角系统”的形式，并对应到 `chunk_kda` 的实现。

### 2.1 chunk 内累计 gate 与“相对衰减”

一个 chunk 长度 `BT=64`，chunk 内 token 索引用 `i=0..BT-1`。

定义 chunk 内累计 gate（对 `K` 维逐元素）：

$$
G_i = \sum_{t=0}^{i} g_t \in R^K
$$

那么从 token `j` 的更新传递到 token `i`（`i\ge j`）之前，经历 `j+1..i` 的衰减，等价于：

$$
\operatorname{Decay}(i\leftarrow j)=\operatorname{diag}(\exp(G_i - G_j))
$$

对齐到代码：
- `chunk_kda` 在进入 `chunk_kda_fwd_intra` 前执行 `chunk_local_cumsum(g, scale=RCP_LN2)`；
- Triton kernel 用 `exp2` 实现上述 `exp(G_i-G_j)`。

### 2.2 把 `β` 吸收到 value 侧：定义创新量（innovation）

rank-1 串行更新里，每步新增项是：

$$
\Delta S_i = \beta_i k_i (v_i^\top - k_i^\top \hat S_i)
$$

定义创新量（把 `β_i` 吸进来）：

$$
t_i^\top \triangleq \beta_i (v_i^\top - k_i^\top \hat S_i) \in R^{1\times V}
$$

则更新变为：

$$
\Delta S_i = k_i t_i^\top
$$

### 2.3 写出 chunk 内 update 前状态 \hat S_i

令 `S_0` 是 chunk 起点（进入该 chunk 前）的状态。

第 `i` 步 update 之前的状态（经历了 `0..i` 的衰减，且加上 `0..i-1` 的更新并衰减到 `i`）为：

$$
\hat S_i
=
\operatorname{diag}(\exp(G_i))S_0
\;+\;
\sum_{j<i}
\operatorname{diag}(\exp(G_i-G_j))\,k_j t_j^\top
$$

左乘 `k_i^\top`：

$$
\begin{aligned}
k_i^\top \hat S_i
&=
(\exp(G_i)\odot k_i)^\top S_0
\;+\;
\sum_{j<i}
\langle k_i,\;\exp(G_i-G_j)\odot k_j\rangle\; t_j^\top
\end{aligned}
$$

### 2.4 得到严格下三角系统：`(I+L) t = RHS`

由 2.2：

$$
t_i^\top = \beta_i v_i^\top - \beta_i k_i^\top \hat S_i
$$

代入 2.3 并整理（不省略）：

$$
t_i^\top
\;+\;
\sum_{j<i}
\underbrace{\Big(\beta_i\langle k_i,\exp(G_i-G_j)\odot k_j\rangle\Big)}_{L_{i,j}}
\;t_j^\top

=
\beta_i v_i^\top
-
\beta_i(\exp(G_i)\odot k_i)^\top S_0
$$

定义严格下三角矩阵 `L∈R^{BT×BT}`：

$$
L_{i,j}=
\begin{cases}
\beta_i\langle k_i,\exp(G_i-G_j)\odot k_j\rangle, & i>j \\
0, & i\le j
\end{cases}
$$

令 `t∈R^{BT×V}`（第 `i` 行是 `t_i^\top`），并把右端项按行堆叠成 `RHS∈R^{BT×V}`，则：

$$
(I+L)\;t = RHS
$$

### 2.5 引入 `A=(I+L)^{-1}` 与 `w,u`：`t = u - w S0`

因为 `L` 严格下三角，`I+L` 可逆且逆仍下三角。

令：

$$
A \triangleq (I+L)^{-1}
$$

令 `\tilde k_i = \exp(G_i)\odot k_i`，把所有 `\tilde k_i` 按行堆叠得 `\tilde K ∈ R^{BT×K}`。

右端项可写为：

$$
RHS = (\beta\odot V) - (\beta\odot \tilde K)S_0
$$

于是：

$$
\begin{aligned}
t &= A(\beta\odot V) - A(\beta\odot \tilde K)S_0 \\
u &\triangleq A(\beta\odot V) \in R^{BT\times V} \\
w &\triangleq A(\beta\odot \tilde K) \in R^{BT\times K} \\
\Rightarrow\quad t &= u - wS_0
\end{aligned}
$$

对齐到代码：
- `chunk_kda_fwd_intra` 构造并求逆得到 `A`，存到 `Akk`（语义就是这里的 `A`）。
- `recompute_w_u_fwd` 在 `fla/ops/kda/wy_fast.py:recompute_w_u_fwd_kernel` 内做：
  - `u = Akk @ (beta * v)`
  - `w = Akk @ (beta * exp(G) * k)`（代码里用 `exp2`）
- `chunk_gated_delta_rule_fwd_h` 在 kernel 内做 `v_new = u - w @ state`，这里的 `v_new` 对应 `t`。

### 2.6 chunk 末状态：为什么需要 `kg`

chunk 末累计 gate 为 `G_last`。

chunk 末状态：

$$
S_{\text{end}}
=
\operatorname{diag}(\exp(G_{\text{last}}))S_0
\;+\;
\sum_{i}
\operatorname{diag}(\exp(G_{\text{last}}-G_i))\,k_i\,t_i^\top
$$

定义衰减到 chunk 末的 key：

$$
k_i^{(\text{to end})} \triangleq \exp(G_{\text{last}}-G_i)\odot k_i
$$

对齐到代码：`recompute_w_u_fwd_kernel` 里 `STORE_KG` 计算的 `kg` 就是 `k^{(to end)}`。

### 2.7 输出：为什么需要 `Aqk`，以及为何能复用 `chunk_gla_fwd_o_gk`

输出写成：

$$
\begin{aligned}
o_i
&=
(\exp(G_i)\odot q_i)^\top S_0
\;+\;
\sum_{j\le i}
\underbrace{\langle q_i,\exp(G_i-G_j)\odot k_j\rangle}_{Aqk_{i,j}}
\; t_j^\top
\end{aligned}
$$

其中：

$$
Aqk_{i,j} = \langle q_i,\exp(G_i-G_j)\odot k_j\rangle \cdot \text{scale}
$$

这就是 `chunk_kda_fwd_intra` 里算的 `Aqk`（标量下三角矩阵），然后喂给 `chunk_gla_fwd_o_gk` 完成输出。

---

## 3. Rank-r 修正：严格定义（保持对 S 的仿射线性）

### 3.1 Rank-r 的输入与 shape

对每个时间步 `t`：
- `q_t ∈ R^K`（不变）
- `g_t ∈ R^K`（先按“rank 共享 gate”设计；若要 rank 独立 gate，见第 6 节）

新增 rank 维：
- `K_t ∈ R^{K×r}`：`r` 个 key 向量组成的矩阵
- `V_t ∈ R^{V×r}`：`r` 个 value 向量组成的矩阵
- `B_t ∈ R^{r×r}`：更新强度矩阵（常见选 `diag(β_t)`，`β_t∈R^r`）

状态仍是 `S_t ∈ R^{K×V}`（共享，不拆成 r 份）。

### 3.2 Rank-r 串行递推（直接推广）

令 `D_t = diag(exp(g_t))`，先衰减：

$$
\hat S_t = D_t S_{t-1}
$$

定义：

$$
T_t^\top \triangleq B_t (V_t^\top - K_t^\top \hat S_t)\in R^{r\times V}
$$

更新：

$$
S_t = \hat S_t + K_t T_t^\top
$$

合并写成：

$$
S_t = D_t S_{t-1} + K_t B_t (V_t^\top - K_t^\top D_t S_{t-1})
$$

### 3.3 仍然仿射线性（关键性质）

展开：

$$
S_t
= (I - K_t B_t K_t^\top) D_t S_{t-1} + K_t B_t V_t^\top
$$

因为 `K_t B_t K_t^\top` 秩至多为 `r`，所以仍是“对角+低秩”的线性递推结构。

---

## 4. Rank-r 的 chunk 化推导（严格、不跳步）

### 4.1 累计 gate（同 2.1）

$$
G_i = \sum_{t=0}^{i} g_t,\qquad
\operatorname{Decay}(i\leftarrow j)=\operatorname{diag}(\exp(G_i-G_j))
$$

### 4.2 创新量 `T_i`（把 `B_i` 吸收到 value 侧）

$$
T_i^\top \triangleq B_i(V_i^\top - K_i^\top \hat S_i)\in R^{r\times V}
$$

更新是 `ΔS_i = K_i T_i^\top`。

### 4.3 chunk 内 update 前状态 \hat S_i

$$
\hat S_i
=
\operatorname{diag}(\exp(G_i))S_0
\;+\;
\sum_{j<i}
\operatorname{diag}(\exp(G_i-G_j))\,K_j T_j^\top
$$

### 4.4 左乘 `K_i^\top`（得到 r×V）

$$
K_i^\top \hat S_i
=
K_i^\top \operatorname{diag}(\exp(G_i))S_0
\;+\;
\sum_{j<i}
K_i^\top \operatorname{diag}(\exp(G_i-G_j))K_j \; T_j^\top
$$

### 4.5 得到块下三角系统

代回定义并整理：

$$
T_i^\top
\;+\;
\sum_{j<i}
\underbrace{\Big(B_i K_i^\top \operatorname{diag}(\exp(G_i-G_j))K_j\Big)}_{L_{i,j}\in R^{r\times r}}
\;T_j^\top
=
B_i V_i^\top
-
B_i K_i^\top \operatorname{diag}(\exp(G_i))S_0
$$

把所有 `T_i^\top` 纵向堆叠成 `(BT·r)×V` 的矩阵 `T`，得到：

$$
(I+L)\;T = RHS
$$

其中 `I+L` 是 `BT×BT` 的块矩阵，每块 `r×r`。

### 4.6 与 rank-1 同构的解形式：`T = U - W S0`

令：

$$
A\triangleq(I+L)^{-1}
$$

定义 gated key：

$$
\tilde K_i \triangleq \operatorname{diag}(\exp(G_i))K_i\in R^{K\times r}
$$

右端项：

$$
RHS_i^\top = B_i V_i^\top - B_i \tilde K_i^\top S_0
$$

从而：

$$
T = A\,(BV^\top) - A\,(B\tilde K^\top)S_0
$$

定义：

$$
U\triangleq A\,(BV^\top),\qquad
W\triangleq A\,(B\tilde K^\top)
$$

逐 token 写就是：

$$
T_i^\top = U_i^\top - W_i^\top S_0
$$

### 4.7 chunk 末状态与输出

chunk 末状态：

$$
S_{\text{end}}
=
\operatorname{diag}(\exp(G_{\text{last}}))S_0
\;+\;
\sum_i
\operatorname{diag}(\exp(G_{\text{last}}-G_i))\,K_i\,T_i^\top
$$

输出：

$$
o_i
=
q_i^\top \operatorname{diag}(\exp(G_i))S_0
\;+\;
\sum_{j\le i}
q_i^\top \operatorname{diag}(\exp(G_i-G_j))K_j \; T_j^\top
$$

注意第二项中 `q_i^\top diag(...) K_j` 是 `1×r`，因此 `Aqk` 不再是标量。

---

## 5. 对齐到现有 FLA KDA kernel：rank-1 各量是什么、rank-r 应扩成什么

### 5.1 `g`：从 log gate 到 chunk 累计 gate `G`

文件：`fla/ops/kda/chunk.py`

- 若 `use_gate_in_kernel=True`：先 `kda_gate_fwd`
- 再 `g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)`

数学上这一步把“逐 token 的 log gate”变成“chunk 内累计 gate `G`”（只是换到 log2 空间）。

### 5.2 `Akk`：当前实现存的是 `A=(I+L)^{-1}`（`beta` 在后面乘）

`chunk_kda_fwd_intra` 输出的 `Akk` 对应推导里的 `A=(I+L)^{-1}`。

`beta` 的乘法在 `recompute_w_u_fwd_kernel` 内部做（对 `v` 与 `k` 的行缩放）。

rank-r 时应扩展为块矩阵 `A_{i,j}∈R^{r×r}`。存储布局建议优先考虑把 `(BT,r)` flatten 成一个轴，避免 6 维张量导致的 stride/地址计算复杂。

### 5.3 `recompute_w_u_fwd`：实现的就是 `U=A(BV)`, `W=A(B\tilde K)`

对齐 `fla/ops/kda/wy_fast.py:recompute_w_u_fwd_kernel`：

- `u = A @ (beta*v)`
- `w = A @ (beta*exp(G)*k)`
- 可选输出 `kg`

rank-r 时就是把 `beta` 换成 `B_i`（对角或满矩阵）并把 `k,v` 变成矩阵/多向量。

### 5.4 `kg`：key 衰减到 chunk 末

rank-1：

$$
kg_i = \exp(G_{last}-G_i)\odot k_i
$$

rank-r：

$$
KG_i = \operatorname{diag}(\exp(G_{last}-G_i))K_i
$$

### 5.5 输出侧 `Aqk`：从标量到长度 r 的向量

rank-1：

$$
Aqk_{i,j}=\langle q_i,\exp(G_i-G_j)\odot k_j\rangle
$$

rank-r：

$$
Aqk_{i,j} \triangleq q_i^\top \operatorname{diag}(\exp(G_i-G_j))K_j \in R^{1\times r}
$$

这会影响到输出 kernel 的数据布局/计算方式（下一节详细列）。

---

## 6. 做“精确 rank-r”时，FLA KDA kernel 需要改哪些点（按文件列）

下面默认你要实现的是第 3/4 节的严格 rank-r（块系统、rank 间全耦合），而不是近似。

### 6.1 Python 入口与 shape 检查

文件：`fla/ops/kda/chunk.py`

建议新增新算子（不要直接改 `chunk_kda`，保持兼容）：

- `chunk_kda_rankr(q, K, V, g, B, ...)`

建议 shape（对齐推导）：

- `q: [B, T, H, K]`
- `K: [B, T, H, K, r]`
- `V: [B, T, H, V, r]`（或 `[B,T,H,r,V]`）
- `B: [B, T, H, r]`（对角）或 `[B, T, H, r, r]`（满矩阵）
- `g: [B, T, H, K]`（rank 共享；rank 独立 gate 则为 `[B,T,H,K,r]`）
- `initial_state: [N, H, K, V]`（共享 state，rank-r 的核心就是“不拆 state”）

### 6.2 构造并求逆块下三角：`chunk_kda_fwd_intra_*`

文件：
- `fla/ops/kda/chunk_intra_token_parallel.py`
- `fla/ops/kda/chunk_intra.py`

需要把 `L` 的标量条目扩展成 `r×r` 块：

$$
L_{i,j} = B_i\;K_i^\top \operatorname{diag}(\exp(G_i-G_j))K_j \in R^{r\times r}
$$

工程落地常见两种方式：

方式 A：块矩阵显式实现（建议先做对角 B）
- 保持 token 维 `BT=64`，rank `r` 小（2/4/8）；
- kernel 内把每个块当 `r×r` 小矩阵做累加/乘法；
- 产出 `A=(I+L)^{-1}` 的同样块结构。

方式 B：flatten 成大矩阵（实现直观但开销大）
- 把未知量按 `(i,rank)` 展开成长度 `BT*r`；
- `A` 视作 `(BT*r)×(BT*r)` 大矩阵；
- 当前 16×16 子块求逆会变成 `(16*r)×(16*r)` 子块，`r=4` 时就是 64×64，寄存器压力很大。

### 6.3 `recompute_w_u_fwd` / `prepare_wy_repr_bwd`

文件：`fla/ops/kda/wy_fast.py`

rank-r 需要支持：

- `U = A @ (B V)`（每 token：`B_i V_i^\top` 是 `r×V`）
- `W = A @ (B \tilde K^\top)`（每 token：`B_i \tilde K_i^\top` 是 `r×K`）
- 以及输出 `KG`（衰减到 chunk 末的 `K×r`）

反向：
- 需要把当前 `prepare_wy_repr_bwd_kernel` 的标量推导扩成块矩阵（至少覆盖 `dK,dV,dB,dA,dg`）。

### 6.4 delta-rule chunk kernel：rank-1 外积 → rank-r 小 GEMM

文件：`fla/ops/common/chunk_delta_h.py`

rank-1 的 chunk 内更新是：

$$
S \;+=\; \sum_i kg_i\;t_i^\top
$$

rank-r 变成：

$$
S \;+=\; \sum_i KG_i\;T_i^\top
$$

因此 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 至少要支持：

- `k` 从 `[T,K]` 变成 `[T,K,r]`
- `v`/`v_new` 从 `[T,V]` 变成 `[T,r,V]`
- `w` 从 `[T,K]` 变成 `[T,r,K]`（用于计算 `T = U - W @ S0`）

### 6.5 输出：Aqk 从标量变成向量后的处理

rank-r 输出贡献是：

$$
\sum_{j\le i}
\big(q_i^\top \operatorname{diag}(\exp(G_i-G_j))K_j\big)\;T_j^\top
$$

你有两条路线：

1) 改 `chunk_gla_*` 支持 rank 维  
这会影响到 `fla/ops/gla/chunk.py` 的 forward/backward（牵一发动全身）。

2) 不改 `chunk_gla_*`，把 rank 展开成 head 维做 r 次贡献，然后 reduce-sum（更工程友好）
- 把 `T_j^\top` 的 `r` 行当作 `HV=H*r` 个“子 head 的 value”（每个子 head 的 `V` 维向量）
- 把 `Aqk` 的 `r` 分量当作对应子 head 的标量注意力系数
- 调 `chunk_gla_fwd_o_gk` 得到 `[B,T,H*r,V]`，再 reshape 并沿 `r` reduce-sum 得到 `[B,T,H,V]`
- baseline（来自旧状态/`h`）要保证只加一次。更稳妥的组合方式是：
  - 单独算 baseline：`baseline_i = (exp(G_i)⊙q_i)^\top S0`（可用 `einsum`）
  - 让 `chunk_gla` 只算 `A@T`：把 `h` 置 0，再加 baseline

---

## 7. 合理近似（尽量复用现有 rank-1 kernel）——两条可快速验证的路线

精确 rank-r 会牵涉到“块下三角求逆 + delta-rule 更新 + 输出”三块大改造。若你想先快速验证 idea，可考虑以下近似。

### 7.1 近似 A：忽略 rank 间耦合（把块 Gram 近似成对角）

rank-r 的关键耦合项是：

$$
K_i^\top \operatorname{diag}(\exp(G_i-G_j))K_j \in R^{r\times r}
$$

近似假设：

$$
K_i^\top \operatorname{diag}(\exp(G_i-G_j))K_j \approx \operatorname{diag}\Big(\operatorname{diag}(K_i^\top \operatorname{diag}(\exp(G_i-G_j))K_j)\Big)
$$

只保留同 rank 的内积，丢掉 cross-rank 内积。

效果：
- 块系统退化成 `r` 个互不耦合的标量系统；
- 每个 rank 分量都像一个独立的 rank-1 KDA；
- 最终 state / output 通过对 `r` 个 rank 的贡献求和得到。

工程落地（尽量少改 kernel）的一个做法：

1) 为每个 rank `a=1..r` 构造一组 rank-1 输入：`k^{(a)}, v^{(a)}, beta^{(a)}`（共享同一个 `q,g,S0`）
2) 分别运行 `chunk_kda(q, k^{(a)}, v^{(a)}, g, beta^{(a)}, initial_state=S0, ...)` 得到 `(o^{(a)}, S_end^{(a)})`
3) 由于每次运行都会包含同一个“旧状态 baseline”，合并要去重：
   - baseline：
     $$
     o^{(base)}_i = (exp(G_i)\odot q_i)^\top S_0
     $$
     可用 `einsum` 直接算（不改 kernel）。
   - 合并输出：
     $$
     o_i \approx o^{(base)}_i + \sum_{a=1}^r\Big(o^{(a)}_i - o^{(base)}_i\Big)
     $$
   - 合并最终状态：
     $$
     S_{end} \approx D_{end}S_0 + \sum_{a=1}^r\Big(S_{end}^{(a)} - D_{end}S_0\Big)
     $$

这个近似的“合理性来源”：
- 若训练时鼓励 `K_t` 的各列在 `diag(exp(G))` 加权下近似正交，则 cross-r 项很小；
- 此时块系统近似对角成立。

代价：算子调用次数乘以 `r`（`r=2/4` 时研究验证可能可接受）。

### 7.2 近似 B：把 rank-r 当作 token 内的 r 个 micro-step（串行 rank-1）

把每个 token 的 rank-r 更新拆成 r 次 rank-1 更新并串行应用：

$$
\prod_{a=1}^r (I-\beta_{t,a}k_{t,a}k_{t,a}^\top)D_t
$$

若 `β` 较小，一阶近似：

$$
\prod_{a=1}^r (I-\beta_{t,a}k_{t,a}k_{t,a}^\top)
\approx
I-\sum_{a=1}^r \beta_{t,a}k_{t,a}k_{t,a}^\top
$$

从而接近第 3 节的“同时 rank-r”定义。

工程上相当于把序列长度从 `T` 变成 `T*r`，并在 micro-step 间把 gate 设为 0（衰减只在 token 间发生）。

优点：最大化复用现有 rank-1 kernel；缺点：模型定义改变且吞吐下降。

本仓库代码实现（近似 B）：
- 算子封装：`fla/ops/kda/microstep.py`  
  - `chunk_kda_rank_r_microstep(...)`：训练/长序列（底层调用 `chunk_kda`）  
  - `fused_recurrent_kda_rank_r_microstep(...)`：短序列/推理（底层调用 `fused_recurrent_kda`；要求传入 **log-decay g**）
- 单测：`tests/ops/test_kda_microstep.py`

---

## 8. 推荐实现路线（从易到难）

1) 先实现 naive rank-r 参考（PyTorch）  
在 `fla/ops/kda/naive.py` 新增 `naive_recurrent_kda_rankr/naive_chunk_kda_rankr`，用第 3/4 节公式直接写，作为正确性基准。

2) 先落地近似 A（r 次 rank-1 + baseline 去重）  
先在 Python 层把 idea 跑起来，观察收敛/稳定性，再决定值不值得大改 kernel。

3) 再做精确 rank-r 的 intra(Akk/Aqk) + w/u（最硬的一步）  
先只做 forward 并在小尺寸上对齐 naive；再补 backward。

4) 最后做 delta-rule 与输出侧的 rank-r 支持  
输出侧建议先用“rank 展开成 head + reduce-sum”复用 `chunk_gla_*`，避免改 GLA 通用 kernel。

---

## 9. 你接下来如果要我继续做什么

我可以继续帮你做两类工作（你选其一）：

- A) 在 `fla/ops/kda/naive.py` 实现严格 `naive_recurrent_kda_rankr/naive_chunk_kda_rankr`（含单测），把第 3/4 节推导落成可跑 reference；
- B) 基于“近似 A（r 次 rank-1 + baseline 去重）”，在 `fla/ops/kda` 上新增一个 Python 封装算子（不改 Triton kernel），让你能快速跑实验对比。
