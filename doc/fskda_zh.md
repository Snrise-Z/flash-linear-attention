# FSKDA（Fast/Slow Surprise-aware Kimi Delta Attention）中文说明与变更记录

本文档说明 **FSKDA（Fast/Slow Surprise-aware Kimi Delta Attention）** 的设计与实现：在 fast/slow 双记忆（FKDA）基础上，引入 token-local 的 surprise-aware gate 来生成 `β_fast/β_slow/λ` 与 `g_raw_fast/g_raw_slow`，并保持 `chunk_kda` 并行可用。

---

## 1. 公式（单头写法）

fast/slow 两份记忆分别做 KDA 型更新：

`S_t^(x) = (I - β_t^(x) k_t k_t^T) Diag(α_t^(x)) S_{t-1}^(x) + β_t^(x) k_t v_t^T,  x∈{f,s}`

输出融合：

`o_t = λ_t o_t^(f) + (1-λ_t) o_t^(s),  λ_t∈(0,1)`

其中：
- `β_t^(f), β_t^(s), λ_t` 都是 **每 token/head 的标量**；
- `α_t^(x)` 仍由 KDA gate 内核根据 `g_raw^(x), A_log, dt_bias` 计算（`use_gate_in_kernel=True`）。

---

## 2. Surprise-aware gate（token-local）

FSKDA 的 gate 只依赖 token-local 可并行计算的量（不依赖 `S_{t-1}`），因此不会破坏递推的仿射线性。

实现中复用了 SKDA 风格的“少量强相关统计量 + 轻量结构信息”的思路，输出：
- `β_fast, β_slow`
- `λ`
- `amp_fast, amp_slow`，并构造 `g_raw^(x) = amp_x * normalize(k)` 供 KDA gate 计算 `α`。

---

## 3. 并行性说明

FSKDA 仍满足 chunk 并行前提：
- 对每个 token，所有 gate 输入只依赖 `(k,v,hidden_states,position,head_id)` 的局部投影/统计；
- 不依赖 `S_{t-1}`；
- fast/slow 两条支路分别调用 `chunk_kda`，整体仍可 chunk-parallel。

---

## 4. Ablation 开关

- `fix_lambda`：把融合系数 `λ` 固定为常数（例如 0.5）。
- `share_decay_gate`：fast/slow 共享同一条 decay gate（`α^(f)=α^(s)`）。

---

## 5. 代码变更（文件级）

相关实现：
- `fla/layers/fskda.py`
- `fla/models/fskda/*`
- `examples/train_fskda_wikitext103.py`
- `examples/eval_fskda_wikitext103.py`
- `doc/fskda_zh.md`

---

## 6. 训练与评测（WikiText-103）

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

