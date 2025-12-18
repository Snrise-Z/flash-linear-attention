# FSNKDA（Fast/Slow Surprise-aware Normalized Kimi Delta Attention）中文说明与变更记录

本文档说明 **FSNKDA（Fast/Slow Surprise-aware Normalized Kimi Delta Attention）**：在 FSKDA（fast/slow + surprise-aware）基础上，对两条支路的 `β` 施加 NKDA 风格的 key-norm 归一化，使更新对 `||k||` 的尺度更不敏感。

---

## 1. 公式（单头写法）

先做归一化：

`β' = β / (||k||^2 + eps)`

然后仍是 KDA 形态（fast/slow 两条支路分别递推）：

`S_t^(x) = (I - β_t'^(x) k_t k_t^T) Diag(α_t^(x)) S_{t-1}^(x) + β_t'^(x) k_t v_t^T,  x∈{f,s}`

输出融合：

`o_t = λ_t o_t^(f) + (1-λ_t) o_t^(s)`

---

## 2. 注意事项：不要开启 kernel 内 `q/k` L2Norm

当 kernel 内对 `k` 做 L2Norm 时，`||k||≈1`，`β/(||k||^2+eps)` 会退化为近似常数缩放，FSNKDA 的归一化意义变弱。

因此 FSNKDA 强制：
- `use_beta_norm=True`
- `use_qk_l2norm_in_kernel=False`

---

## 3. 代码变更（文件级）

新增：
- `fla/layers/fsnkda.py`
- `fla/models/fsnkda/*`
- `examples/train_fsnkda_wikitext103.py`
- `examples/eval_fsnkda_wikitext103.py`
- `doc/fsnkda_zh.md`

---

## 4. 训练与评测（WikiText-103）

训练示例（单卡）：

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/train_fsnkda_wikitext103.py \
  --output_dir exp/fsnkda-wt103 \
  --seq_len 1024 \
  --max_steps 2000 \
  --tokenized_cache data/wt103_tok \
  --fp16 \
  --preflight_compile
```

评测示例：

```bash
python examples/eval_fsnkda_wikitext103.py \
  --model exp/fsnkda-wt103 \
  --split validation \
  --seq_len 1024 \
  --tokenized_cache data/wt103_tok \
  --dtype fp16 \
  --generate
```

