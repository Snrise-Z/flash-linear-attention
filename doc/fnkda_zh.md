# FNKDA（Fast/Slow Normalized Kimi Delta Attention）中文说明与变更记录

本文档说明本仓库中引入 **FNKDA（Fast/Slow Normalized Kimi Delta Attention）** 的全部相关变更：在 FKDA（fast/slow 双记忆）的基础上，对两条支路的 `β` 应用 NKDA 风格的 key-norm 归一化，使更新对 `||k||` 的尺度更不敏感。

---

## 1. 公式（单头写法）

先由 FKDA 产生 `β_fast, β_slow`，再做归一化：

`β' = β / (||k||^2 + eps)`

然后仍是 KDA 形态：

`S_t^(x) = (I - β_t'^(x) k_t k_t^T) Diag(α_t^(x)) S_{t-1}^(x) + β_t'^(x) k_t v_t^T`

输出融合保持不变：

`o_t = λ_t o_t^(f) + (1-λ_t) o_t^(s)`

---

## 2. 注意事项：不要开启 kernel 内 `q/k` L2Norm

当 `use_qk_l2norm_in_kernel=True` 时，`||k||≈1`，`β/(||k||^2+eps)` 归一化会退化，FNKDA 的核心意义会变弱。

因此实现中禁止 `use_qk_l2norm_in_kernel=True`。

---

## 3. 代码变更（文件级）

新增：
- `fla/layers/fnkda.py`
- `fla/models/fnkda/__init__.py`
- `fla/models/fnkda/configuration_fnkda.py`
- `fla/models/fnkda/modeling_fnkda.py`
- `examples/train_fnkda_wikitext103.py`
- `examples/eval_fnkda_wikitext103.py`
- `doc/fnkda_zh.md`

修改：
- `fla/layers/__init__.py`：导出 FNKDA attention 层
- `fla/models/__init__.py`：导出 `FNKDAConfig/FNKDAModel/FNKDAForCausalLM`

---

## 4. 训练与评测（WikiText-103）

训练示例（单卡）：

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/train_fnkda_wikitext103.py \
  --output_dir exp/fnkda-wt103 \
  --seq_len 1024 \
  --max_steps 2000 \
  --tokenized_cache data/wt103_tok \
  --fp16 \
  --preflight_compile
```

评测示例：

```bash
python examples/eval_fnkda_wikitext103.py \
  --model exp/fnkda-wt103 \
  --split validation \
  --seq_len 1024 \
  --tokenized_cache data/wt103_tok \
  --dtype fp16 \
  --generate
```

