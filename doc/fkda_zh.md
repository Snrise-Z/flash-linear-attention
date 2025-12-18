# FKDA（Fast/Slow Kimi Delta Attention）中文说明与变更记录

本文档说明本仓库中引入 **FKDA（Fast/Slow Kimi Delta Attention）** 的全部相关变更：在保持 KDA `chunk_kda` 并行的前提下，为每层维护两份 KDA 记忆（fast/slow），并用 `λ` 融合输出。

---

## 1. 公式（单头写法）

fast/slow 两份记忆分别做 KDA 型更新：

`S_t^(x) = (I - β_t^(x) k_t k_t^T) Diag(α_t^(x)) S_{t-1}^(x) + β_t^(x) k_t v_t^T,  x∈{f,s}`

输出融合：

`o_t = λ_t o_t^(f) + (1-λ_t) o_t^(s)`

其中 `α_t^(x)` 仍由 KDA gate 内核计算（`use_gate_in_kernel=True`），所以并行性与 kernel 复用保持不变。

---

## 2. 并行性说明

FKDA 的门控（`β_fast/β_slow/λ` 与 `g_fast/g_slow`）只依赖当前 token 的 `hidden_states` 投影结果，不依赖 `S_{t-1}`，因此递推保持仿射线性，可继续使用 `chunk_kda` 做 chunk-parallel prefix-scan。

实现上调用 `chunk_kda` 两次（fast/slow 各一次）。

---

## 3. 代码变更（文件级）

新增：
- `fla/layers/fkda.py`
- `fla/models/fkda/__init__.py`
- `fla/models/fkda/configuration_fkda.py`
- `fla/models/fkda/modeling_fkda.py`
- `examples/train_fkda_wikitext103.py`
- `examples/eval_fkda_wikitext103.py`
- `doc/fkda_zh.md`

修改：
- `fla/layers/__init__.py`：导出 FKDA attention 层
- `fla/models/__init__.py`：导出 `FKDAConfig/FKDAModel/FKDAForCausalLM`

---

## 4. 训练与评测（WikiText-103）

训练示例（单卡）：

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/train_fkda_wikitext103.py \
  --output_dir exp/fkda-wt103 \
  --seq_len 1024 \
  --max_steps 2000 \
  --tokenized_cache data/wt103_tok \
  --fp16 \
  --preflight_compile
```

评测示例：

```bash
python examples/eval_fkda_wikitext103.py \
  --model exp/fkda-wt103 \
  --split validation \
  --seq_len 1024 \
  --tokenized_cache data/wt103_tok \
  --dtype fp16 \
  --generate
```

