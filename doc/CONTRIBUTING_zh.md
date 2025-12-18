# 贡献指南（中文）

本文档约定在本仓库中新增/改进模型与算子（尤其是 KDA 系列变体）时的最小交付标准与提交流程，目的是让每个改进都可复现、可回归、可对照实验。

---

## 1. 总体原则

- **每个改进必须可复现**：任何新增的算法/实现都必须提供“能跑通”的训练与推理/评测路径。
- **每个改进必须可解释**：必须提供独立文档说明动机、公式/约束、实现细节与开关。
- **每个改进必须可回归**：不应破坏现有模型/算子；如有兼容性调整需说明原因与影响面。
- **并行性优先**：若目标是保持 chunk 并行/内核复用，设计必须明确说明“哪些量不能依赖 `S_{t-1}`（或等价的递推状态）”。

---

## 2. 强制要求：文档 + 训练脚本 + 推理/评测脚本

对任何“改进/新变体”（例如：NKDA/SKDA/SNKDA 等）都必须满足：

### 2.1 独立中文文档

- 路径：`doc/<variant>_zh.md`
- 内容必须覆盖：
  - 改进动机与与基线（KDA/NKDA/…）的关系
  - 关键公式/递推形式
  - 并行性/内核兼容性约束与解释（若适用）
  - 实现位置（文件级变更列表）
  - 关键配置项（Config 字段）与默认值
  - 推荐的训练/评测命令（最小可跑通）

### 2.2 独立训练脚本

- 路径：`examples/train_<variant>_wikitext103.py`
- 最低要求：
  - 使用该变体对应的 `*Config`（例如 `SKDAConfig`）
  - 支持从头训练一个“小规模可跑通”的配置（默认参数即可）
  - 支持 `--tokenized_cache`（避免重复 tokenize）
  - 支持 `--fp16/--bf16`（自动或手动选择）
  - 支持 `--preflight_compile`（可选，但强烈建议，用于提前触发 Triton 编译）
  - 支持常见稳定性开关（例如禁用 fused norm 的开关，如该模型/环境需要）

### 2.3 独立推理/评测脚本

- 路径：`examples/eval_<variant>_wikitext103.py`
- 最低要求：
  - 能在 WikiText-103 上计算 `loss` 与 `ppl`
  - 可选：支持 `--generate` 做 generation sanity check
  - 允许通过 CLI 开关覆盖关键 ablation 参数（若有）

---

## 3. 目录/命名规范

### 3.1 Layer / Model 命名

- Layer 文件：`fla/layers/<variant>.py`
- Model 目录：`fla/models/<variant>/`
  - `configuration_<variant>.py`
  - `modeling_<variant>.py`
  - `__init__.py`（负责 HF auto 注册）
- Config：
  - `model_type` 必须唯一（例如：`nkda`/`skda`/`snkda`）
  - 新增参数必须有默认值并记录在 `doc/<variant>_zh.md`

### 3.2 可复用内核/并行性说明

如果你的变体目标是“保持 chunk 并行并复用 `chunk_kda`/类似内核”：
- 你的 gate/步长等控制量 **不能依赖 `S_{t-1}`**（包括 `S_{t-1}^T k_t` 这类需要读取状态的项）
- 必须在文档里明确说明这一点，并解释你的 proxy/近似如何保持 `S_t = A_t S_{t-1} + B_t` 的仿射结构

---

## 4. 提交（commit）规则：每个改进是一个“自包含提交”

我们要求“每个改进”对应的引入提交是自包含的：

- 该提交必须同时包含：
  - 变体实现代码（layer/model/config/注册）
  - `doc/<variant>_zh.md`
  - `examples/train_<variant>_wikitext103.py`
  - `examples/eval_<variant>_wikitext103.py`
- 后续对该改进的补充（例如补脚本/补文档）应当 **合并回该改进的引入提交**：
  - 推荐做法：使用 `git commit --fixup=<introduce_commit>` + `git rebase -i --autosquash`

> 这样做的好处：任何人只要 checkout 到该提交，就能立刻读文档、跑训练、做评测；不会出现“代码加了但脚本/文档散落在后续提交里”的情况。

---

## 5. 建议的最小验证流程（提交前）

在本地（至少）完成以下之一：

- **静态检查**：`python -m compileall -q <涉及的脚本/模块>`
- **单测（如果你改了算子/核心逻辑）**：优先跑相关子集（例如 `pytest -q -k kda`）
- **端到端冒烟**：用你的 `train_<variant>_wikitext103.py` 跑少量 step（例如 `--max_steps 20`），并用 `eval_<variant>_wikitext103.py` 能跑出 loss/ppl

---

## 6. 常见坑与建议

- **BF16 支持**：Turing（如 2080Ti）不支持 bf16；脚本要能用 fp16 跑通。
- **Triton 首次编译很慢**：建议提供 `--preflight_compile` 并在文档里说明。
- **DataParallel 与 Triton autotune**：多卡可见时 HF Trainer 可能走 DP，某些环境下会不稳定；建议在文档/脚本提示用 `CUDA_VISIBLE_DEVICES=0` 或 `accelerate launch --num_processes 1`。

