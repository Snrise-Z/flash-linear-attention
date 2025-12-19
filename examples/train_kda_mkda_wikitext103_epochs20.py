#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from functools import partial
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

import fla  # noqa: F401  (registers FLA models/configs with HF auto classes)
from fla.models import KDAConfig, MKDAConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train both KDA and MKDA on WikiText-103 for a fixed budget (max_epochs=20), "
            "save best validation checkpoint, then run test eval using best-val checkpoint."
        )
    )

    p.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or local path.")
    p.add_argument("--dataset_name", type=str, default="wikitext", help="HF datasets name.")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1", help="HF datasets config.")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--cache_dir", type=str, default=None, help="HF datasets cache_dir.")
    p.add_argument("--tokenized_cache", type=str, default=None, help="If set, save/load tokenized dataset here.")

    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)

    p.add_argument("--output_root", type=str, default="exp/kda-mkda-wt103-e20")
    p.add_argument("--resume_kda", type=str, default=None, help="Checkpoint dir to resume KDA training from.")
    p.add_argument("--resume_mkda", type=str, default=None, help="Checkpoint dir to resume MKDA training from.")

    # Shared architecture
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--expand_v", type=float, default=1.0)
    p.add_argument("--attn_mode", type=str, default="chunk", choices=["chunk", "fused_recurrent"])
    p.add_argument("--use_short_conv", action="store_true", default=False)
    p.add_argument("--allow_neg_eigval", action="store_true", default=False)
    p.add_argument("--no_fuse_norm", action="store_true", default=False, help="Disable Triton fused RMSNorm.")
    p.add_argument("--no_fuse_swiglu", action="store_true", default=False, help="Disable fused SwiGLU.")
    p.add_argument("--no_fuse_cross_entropy", action="store_true", default=False, help="Disable fused CE.")

    # MKDA-specific
    p.add_argument("--micro_rank", type=int, default=4)
    p.add_argument("--micro_readout_mode", type=str, default="mix", choices=["mix", "last"])
    p.add_argument("--beta_reg_lambda", type=float, default=0.0)
    p.add_argument("--beta_reg_max", type=float, default=1.0)
    p.add_argument("--orth_reg_lambda", type=float, default=0.0)

    # Training budget
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fp16", action="store_true", default=None, help="Force fp16 training (override auto).")
    p.add_argument("--bf16", action="store_true", default=None, help="Force bf16 training (override auto).")
    p.add_argument("--dataloader_num_workers", type=int, default=0, help="DataLoader workers (0 is most stable).")
    p.add_argument(
        "--preflight_compile",
        action="store_true",
        default=False,
        help="Run one tiny forward/backward to trigger Triton compilation before Trainer starts.",
    )

    return p.parse_args()


def _detect_mixed_precision(args: argparse.Namespace) -> tuple[bool, bool]:
    if args.fp16 is True and args.bf16 is True:
        raise ValueError("Only one of --fp16/--bf16 can be set.")

    if args.fp16 is True:
        return True, False
    if args.bf16 is True:
        return False, True

    if not torch.cuda.is_available():
        return False, False

    bf16_supported = False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_supported = False
    return (not bf16_supported), bf16_supported


def _normalize_text(example: dict[str, Any], text_column: str) -> dict[str, Any]:
    text = example.get(text_column, "")
    if text is None:
        text = ""
    return {"text": text.strip()}


def _tokenize_batch(examples: dict[str, list[Any]], tokenizer, eos_token_id: int) -> dict[str, list[list[int]]]:
    texts = [t for t in examples["text"] if t]
    if not texts:
        return {"input_ids": []}

    tokenized = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    tokenized = [ids + [eos_token_id] for ids in tokenized if len(ids) > 0]
    return {"input_ids": tokenized}


def _group_texts(examples: dict[str, list[list[int]]], seq_len: int) -> dict[str, list[list[int]]]:
    if not examples["input_ids"]:
        return {"input_ids": [], "labels": []}

    concatenated = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)

    total_len = (len(concatenated) // seq_len) * seq_len
    if total_len == 0:
        return {"input_ids": [], "labels": []}

    input_ids = [concatenated[i : i + seq_len] for i in range(0, total_len, seq_len)]
    return {"input_ids": input_ids, "labels": input_ids.copy()}


def load_or_build_tokenized_dataset(args: argparse.Namespace, tokenizer) -> DatasetDict:
    if args.tokenized_cache is not None and os.path.isdir(args.tokenized_cache):
        return load_from_disk(args.tokenized_cache)

    raw: DatasetDict = load_dataset(args.dataset_name, args.dataset_config, cache_dir=args.cache_dir)
    raw = raw.map(partial(_normalize_text, text_column=args.text_column), desc="Normalize text")
    raw = raw.filter(lambda ex: bool(ex["text"]), desc="Drop empty lines")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; please set a tokenizer with EOS.")

    remove_cols = list(raw["train"].features.keys())
    tokenized = raw.map(
        partial(_tokenize_batch, tokenizer=tokenizer, eos_token_id=eos_id),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=remove_cols,
        desc="Tokenize",
    )
    lm_ds = tokenized.map(
        partial(_group_texts, seq_len=args.seq_len),
        batched=True,
        num_proc=args.num_proc,
        desc=f"Group texts into chunks of {args.seq_len}",
    )

    if args.max_train_samples is not None:
        lm_ds["train"] = lm_ds["train"].select(range(min(args.max_train_samples, len(lm_ds["train"]))))
    if "validation" in lm_ds and args.max_eval_samples is not None:
        lm_ds["validation"] = lm_ds["validation"].select(range(min(args.max_eval_samples, len(lm_ds["validation"]))))
    if "test" in lm_ds and args.max_test_samples is not None:
        lm_ds["test"] = lm_ds["test"].select(range(min(args.max_test_samples, len(lm_ds["test"]))))

    if args.tokenized_cache is not None:
        os.makedirs(args.tokenized_cache, exist_ok=True)
        lm_ds.save_to_disk(args.tokenized_cache)

    return lm_ds


@torch.no_grad()
def evaluate_split(model, dataset, *, batch_size: int, device: str) -> dict[str, float]:
    from torch.utils.data import DataLoader

    model.eval()
    collator = DefaultDataCollator()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        tokens = labels.numel()
        total_loss += float(out.loss) * tokens
        total_tokens += tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


def _build_kda_config(args: argparse.Namespace, tokenizer) -> KDAConfig:
    return KDAConfig(
        attn_mode=args.attn_mode,
        hidden_size=args.hidden_size,
        expand_v=args.expand_v,
        use_short_conv=args.use_short_conv,
        allow_neg_eigval=args.allow_neg_eigval,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        max_position_embeddings=args.seq_len,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        fuse_norm=not args.no_fuse_norm,
        fuse_swiglu=not args.no_fuse_swiglu,
        fuse_cross_entropy=not args.no_fuse_cross_entropy,
    )


def _build_mkda_config(args: argparse.Namespace, tokenizer) -> MKDAConfig:
    return MKDAConfig(
        attn_mode=args.attn_mode,
        hidden_size=args.hidden_size,
        expand_v=args.expand_v,
        use_short_conv=args.use_short_conv,
        allow_neg_eigval=args.allow_neg_eigval,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        micro_rank=args.micro_rank,
        micro_readout_mode=args.micro_readout_mode,
        beta_reg_lambda=args.beta_reg_lambda,
        beta_reg_max=args.beta_reg_max,
        orth_reg_lambda=args.orth_reg_lambda,
        max_position_embeddings=args.seq_len,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        fuse_norm=not args.no_fuse_norm,
        fuse_swiglu=not args.no_fuse_swiglu,
        fuse_cross_entropy=not args.no_fuse_cross_entropy,
    )


def _train_one(
    *,
    run_name: str,
    output_dir: str,
    resume_from_checkpoint: str | None,
    config,
    dataset: DatasetDict,
    tokenizer,
    fp16: bool,
    bf16: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_config(config)

    if args.preflight_compile and torch.cuda.is_available():
        mp_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        tmp_model = model.to("cuda", dtype=mp_dtype).train()
        t = min(args.seq_len, 128)
        input_ids = torch.randint(0, config.vocab_size, (1, t), device="cuda")
        out = tmp_model(input_ids=input_ids, labels=input_ids, use_cache=False)
        out.loss.backward()
        torch.cuda.synchronize()
        tmp_model = tmp_model.to("cpu")
        del tmp_model
        model = model.to(dtype=torch.float32)

    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=("validation" in dataset),
        eval_strategy="epoch" if "validation" in dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=("validation" in dataset),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        num_train_epochs=float(args.max_epochs),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        seed=args.seed,
        fp16=fp16,
        bf16=bf16,
        report_to="none",
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        data_collator=DefaultDataCollator(),
        processing_class=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # With load_best_model_at_end=True, trainer.model is the best-val model here.
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    best_ckpt = trainer.state.best_model_checkpoint
    state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {}
    state["run_name"] = run_name
    state["best_model_checkpoint"] = best_ckpt
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    metrics = dict(train_result.metrics)
    metrics["best_model_checkpoint"] = best_ckpt
    metrics["train_ppl"] = math.exp(metrics["train_loss"]) if "train_loss" in metrics else None
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Test on best-val checkpoint (budget-limited best).
    test_metrics = None
    if "test" in dataset:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mp_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        best_model = AutoModelForCausalLM.from_pretrained(best_ckpt or output_dir, torch_dtype=mp_dtype).to(device)
        test_metrics = evaluate_split(best_model, dataset["test"], batch_size=args.per_device_eval_batch_size, device=device)
        with open(os.path.join(output_dir, "test_bestval.json"), "w", encoding="utf-8") as f:
            json.dump({"best_model_checkpoint": best_ckpt, "test": test_metrics}, f, ensure_ascii=False, indent=2)
        print(f"[{run_name}] test(best-val): loss={test_metrics['loss']:.6f} ppl={test_metrics['perplexity']:.3f}", flush=True)

    return {"train": metrics, "test": test_metrics, "best_model_checkpoint": best_ckpt}


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp16, bf16 = _detect_mixed_precision(args)

    dataset = load_or_build_tokenized_dataset(args, tokenizer)

    kda_dir = os.path.join(args.output_root, "kda")
    mkda_dir = os.path.join(args.output_root, "mkda")

    print(f"[budget] max_epochs={args.max_epochs} seq_len={args.seq_len}", flush=True)
    print(f"[paths] output_root={args.output_root}", flush=True)

    kda_cfg = _build_kda_config(args, tokenizer)
    mkda_cfg = _build_mkda_config(args, tokenizer)

    kda_out = _train_one(
        run_name="kda",
        output_dir=kda_dir,
        resume_from_checkpoint=args.resume_kda,
        config=kda_cfg,
        dataset=dataset,
        tokenizer=tokenizer,
        fp16=fp16,
        bf16=bf16,
        args=args,
    )
    mkda_out = _train_one(
        run_name="mkda",
        output_dir=mkda_dir,
        resume_from_checkpoint=args.resume_mkda,
        config=mkda_cfg,
        dataset=dataset,
        tokenizer=tokenizer,
        fp16=fp16,
        bf16=bf16,
        args=args,
    )

    with open(os.path.join(args.output_root, "summary_bestval_test.json"), "w", encoding="utf-8") as f:
        json.dump({"kda": kda_out, "mkda": mkda_out}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

