#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from fla.models import MKDAConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small MKDA (micro-step rank-r) model on WikiText-103 (HF Trainer).")

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

    p.add_argument("--output_dir", type=str, default="exp/mkda-wikitext103")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--expand_v", type=float, default=1.0)
    p.add_argument("--attn_mode", type=str, default="chunk", choices=["chunk", "fused_recurrent"])
    p.add_argument("--micro_rank", type=int, default=4, help="Rank r for micro-step approximation (sequence expands to T*r).")
    p.add_argument("--micro_fill_g_raw", type=float, default=-1.0e4, help="Raw gate fill value for non-first micro-steps.")

    p.add_argument("--use_short_conv", action="store_true", default=False)
    p.add_argument("--allow_neg_eigval", action="store_true", default=False)
    p.add_argument("--no_fuse_norm", action="store_true", default=False, help="Disable Triton fused RMSNorm.")
    p.add_argument("--no_fuse_swiglu", action="store_true", default=False, help="Disable fused SwiGLU.")
    p.add_argument("--no_fuse_cross_entropy", action="store_true", default=False, help="Disable fused CE.")

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
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

    raw: DatasetDict = load_dataset(
        args.dataset_name,
        args.dataset_config,
        cache_dir=args.cache_dir,
    )

    raw = raw.map(partial(_normalize_text, text_column=args.text_column), desc="Normalize text")
    raw = raw.filter(lambda ex: bool(ex["text"]), desc="Drop empty lines")
    remove_cols = list(raw["train"].features.keys())

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; please set a tokenizer with EOS.")

    tokenized = raw.map(
        partial(_tokenize_batch, tokenizer=tokenizer, eos_token_id=eos_id),
        batched=True,
        remove_columns=remove_cols,
        num_proc=args.num_proc,
        desc="Tokenize",
    )

    lm_ds = tokenized.map(
        partial(_group_texts, seq_len=args.seq_len),
        batched=True,
        num_proc=args.num_proc,
        desc="Group texts",
    )

    if args.max_train_samples is not None:
        lm_ds["train"] = lm_ds["train"].select(range(min(args.max_train_samples, len(lm_ds["train"]))))
    if "validation" in lm_ds and args.max_eval_samples is not None:
        lm_ds["validation"] = lm_ds["validation"].select(range(min(args.max_eval_samples, len(lm_ds["validation"]))))

    if args.tokenized_cache is not None:
        os.makedirs(args.tokenized_cache, exist_ok=True)
        lm_ds.save_to_disk(args.tokenized_cache)

    return lm_ds


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp16, bf16 = _detect_mixed_precision(args)

    config = MKDAConfig(
        attn_mode=args.attn_mode,
        hidden_size=args.hidden_size,
        expand_v=args.expand_v,
        use_short_conv=args.use_short_conv,
        allow_neg_eigval=args.allow_neg_eigval,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        micro_rank=args.micro_rank,
        micro_fill_g_raw=args.micro_fill_g_raw,
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
    model = AutoModelForCausalLM.from_config(config)

    dataset = load_or_build_tokenized_dataset(args, tokenizer)

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
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=("validation" in dataset),
        eval_strategy="steps" if "validation" in dataset else "no",
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
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

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics["train_ppl"] = math.exp(metrics["train_loss"]) if "train_loss" in metrics else None
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()

