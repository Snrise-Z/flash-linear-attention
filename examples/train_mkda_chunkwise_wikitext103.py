#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from functools import partial
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments

import fla  # noqa: F401
from fla.models import MKDAConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train chunkwise-parallel MKDA on WikiText-103 (HF Trainer).")

    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--tokenized_cache", type=str, default=None)

    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

    p.add_argument("--output_dir", type=str, default="exp/mkda-chunkwise-wt103")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--expand_v", type=float, default=1.0)
    p.add_argument("--micro_rank", type=int, default=4, help="Number of keys per token (R).")
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--rank_mix", type=str, default="none", choices=["none", "kv"], help="Optional rank-mix projection (default: none).")

    p.add_argument("--use_short_conv", action="store_true", default=False)
    p.add_argument("--allow_neg_eigval", action="store_true", default=False)
    p.add_argument("--no_fuse_norm", action="store_true", default=False)
    p.add_argument("--no_fuse_swiglu", action="store_true", default=False)
    p.add_argument("--no_fuse_cross_entropy", action="store_true", default=False)

    p.add_argument("--beta_reg_lambda", type=float, default=0.0)
    p.add_argument("--beta_reg_max", type=float, default=1.0)
    p.add_argument("--orth_reg_lambda", type=float, default=0.0)

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

    p.add_argument("--fp16", action="store_true", default=None)
    p.add_argument("--bf16", action="store_true", default=None)
    p.add_argument("--dataloader_num_workers", type=int, default=0)

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
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
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
    concatenated: list[int] = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)
    total_len = (len(concatenated) // seq_len) * seq_len
    if total_len == 0:
        return {"input_ids": [], "labels": []}
    input_ids = [concatenated[i : i + seq_len] for i in range(0, total_len, seq_len)]
    return {"input_ids": input_ids, "labels": input_ids.copy()}


def load_or_build_tokenized_dataset(args: argparse.Namespace, tokenizer) -> DatasetDict:
    if args.tokenized_cache is not None and os.path.isdir(args.tokenized_cache):
        ds: DatasetDict = load_from_disk(args.tokenized_cache)
        return ds

    raw: DatasetDict = load_dataset(args.dataset_name, args.dataset_config, cache_dir=args.cache_dir)
    raw = raw.map(partial(_normalize_text, text_column=args.text_column), desc="Normalize text")
    raw = raw.filter(lambda ex: bool(ex["text"]), desc="Drop empty lines")
    remove_cols = list(raw["train"].features.keys())

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; please set a tokenizer with EOS.")

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

    if args.tokenized_cache is not None:
        os.makedirs(args.tokenized_cache, exist_ok=True)
        lm_ds.save_to_disk(args.tokenized_cache)

    return lm_ds


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp16, bf16 = _detect_mixed_precision(args)

    config = MKDAConfig(
        mkda_impl="chunkwise",
        chunk_size=args.chunk_size,
        attn_mode="chunk",  # unused for chunkwise impl
        hidden_size=args.hidden_size,
        expand_v=args.expand_v,
        use_short_conv=args.use_short_conv,
        allow_neg_eigval=args.allow_neg_eigval,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        micro_rank=args.micro_rank,
        rank_mix=args.rank_mix,
        max_position_embeddings=args.seq_len,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        fuse_norm=not args.no_fuse_norm,
        fuse_swiglu=not args.no_fuse_swiglu,
        fuse_cross_entropy=not args.no_fuse_cross_entropy,
        beta_reg_lambda=args.beta_reg_lambda,
        beta_reg_max=args.beta_reg_max,
        orth_reg_lambda=args.orth_reg_lambda,
    )
    model = AutoModelForCausalLM.from_config(config)

    dataset = load_or_build_tokenized_dataset(args, tokenizer)

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
