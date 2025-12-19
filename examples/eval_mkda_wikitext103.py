#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from functools import partial
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

import fla  # noqa: F401  (registers FLA models/configs with HF auto classes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an MKDA (micro-step rank-r) model on WikiText-103 (loss/perplexity).")

    p.add_argument("--model", type=str, required=True, help="Path to a trained MKDA HF checkpoint directory.")
    p.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name/path (default: use --model).")

    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--tokenized_cache", type=str, default=None, help="If set, save/load tokenized dataset here.")
    p.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])

    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])

    p.add_argument("--generate", action="store_true", help="Also run a small generate() sanity check.")
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument(
        "--print_microstep_stats",
        action="store_true",
        default=False,
        help="Print micro-step config stats from model config, then continue.",
    )

    return p.parse_args()


def _detect_dtype(dtype_flag: str) -> torch.dtype | None:
    if dtype_flag == "fp32":
        return torch.float32
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "bf16":
        return torch.bfloat16
    if dtype_flag != "auto":
        raise ValueError(f"Unknown dtype: {dtype_flag}")

    if not torch.cuda.is_available():
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


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


def load_or_build_tokenized_split(args: argparse.Namespace, tokenizer) -> Any:
    if args.tokenized_cache is not None and os.path.isdir(args.tokenized_cache):
        ds: DatasetDict = load_from_disk(args.tokenized_cache)
        return ds[args.split]

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

    if args.tokenized_cache is not None:
        os.makedirs(args.tokenized_cache, exist_ok=True)
        lm_ds.save_to_disk(args.tokenized_cache)

    split_ds = lm_ds[args.split]
    if args.max_samples is not None:
        split_ds = split_ds.select(range(min(args.max_samples, len(split_ds))))
    return split_ds


@torch.no_grad()
def evaluate_ppl(model, dataloader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)

        tokens = labels.numel()
        total_loss += float(out.loss) * tokens
        total_tokens += tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    tok_path = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _detect_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)

    if args.print_microstep_stats:
        cfg = getattr(model, "config", None)
        micro_rank = getattr(cfg, "micro_rank", None)
        micro_fill_g_raw = getattr(cfg, "micro_fill_g_raw", None)
        print("[mkda] model_type=", getattr(cfg, "model_type", None), flush=True)
        print(f"[mkda] micro_rank={micro_rank} micro_fill_g_raw={micro_fill_g_raw}", flush=True)
        if micro_rank is not None:
            print(f"[mkda] seq_len={args.seq_len} expanded_len(T*r)={args.seq_len * int(micro_rank)}", flush=True)

    dataset = load_or_build_tokenized_split(args, tokenizer)
    collator = DefaultDataCollator()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    results = evaluate_ppl(model, dataloader, args.device)
    print(f"split={args.split} seq_len={args.seq_len} batch_size={args.batch_size}")
    print(f"loss={results['loss']:.6f} ppl={results['perplexity']:.3f}")

    if args.generate:
        model.eval()
        enc = tokenizer(args.prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(args.device)
        attention_mask = enc.attention_mask.to(args.device) if "attention_mask" in enc else None
        gen = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        print("--- prompt ---")
        print(args.prompt)
        print("--- generation ---")
        print(tokenizer.decode(gen[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
