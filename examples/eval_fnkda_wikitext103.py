#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

import fla  # noqa: F401

from _wikitext103_common import (
    detect_dtype_eval,
    evaluate_ppl,
    load_or_build_tokenized_split,
    override_attr_on_modules,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a FNKDA model on WikiText-103 (loss/perplexity).")

    p.add_argument("--model", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None)

    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--tokenized_cache", type=str, default=None)
    p.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])

    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])

    p.add_argument("--fix_lambda", type=float, default=None)
    p.add_argument("--share_decay_gate", action="store_true", default=False)

    p.add_argument("--generate", action="store_true")
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--max_new_tokens", type=int, default=64)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    tok_path = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = detect_dtype_eval(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)

    if args.fix_lambda is not None:
        changed = override_attr_on_modules(model, "fix_lambda", float(args.fix_lambda))
        print(f"[ablation] set fix_lambda={args.fix_lambda} on {changed} modules")
    if args.share_decay_gate:
        changed = override_attr_on_modules(model, "share_decay_gate", True)
        print(f"[ablation] set share_decay_gate=True on {changed} modules")

    dataset = load_or_build_tokenized_split(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        cache_dir=args.cache_dir,
        tokenized_cache=args.tokenized_cache,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        num_proc=args.num_proc,
        split=args.split,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=DefaultDataCollator())

    results = evaluate_ppl(model, dataloader, args.device)
    print(f"split={args.split} seq_len={args.seq_len} batch_size={args.batch_size}")
    print(f"loss={results['loss']:.6f} ppl={results['perplexity']:.3f}")

    if args.generate:
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

