#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla  # noqa: F401
from fla.models import FNKDAConfig

from _wikitext103_common import (
    build_training_args,
    detect_mixed_precision_train,
    load_or_build_tokenized_dataset,
    maybe_preflight_compile,
    train_with_trainer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small FNKDA model on WikiText-103 (HF Trainer).")

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

    p.add_argument("--output_dir", type=str, default="exp/fnkda-wikitext103")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--expand_v", type=float, default=1.0)
    p.add_argument("--attn_mode", type=str, default="chunk", choices=["chunk", "fused_recurrent"])
    p.add_argument("--use_short_conv", action="store_true", default=False)
    p.add_argument("--allow_neg_eigval", action="store_true", default=False)

    # FNKDA knobs
    p.add_argument("--beta_norm_eps", type=float, default=None)
    p.add_argument("--fix_lambda", type=float, default=None)
    p.add_argument("--share_decay_gate", action="store_true", default=None)

    p.add_argument("--no_fuse_norm", action="store_true", default=False)
    p.add_argument("--no_fuse_swiglu", action="store_true", default=False)
    p.add_argument("--no_fuse_cross_entropy", action="store_true", default=False)

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
    p.add_argument("--preflight_compile", action="store_true", default=False)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    fp16, bf16 = detect_mixed_precision_train(args)

    config_kwargs: dict[str, object] = {}
    if args.beta_norm_eps is not None:
        config_kwargs["beta_norm_eps"] = args.beta_norm_eps
    if args.fix_lambda is not None:
        config_kwargs["fix_lambda"] = args.fix_lambda
    if args.share_decay_gate is not None:
        config_kwargs["share_decay_gate"] = args.share_decay_gate

    config = FNKDAConfig(
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
        **config_kwargs,
    )
    model = AutoModelForCausalLM.from_config(config)

    dataset = load_or_build_tokenized_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        cache_dir=args.cache_dir,
        tokenized_cache=args.tokenized_cache,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        num_proc=args.num_proc,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    maybe_preflight_compile(
        enabled=args.preflight_compile,
        model=model,
        vocab_size=config.vocab_size,
        seq_len=args.seq_len,
        fp16=fp16,
        bf16=bf16,
    )

    train_args = build_training_args(args, fp16=fp16, bf16=bf16, has_validation=("validation" in dataset))
    train_with_trainer(
        model=model,
        dataset=dataset,
        train_args=train_args,
        resume_from_checkpoint=args.resume_from_checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

