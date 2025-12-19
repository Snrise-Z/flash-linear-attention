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
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator

import fla  # noqa: F401  (registers FLA models/configs with HF auto classes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate KDA+MKDA runs on WikiText-103 by selecting each run's best validation checkpoint "
            "(from trainer_state.json) and reporting validation/test perplexity."
        )
    )

    p.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory containing subdirs 'kda' and 'mkda' (e.g., exp/kda-mkda-wt103-e20).",
    )
    p.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name/path (default: use output_root/kda).")

    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--tokenized_cache", type=str, default=None, help="If set, save/load tokenized dataset here.")

    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])

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


def load_or_build_tokenized_dataset(args: argparse.Namespace, tokenizer) -> DatasetDict:
    if args.tokenized_cache is not None and os.path.isdir(args.tokenized_cache):
        ds: DatasetDict = load_from_disk(args.tokenized_cache)
    else:
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
        ds = tokenized.map(
            partial(_group_texts, seq_len=args.seq_len),
            batched=True,
            num_proc=args.num_proc,
            desc=f"Group texts into chunks of {args.seq_len}",
        )
        if args.tokenized_cache is not None:
            os.makedirs(args.tokenized_cache, exist_ok=True)
            ds.save_to_disk(args.tokenized_cache)

    if "validation" in ds and args.max_eval_samples is not None:
        ds["validation"] = ds["validation"].select(range(min(args.max_eval_samples, len(ds["validation"]))))
    if "test" in ds and args.max_test_samples is not None:
        ds["test"] = ds["test"].select(range(min(args.max_test_samples, len(ds["test"]))))
    return ds


@torch.no_grad()
def evaluate_ppl(model, dataset, *, batch_size: int, device: str) -> dict[str, float]:
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


def _best_checkpoint(run_dir: str) -> str:
    state_path = os.path.join(run_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing {state_path}")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    ckpt = state.get("best_model_checkpoint") or state.get("best_model_checkpoint", None)
    if not ckpt:
        raise ValueError(f"No best_model_checkpoint found in {state_path}")
    return ckpt


def _eval_one(*, name: str, run_dir: str, dataset: DatasetDict, dtype: torch.dtype | None, device: str, batch_size: int):
    ckpt = _best_checkpoint(run_dir)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dtype).to(device)

    out = {"name": name, "best_model_checkpoint": ckpt}
    if "validation" in dataset:
        out["validation"] = evaluate_ppl(model, dataset["validation"], batch_size=batch_size, device=device)
    if "test" in dataset:
        out["test"] = evaluate_ppl(model, dataset["test"], batch_size=batch_size, device=device)

    out_path = os.path.join(run_dir, "eval_bestval_val_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[{name}] wrote {out_path}", flush=True)
    if "validation" in out:
        print(f"[{name}] val:  loss={out['validation']['loss']:.6f} ppl={out['validation']['perplexity']:.3f}", flush=True)
    if "test" in out:
        print(f"[{name}] test: loss={out['test']['loss']:.6f} ppl={out['test']['perplexity']:.3f}", flush=True)
    return out


def main() -> None:
    args = parse_args()
    dtype = _detect_dtype(args.dtype)

    tok_path = args.tokenizer if args.tokenizer is not None else os.path.join(args.output_root, "kda")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_or_build_tokenized_dataset(args, tokenizer)

    kda_dir = os.path.join(args.output_root, "kda")
    mkda_dir = os.path.join(args.output_root, "mkda")

    results = {
        "output_root": args.output_root,
        "kda": _eval_one(name="kda", run_dir=kda_dir, dataset=dataset, dtype=dtype, device=args.device, batch_size=args.batch_size),
        "mkda": _eval_one(name="mkda", run_dir=mkda_dir, dataset=dataset, dtype=dtype, device=args.device, batch_size=args.batch_size),
    }
    with open(os.path.join(args.output_root, "eval_summary_bestval_val_test.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

