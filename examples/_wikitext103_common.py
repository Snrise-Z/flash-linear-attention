from __future__ import annotations

import argparse
import math
import os
from functools import partial
from typing import Any, Callable

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import DefaultDataCollator, Trainer, TrainingArguments


def add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or local path.")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--tokenized_cache", type=str, default=None)

    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=8)


def add_train_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

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

    p.add_argument("--no_fuse_norm", action="store_true", default=False)
    p.add_argument("--no_fuse_swiglu", action="store_true", default=False)
    p.add_argument("--no_fuse_cross_entropy", action="store_true", default=False)


def add_eval_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])

    p.add_argument("--generate", action="store_true")
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--max_new_tokens", type=int, default=64)


def detect_mixed_precision_train(args: argparse.Namespace) -> tuple[bool, bool]:
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


def detect_dtype_eval(dtype_flag: str) -> torch.dtype | None:
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


def normalize_text(example: dict[str, Any], text_column: str) -> dict[str, Any]:
    text = example.get(text_column, "")
    if text is None:
        text = ""
    return {"text": text.strip()}


def tokenize_batch(examples: dict[str, list[Any]], tokenizer, eos_token_id: int) -> dict[str, list[list[int]]]:
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


def group_texts(examples: dict[str, list[list[int]]], seq_len: int) -> dict[str, list[list[int]]]:
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


def load_or_build_tokenized_dataset(
    *,
    dataset_name: str,
    dataset_config: str,
    text_column: str,
    cache_dir: str | None,
    tokenized_cache: str | None,
    tokenizer,
    seq_len: int,
    num_proc: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> DatasetDict:
    if tokenized_cache is not None and os.path.isdir(tokenized_cache):
        ds: DatasetDict = load_from_disk(tokenized_cache)
        if max_train_samples is not None:
            ds["train"] = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))
        if "validation" in ds and max_eval_samples is not None:
            ds["validation"] = ds["validation"].select(range(min(max_eval_samples, len(ds["validation"]))))
        return ds

    raw: DatasetDict = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    raw = raw.map(partial(normalize_text, text_column=text_column), desc="Normalize text")
    raw = raw.filter(lambda ex: bool(ex["text"]), desc="Drop empty lines")
    remove_cols = list(raw["train"].features.keys())

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; please set a tokenizer with EOS.")

    tokenized = raw.map(
        partial(tokenize_batch, tokenizer=tokenizer, eos_token_id=eos_id),
        batched=True,
        remove_columns=remove_cols,
        num_proc=num_proc,
        desc="Tokenize",
    )
    lm_ds = tokenized.map(
        partial(group_texts, seq_len=seq_len),
        batched=True,
        num_proc=num_proc,
        desc="Group texts",
    )

    if tokenized_cache is not None:
        os.makedirs(tokenized_cache, exist_ok=True)
        lm_ds.save_to_disk(tokenized_cache)

    if max_train_samples is not None:
        lm_ds["train"] = lm_ds["train"].select(range(min(max_train_samples, len(lm_ds["train"]))))
    if "validation" in lm_ds and max_eval_samples is not None:
        lm_ds["validation"] = lm_ds["validation"].select(range(min(max_eval_samples, len(lm_ds["validation"]))))
    return lm_ds


def load_or_build_tokenized_split(
    *,
    dataset_name: str,
    dataset_config: str,
    text_column: str,
    cache_dir: str | None,
    tokenized_cache: str | None,
    tokenizer,
    seq_len: int,
    num_proc: int,
    split: str,
    max_samples: int | None,
) -> Any:
    if tokenized_cache is not None and os.path.isdir(tokenized_cache):
        ds: DatasetDict = load_from_disk(tokenized_cache)
        split_ds = ds[split]
        if max_samples is not None:
            split_ds = split_ds.select(range(min(max_samples, len(split_ds))))
        return split_ds

    raw: DatasetDict = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    raw = raw.map(partial(normalize_text, text_column=text_column), desc="Normalize text")
    raw = raw.filter(lambda ex: bool(ex["text"]), desc="Drop empty lines")
    remove_cols = list(raw["train"].features.keys())

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; please set a tokenizer with EOS.")

    tokenized = raw.map(
        partial(tokenize_batch, tokenizer=tokenizer, eos_token_id=eos_id),
        batched=True,
        remove_columns=remove_cols,
        num_proc=num_proc,
        desc="Tokenize",
    )
    lm_ds = tokenized.map(
        partial(group_texts, seq_len=seq_len),
        batched=True,
        num_proc=num_proc,
        desc="Group texts",
    )

    if tokenized_cache is not None:
        os.makedirs(tokenized_cache, exist_ok=True)
        lm_ds.save_to_disk(tokenized_cache)

    split_ds = lm_ds[split]
    if max_samples is not None:
        split_ds = split_ds.select(range(min(max_samples, len(split_ds))))
    return split_ds


def build_training_args(args: argparse.Namespace, *, fp16: bool, bf16: bool, has_validation: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=has_validation,
        eval_strategy="steps" if has_validation else "no",
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


def maybe_preflight_compile(
    *,
    enabled: bool,
    model,
    vocab_size: int,
    seq_len: int,
    fp16: bool,
    bf16: bool,
) -> None:
    if not enabled or not torch.cuda.is_available():
        return
    mp_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    print(
        f"[preflight] compiling kernels on {torch.cuda.get_device_name(0)} "
        f"(cc={torch.cuda.get_device_capability(0)}) dtype={mp_dtype} ...",
        flush=True,
    )
    tmp_model = model.to("cuda", dtype=mp_dtype).train()
    t = min(seq_len, 128)
    input_ids = torch.randint(0, vocab_size, (1, t), device="cuda")
    out = tmp_model(input_ids=input_ids, labels=input_ids, use_cache=False)
    out.loss.backward()
    torch.cuda.synchronize()
    tmp_model = tmp_model.to("cpu")
    del tmp_model
    model = model.to(dtype=torch.float32)
    print("[preflight] done.", flush=True)


def train_with_trainer(
    *,
    model,
    dataset: DatasetDict,
    train_args: TrainingArguments,
    resume_from_checkpoint: str | None,
    output_dir: str,
) -> None:
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=DefaultDataCollator(),
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    metrics = train_result.metrics
    if "train_loss" in metrics:
        metrics["train_ppl"] = math.exp(metrics["train_loss"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


@torch.no_grad()
def evaluate_ppl(model, dataloader, device: str) -> dict[str, float]:
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


def override_attr_on_modules(model, attr: str, value) -> int:
    changed = 0
    for module in model.modules():
        if hasattr(module, attr):
            setattr(module, attr, value)
            changed += 1
    return changed

