#!/usr/bin/env python
"""
准备 wikitext-103 数据集的 tokenized cache
用于 KDA 和 MKDA 训练脚本
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="准备 tokenized cache 用于 wikitext-103 训练")
    
    p.add_argument(
        "--tokenizer", 
        type=str, 
        default="gpt2", 
        help="Tokenizer 名称或本地路径"
    )
    p.add_argument(
        "--dataset_name", 
        type=str, 
        default="wikitext", 
        help="HF datasets 名称"
    )
    p.add_argument(
        "--dataset_config", 
        type=str, 
        default="wikitext-103-raw-v1", 
        help="HF datasets 配置"
    )
    p.add_argument(
        "--text_column", 
        type=str, 
        default="text",
        help="文本列名"
    )
    p.add_argument(
        "--cache_dir", 
        type=str, 
        default=None, 
        help="HF datasets cache_dir"
    )
    p.add_argument(
        "--tokenized_cache", 
        type=str, 
        default="./data/wikitext103_gpt2_1024", 
        help="保存 tokenized dataset 的路径"
    )
    p.add_argument(
        "--seq_len", 
        type=int, 
        default=1024,
        help="序列长度"
    )
    p.add_argument(
        "--num_proc", 
        type=int, 
        default=8,
        help="处理数据的进程数"
    )
    
    return p.parse_args()


def _normalize_text(example: dict[str, Any], text_column: str) -> dict[str, Any]:
    """规范化文本"""
    text = example.get(text_column, "")
    if text is None:
        text = ""
    return {"text": text.strip()}


def _tokenize_batch(
    examples: dict[str, list[Any]], 
    tokenizer, 
    eos_token_id: int
) -> dict[str, list[list[int]]]:
    """批量tokenize文本"""
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


def _group_texts(
    examples: dict[str, list[list[int]]], 
    seq_len: int
) -> dict[str, list[list[int]]]:
    """将文本分组为固定长度的序列"""
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


def main() -> None:
    args = parse_args()
    
    # 检查是否已存在
    if os.path.isdir(args.tokenized_cache):
        print(f"⚠️  Tokenized cache 已存在: {args.tokenized_cache}")
        print("如果要重新生成，请先删除该目录。")
        return
    
    print(f"📥 加载 tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer 没有 eos_token_id，请设置一个带 EOS 的 tokenizer。")
    
    print(f"📥 加载数据集: {args.dataset_name} ({args.dataset_config})")
    print("⚠️  强制重新下载以避免缓存问题...")
    raw: DatasetDict = load_dataset(
        args.dataset_name,
        args.dataset_config,
        cache_dir=args.cache_dir,
        download_mode="force_redownload"
    )
    
    print("🔧 规范化文本...")
    raw = raw.map(
        partial(_normalize_text, text_column=args.text_column), 
        desc="规范化文本"
    )
    
    print("🧹 过滤空行...")
    raw = raw.filter(lambda ex: bool(ex["text"]), desc="过滤空行")
    
    remove_cols = list(raw["train"].features.keys())
    
    print(f"🔤 Tokenize 文本 (使用 {args.num_proc} 进程)...")
    tokenized = raw.map(
        partial(_tokenize_batch, tokenizer=tokenizer, eos_token_id=eos_id),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=remove_cols,
        desc="Tokenize",
    )
    
    print(f"📦 将文本分组为 {args.seq_len} token 的序列...")
    lm_ds = tokenized.map(
        partial(_group_texts, seq_len=args.seq_len),
        batched=True,
        num_proc=args.num_proc,
        desc=f"分组文本 (seq_len={args.seq_len})",
    )
    
    print(f"\n📊 数据集统计:")
    for split_name, split_data in lm_ds.items():
        print(f"  - {split_name}: {len(split_data)} 样本")
    
    print(f"\n💾 保存 tokenized cache 到: {args.tokenized_cache}")
    os.makedirs(args.tokenized_cache, exist_ok=True)
    lm_ds.save_to_disk(args.tokenized_cache)
    
    print(f"\n✅ 完成！tokenized cache 已保存到: {args.tokenized_cache}")
    print(f"\n使用方法:")
    print(f"  python examples/train_kda_wikitext103_epochs20.py \\")
    print(f"    --tokenized_cache {args.tokenized_cache}")


if __name__ == "__main__":
    main()
