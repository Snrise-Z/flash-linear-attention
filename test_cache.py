#!/usr/bin/env python
"""测试 tokenized cache 是否可以正确加载"""

from datasets import load_from_disk

cache_path = "./data/wikitext103_gpt2_1024"

print(f"📂 加载 tokenized cache: {cache_path}")
dataset = load_from_disk(cache_path)

print(f"\n✅ 成功加载！数据集信息:")
print(f"  数据集类型: {type(dataset)}")
print(f"  分割: {list(dataset.keys())}")

for split_name, split_data in dataset.items():
    print(f"\n  {split_name}:")
    print(f"    - 样本数: {len(split_data)}")
    print(f"    - 特征: {split_data.features}")
    if len(split_data) > 0:
        example = split_data[0]
        print(f"    - 第一个样本的 input_ids 长度: {len(example['input_ids'])}")
        print(f"    - 第一个样本的 labels 长度: {len(example['labels'])}")

print("\n✅ tokenized cache 验证成功！")
