#!/usr/bin/env python3
"""
通过 LM Studio API 测速脚本
测试 prompt eval (prefill) 和 eval (decode) 速度
"""

import time
import requests
import json

API_BASE = "http://127.0.0.1:1234/v1"

def test_speed():
    print("=" * 60)
    print("LM Studio API 测速")
    print("=" * 60)

    # 测试连接
    try:
        resp = requests.get(f"{API_BASE}/models", timeout=5)
        print(f"\n已连接到 LM Studio API")
        models = resp.json()
        if 'data' in models:
            for m in models['data']:
                print(f"  已加载模型: {m.get('id', 'unknown')}")
    except Exception as e:
        print(f"无法连接到 API: {e}")
        return

    # 测试 prompt eval (prefill) 速度
    print("\n" + "=" * 60)
    print("测试 Prompt Eval (Prefill) 速度")
    print("=" * 60)

    test_prompts = [
        ("短 prompt (~50 tokens)", "请用一句话解释什么是人工智能。"),
        ("中等 prompt (~200 tokens)", """
请详细解释以下概念：
1. 什么是深度学习？
2. 什么是神经网络？
3. 什么是transformer架构？
4. 什么是注意力机制？
请用中文回答每个问题。
""".strip()),
        ("长 prompt (~500 tokens)", """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的表示。
Transformer架构是2017年由Google提出的一种神经网络架构，它使用自注意力机制来处理序列数据。
注意力机制允许模型在处理输入时关注最相关的部分。
大型语言模型（LLM）是基于Transformer架构的深度学习模型，它们通过在海量文本数据上进行预训练来学习语言的模式和知识。

请总结上述内容的主要观点，并说明这些技术之间的关系。
""".strip()),
    ]

    prompt_results = []

    for name, prompt in test_prompts:
        # 预热
        _ = requests.post(f"{API_BASE}/completions", json={
            "model": "gemma-4-31b-it",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False
        }, timeout=60)

        # 正式测试
        start = time.time()
        resp = requests.post(f"{API_BASE}/completions", json={
            "model": "gemma-4-31b-it",
            "prompt": prompt,
            "max_tokens": 10,
            "temperature": 0,
            "stream": False
        }, timeout=120)
        elapsed = time.time() - start

        data = resp.json()
        prompt_tokens = data.get('usage', {}).get('prompt_tokens', 0)
        completion_tokens = data.get('usage', {}).get('completion_tokens', 0)

        print(f"\n{name}:")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Completion tokens: {completion_tokens}")
        print(f"  时间: {elapsed:.3f}s")
        if prompt_tokens > 0 and elapsed > 0:
            tps = prompt_tokens / elapsed
            print(f"  Prompt Eval 速度: {tps:.1f} tokens/s")
            prompt_results.append((name, prompt_tokens, elapsed, tps))

    # 测试 eval (decode/generation) 速度
    print("\n" + "=" * 60)
    print("测试 Eval (Decode/Generation) 速度")
    print("=" * 60)

    test_prompt = "从前有座山，山里有座庙，"
    gen_lengths = [64, 128, 256, 512]
    eval_results = []

    for max_tokens in gen_lengths:
        # 预热
        _ = requests.post(f"{API_BASE}/completions", json={
            "model": "gemma-4-31b-it",
            "prompt": test_prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False
        }, timeout=60)

        # 正式测试
        start = time.time()
        resp = requests.post(f"{API_BASE}/completions", json={
            "model": "gemma-4-31b-it",
            "prompt": test_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }, timeout=120)
        elapsed = time.time() - start

        data = resp.json()
        completion_tokens = data.get('usage', {}).get('completion_tokens', 0)

        print(f"\n生成 {max_tokens} tokens:")
        print(f"  实际生成: {completion_tokens} tokens")
        print(f"  时间: {elapsed:.3f}s")
        if completion_tokens > 0 and elapsed > 0:
            tps = completion_tokens / elapsed
            print(f"  Eval 速度: {tps:.1f} tokens/s")
            eval_results.append((max_tokens, completion_tokens, elapsed, tps))

    # 打印汇总
    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)

    print("\nPrompt Eval (Prefill) 速度:")
    print("-" * 50)
    for name, tokens, elapsed, tps in prompt_results:
        print(f"  {name}: {tps:.1f} tokens/s ({tokens} tokens in {elapsed:.3f}s)")

    print("\nEval (Decode) 速度:")
    print("-" * 50)
    for max_tokens, generated, elapsed, tps in eval_results:
        print(f"  生成 {max_tokens} tokens: {tps:.1f} tokens/s ({generated} tokens in {elapsed:.3f}s)")

    if eval_results:
        avg_eval_tps = sum(r[3] for r in eval_results) / len(eval_results)
        print(f"\n平均 Eval 速度: {avg_eval_tps:.1f} tokens/s")

if __name__ == "__main__":
    test_speed()
