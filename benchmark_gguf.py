#!/usr/bin/env python3
"""
GGUF 模型测速脚本
测试 prompt eval (prefill) 和 eval (decode) 速度
"""

import time
import argparse
from llama_cpp import Llama

def benchmark_model(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):
    print(f"\n{'='*60}")
    print(f"加载模型: {model_path}")
    print(f"{'='*60}")

    # 加载模型
    start_load = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,  # -1 表示全部加载到 GPU
        n_ctx=n_ctx,
        verbose=True
    )
    load_time = time.time() - start_load
    print(f"\n模型加载时间: {load_time:.2f}s")

    # 测试 prompt eval (prefill) 速度
    print(f"\n{'='*60}")
    print("测试 Prompt Eval (Prefill) 速度")
    print(f"{'='*60}")

    # 不同长度的 prompt 测试
    test_prompts = [
        ("短 prompt (约50 tokens)", "请用一句话解释什么是人工智能。"),
        ("中等 prompt (约200 tokens)", """
请详细解释以下概念：
1. 什么是深度学习？
2. 什么是神经网络？
3. 什么是transformer架构？
4. 什么是注意力机制？
请用中文回答每个问题，每个问题回答2-3句话。
""".strip()),
        ("长 prompt (约500 tokens)", """
阅读以下文章并总结要点：

人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
这些任务包括视觉感知、语音识别、决策制定和语言翻译等。

深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的表示。
深度学习在图像识别、自然语言处理和语音识别等领域取得了重大突破。

Transformer架构是2017年由Google提出的一种神经网络架构，它使用自注意力机制来处理序列数据。
Transformer已经成为自然语言处理领域的主流架构，被广泛应用于各种大型语言模型中。

注意力机制允许模型在处理输入时关注最相关的部分。
这种机制使得模型能够更好地捕捉长距离依赖关系，提高了模型的表现。

大型语言模型（LLM）是基于Transformer架构的深度学习模型，它们通过在海量文本数据上进行预训练来学习语言的模式和知识。
这些模型在各种自然语言处理任务中表现出色，包括文本生成、翻译、摘要和问答等。

请总结上述内容的主要观点，并说明这些技术之间的关系。
""".strip()),
    ]

    prompt_eval_results = []

    for name, prompt in test_prompts:
        # 预热
        _ = llm(prompt, max_tokens=1, temperature=0)

        # 正式测试
        start = time.time()
        output = llm(prompt, max_tokens=10, temperature=0)
        elapsed = time.time() - start

        # 获取 prompt tokens 数量
        prompt_tokens = len(llm.tokenize(prompt.encode()))

        print(f"\n{name}:")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  时间: {elapsed:.3f}s")
        if elapsed > 0:
            tps = prompt_tokens / elapsed
            print(f"  Prompt Eval 速度: {tps:.1f} tokens/s")
            prompt_eval_results.append((name, prompt_tokens, elapsed, tps))

    # 测试 eval (decode/generation) 速度
    print(f"\n{'='*60}")
    print("测试 Eval (Decode/Generation) 速度")
    print(f"{'='*60}")

    # 使用固定 prompt 测试生成速度
    test_prompt = "从前有座山，山里有座庙，"
    gen_lengths = [64, 128, 256, 512]

    eval_results = []

    for max_tokens in gen_lengths:
        # 预热
        _ = llm(test_prompt, max_tokens=1, temperature=0)

        # 正式测试
        start = time.time()
        output = llm(test_prompt, max_tokens=max_tokens, temperature=0.7)
        elapsed = time.time() - start

        generated_tokens = output['usage']['completion_tokens']

        print(f"\n生成 {max_tokens} tokens:")
        print(f"  实际生成: {generated_tokens} tokens")
        print(f"  时间: {elapsed:.3f}s")
        if elapsed > 0:
            tps = generated_tokens / elapsed
            print(f"  Eval 速度: {tps:.1f} tokens/s")
            eval_results.append((max_tokens, generated_tokens, elapsed, tps))

    # 打印汇总
    print(f"\n{'='*60}")
    print("性能汇总")
    print(f"{'='*60}")

    print("\nPrompt Eval (Prefill) 速度:")
    print("-" * 50)
    for name, tokens, elapsed, tps in prompt_eval_results:
        print(f"  {name}: {tps:.1f} tokens/s ({tokens} tokens in {elapsed:.3f}s)")

    print("\nEval (Decode) 速度:")
    print("-" * 50)
    for max_tokens, generated, elapsed, tps in eval_results:
        print(f"  生成 {max_tokens} tokens: {tps:.1f} tokens/s ({generated} tokens in {elapsed:.3f}s)")

    # 计算平均 eval 速度
    if eval_results:
        avg_eval_tps = sum(r[3] for r in eval_results) / len(eval_results)
        print(f"\n平均 Eval 速度: {avg_eval_tps:.1f} tokens/s")

    return llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGUF 模型测速")
    parser.add_argument("model_path", help="GGUF 模型文件路径")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU 层数 (-1 全部)")
    parser.add_argument("--n-ctx", type=int, default=4096, help="上下文长度")

    args = parser.parse_args()

    llm = benchmark_model(args.model_path, args.n_gpu_layers, args.n_ctx)
