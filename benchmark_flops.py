#!/usr/bin/env python3
"""
GPU 算力测试 - 测试不同精度下的实际 FLOPS
支持 FP32, BF16, FP16, FP8, FP4 (通过量化模拟)
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Tuple, List
import os

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_gpu_info():
    """获取 GPU 信息"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        compute_capability = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
        return gpu_name, gpu_memory, compute_capability
    return "Unknown", 0, (0, 0)

def benchmark_matmul(dtype: torch.dtype, m: int, k: int, n: int,
                     warmup: int = 10, iterations: int = 100,
                     use_compile: bool = True) -> Tuple[float, float]:
    """
    测试矩阵乘法性能

    Args:
        dtype: 数据类型
        m, k, n: 矩阵维度 (M x K) @ (K x N) = (M x N)
        warmup: 预热次数
        iterations: 测试次数
        use_compile: 是否使用 torch.compile

    Returns:
        (TFLOPS, 实际计算时间ms)
    """
    device = torch.device('cuda')

    # 创建输入矩阵
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)

    # 定义计算函数
    def matmul_op(a, b):
        return torch.matmul(a, b)

    if use_compile:
        # 使用 torch.compile 优化
        matmul_op = torch.compile(matmul_op, mode="max-autotune")

    # 预热
    for _ in range(warmup):
        c = matmul_op(a, b)
    torch.cuda.synchronize()

    # 正式测试
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        c = matmul_op(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iterations

    # 计算 FLOPS: 矩阵乘法需要 2*m*k*n 次浮点运算
    flops = 2 * m * k * n
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    return tflops, elapsed_ms

def benchmark_linear_layer(dtype: torch.dtype, batch_size: int, in_features: int,
                          out_features: int, warmup: int = 10,
                          iterations: int = 100, use_compile: bool = True) -> Tuple[float, float]:
    """
    测试 Linear 层性能
    """
    device = torch.device('cuda')

    # 创建模型
    linear = nn.Linear(in_features, out_features, bias=False).to(device).to(dtype)

    # 创建输入
    x = torch.randn(batch_size, in_features, dtype=dtype, device=device)

    if use_compile:
        linear = torch.compile(linear, mode="max-autotune")

    # 预热
    for _ in range(warmup):
        y = linear(x)
    torch.cuda.synchronize()

    # 测试
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        y = linear(x)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iterations

    # FLOPS: 2 * batch_size * in_features * out_features
    flops = 2 * batch_size * in_features * out_features
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    return tflops, elapsed_ms

def benchmark_attention(dtype: torch.dtype, batch_size: int, seq_len: int,
                       num_heads: int, head_dim: int, warmup: int = 10,
                       iterations: int = 50, use_compile: bool = True) -> Tuple[float, float]:
    """
    测试 Attention 性能 (简化版，不含 mask)
    """
    device = torch.device('cuda')
    hidden_dim = num_heads * head_dim

    # Q, K, V 投影
    qkv = torch.randn(batch_size, seq_len, 3 * hidden_dim, dtype=dtype, device=device)

    def attention_forward(qkv):
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return attn_output

    if use_compile:
        attention_forward = torch.compile(attention_forward, mode="max-autotune")

    # 预热
    for _ in range(warmup):
        out = attention_forward(qkv)
    torch.cuda.synchronize()

    # 测试
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        out = attention_forward(qkv)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iterations

    # FLOPS for attention (approximate)
    # Q@K^T: batch * heads * seq * seq * head_dim * 2
    # Attn@V: batch * heads * seq * seq * head_dim * 2
    # Softmax and other ops are smaller
    flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    return tflops, elapsed_ms

def simulate_fp4_quantization(tensor: torch.Tensor) -> torch.Tensor:
    """
    模拟 FP4 量化效果 (实际计算仍用更高精度)
    FP4 格式: 1 sign bit, 2 exponent bits, 1 mantissa bit
    动态范围: 2^-6 到 2^4 (约 0.015625 到 16)
    """
    # FP4 的可表示值 (E2M1 格式)
    # 实际 FP4 有 16 个可表示值
    fp4_values = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=tensor.dtype, device=tensor.device)

    # 找最近的 FP4 值
    original_shape = tensor.shape
    tensor_flat = tensor.flatten().unsqueeze(1)

    # 计算距离
    distances = torch.abs(tensor_flat - fp4_values.unsqueeze(0))
    nearest_idx = torch.argmin(distances, dim=1)

    quantized = fp4_values[nearest_idx].view(original_shape)
    return quantized

def simulate_fp8_quantization(tensor: torch.Tensor, format: str = 'e4m3') -> torch.Tensor:
    """
    模拟 FP8 量化效果
    E4M3: 1 sign, 4 exponent, 3 mantissa (范围更大，精度较低)
    E5M2: 1 sign, 5 exponent, 2 mantissa (范围更大，精度更低)
    """
    if format == 'e4m3':
        # FP8 E4M3 范围约 [-448, 448]
        max_val = 448.0
    else:  # e5m2
        # FP8 E5M2 范围约 [-57344, 57344]
        max_val = 57344.0

    # 简化的量化模拟
    scale = max_val / tensor.abs().max().clamp(min=1e-8)
    quantized = (tensor * scale).round() / scale
    return quantized.clamp(-max_val, max_val)

def run_matmul_benchmark(dtype_name: str, m: int, k: int, n: int,
                         use_compile: bool = True) -> dict:
    """运行单个精度的 MatMul benchmark"""

    dtype_map = {
        'FP32': torch.float32,
        'FP16': torch.float16,
        'BF16': torch.bfloat16,
    }

    # 清理缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if dtype_name == 'FP8':
        # FP8 使用 BF16 计算，模拟量化
        dtype = torch.bfloat16
        device = torch.device('cuda')
        a = torch.randn(m, k, dtype=dtype, device=device)
        b = torch.randn(k, n, dtype=dtype, device=device)

        # 模拟 FP8 量化
        a = simulate_fp8_quantization(a, 'e4m3')
        b = simulate_fp8_quantization(b, 'e4m3')

        def matmul_op(a, b):
            return torch.matmul(a, b)

        if use_compile:
            matmul_op = torch.compile(matmul_op, mode="max-autotune")

        # 预热
        for _ in range(10):
            c = matmul_op(a, b)
        torch.cuda.synchronize()

        # 测试
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            c = matmul_op(a, b)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / 100
        flops = 2 * m * k * n
        tflops = flops / (elapsed_ms * 1e-3) / 1e12
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    elif dtype_name == 'FP4':
        # FP4 模拟
        dtype = torch.bfloat16
        device = torch.device('cuda')
        a = torch.randn(m, k, dtype=dtype, device=device)
        b = torch.randn(k, n, dtype=dtype, device=device)

        # 模拟 FP4 量化
        a = simulate_fp4_quantization(a)
        b = simulate_fp4_quantization(b)

        def matmul_op(a, b):
            return torch.matmul(a, b)

        if use_compile:
            matmul_op = torch.compile(matmul_op, mode="max-autotune")

        # 预热
        for _ in range(10):
            c = matmul_op(a, b)
        torch.cuda.synchronize()

        # 测试
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            c = matmul_op(a, b)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / 100
        flops = 2 * m * k * n
        tflops = flops / (elapsed_ms * 1e-3) / 1e12
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    else:
        dtype = dtype_map[dtype_name]
        tflops, elapsed_ms = benchmark_matmul(dtype, m, k, n, use_compile=use_compile)
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        'dtype': dtype_name,
        'tflops': tflops,
        'time_ms': elapsed_ms,
        'memory_gb': memory_gb
    }

def main():
    print("=" * 70)
    print("GPU 算力测试 - 不同精度下的 FLOPS")
    print("=" * 70)

    # GPU 信息
    gpu_name, gpu_memory, compute_cap = get_gpu_info()
    print(f"\nGPU: {gpu_name}")
    print(f"显存: {gpu_memory:.1f} GB")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"torch.compile: 启用 (mode=max-autotune)")

    # 测试矩阵维度
    # 大矩阵更能反映峰值性能
    test_configs = [
        # (M, K, N) - GEMM 尺寸
        (4096, 4096, 4096),    # 中等尺寸
        (8192, 8192, 8192),    # 大尺寸
        (16384, 16384, 16384), # 超大尺寸 (如果显存够)
    ]

    # 数据类型列表
    dtypes = ['FP32', 'FP16', 'BF16', 'FP8', 'FP4']

    results = []

    for m, k, n in test_configs:
        # 估算显存需求 (FP32 最坏情况)
        est_memory = (m * k + k * n + m * n) * 4 / 1024**3 * 3  # 预留空间

        if est_memory > gpu_memory * 0.8:
            print(f"\n跳过 {m}x{k}x{n} (预估需要 {est_memory:.1f}GB)")
            continue

        print(f"\n{'=' * 70}")
        print(f"矩阵尺寸: M={m}, K={k}, N={n} (总计算量: {2*m*k*n/1e12:.2f} TFLOPs)")
        print(f"{'=' * 70}")
        print(f"{'精度':<8} {'TFLOPS':<12} {'时间(ms)':<12} {'显存(GB)':<10} {'相对FP32':<10}")
        print("-" * 70)

        config_results = []

        for dtype_name in dtypes:
            try:
                result = run_matmul_benchmark(dtype_name, m, k, n, use_compile=True)
                config_results.append(result)

                # 计算相对 FP32 的加速比
                if len(config_results) > 0 and config_results[0]['dtype'] == 'FP32':
                    speedup = result['tflops'] / config_results[0]['tflops']
                else:
                    speedup = 1.0

                print(f"{dtype_name:<8} {result['tflops']:<12.2f} {result['time_ms']:<12.3f} {result['memory_gb']:<10.2f} {speedup:<10.2f}x")

            except Exception as e:
                print(f"{dtype_name:<8} 错误: {str(e)[:50]}")

        results.append({
            'config': (m, k, n),
            'results': config_results
        })

    # 额外测试: Attention 性能
    print(f"\n{'=' * 70}")
    print("Attention 层性能测试")
    print(f"{'=' * 70}")

    attn_configs = [
        (1, 4096, 32, 128),   # 单请求, 4K context, 32 heads, 128 dim
        (1, 8192, 32, 128),   # 8K context
        (4, 2048, 32, 128),   # 4 并发, 2K context
    ]

    for batch, seq_len, heads, head_dim in attn_configs:
        print(f"\n配置: batch={batch}, seq_len={seq_len}, heads={heads}, head_dim={head_dim}")

        for dtype_name in ['FP32', 'FP16', 'BF16']:
            dtype_map = {'FP32': torch.float32, 'FP16': torch.float16, 'BF16': torch.bfloat16}
            try:
                tflops, time_ms = benchmark_attention(
                    dtype_map[dtype_name], batch, seq_len, heads, head_dim, use_compile=True
                )
                print(f"  {dtype_name}: {tflops:.2f} TFLOPS, {time_ms:.3f}ms")
            except Exception as e:
                print(f"  {dtype_name}: 错误 - {str(e)[:30]}")

    # 总结
    print(f"\n{'=' * 70}")
    print("性能总结")
    print(f"{'=' * 70}")

    if results:
        # 找到最佳配置
        best_result = max(results[-1]['results'], key=lambda x: x['tflops'])
        print(f"\n峰值算力: {best_result['tflops']:.2f} TFLOPS ({best_result['dtype']})")

        # 理论峰值对比 (RTX 6000 Ada 约 91 TFLOPS FP32, Blackwell 更高)
        # Blackwell RTX PRO 6000 理论峰值约:
        # FP32: ~91 TFLOPS
        # FP16/BF16: ~181 TFLOPS (2x)
        # FP8: ~362 TFLOPS (4x)
        # FP4: ~724 TFLOPS (8x) - 通过 Tensor Core

        print("\n理论峰值参考 (Blackwell RTX PRO 6000):")
        print("  FP32:  ~91 TFLOPS")
        print("  FP16/BF16: ~181 TFLOPS (Tensor Core)")
        print("  FP8:  ~362 TFLOPS (Tensor Core)")
        print("  FP4:  ~724 TFLOPS (Tensor Core, 需要 NVFP4 支持)")

        print("\n注意:")
        print("  - FP8/FP4 测试使用 BF16 模拟量化，实际算力需原生 kernel 支持")
        print("  - 实际性能受 memory bandwidth、kernel 优化等影响")
        print("  - Blackwell 架构对 FP4 有原生支持 (NVFP4)")

if __name__ == "__main__":
    main()
