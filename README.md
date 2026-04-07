参考：RTX Pro 6000 96GB torch 2.x性能
| 精度 | 实测 TFLOPS (16384³) | 相对 FP32 加速 | 理论峰值 |

|------|---------------------|---------------|---------|

| FP32 | 63 | 1.0x | ~91 |

| FP16 | 238 | 3.8x | ~181 |

| BF16 | 287 | 4.6x | ~181 |

| FP8 | 286 | 4.5x | ~362 |

| FP4 | 326 | 5.2x | ~724 |


关键发现

FP32 达到理论峰值的 69% - 63/91 TFLOPS，说明 torch.compile 优化效果不错

BF16 表现最佳 - 287 TFLOPS，超过理论 FP16 峰值，可能 Blackwell 对 BF16 Tensor Core 有特别优化

FP8/FP4 受限于模拟 - 测试用 BF16 模拟量化，实际计算仍是 BF16，所以没有达到理论 2x/4x 加速

原生 NVFP4 kernel 未调用 - 需要 cuBLAS/cuDNN 的原生 FP4 GEMM 支持，当前 PyTorch 还没有
