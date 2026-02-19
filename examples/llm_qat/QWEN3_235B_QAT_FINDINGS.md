# Qwen3-235B QAT on B200 Cluster — Findings

## Environment
- **GPU**: 8× NVIDIA B200 (183 GB each)
- **Driver**: 580.105.08
- **CUDA**: 12.8 (reported as 13.0 by nvidia-smi)
- **PyTorch**: 2.10.0+cu128
- **Model**: `parsed/gamma-qwen3-235b-instruct-matt-merged` (Qwen3MoE, 235B params, 128 experts, 94 layers)
- **Quantization**: NVFP4_DEFAULT_CFG via NVIDIA Model Optimizer
- **Training**: FSDP1 with `Qwen3MoeDecoderLayer` wrapping, bf16 mixed precision

## Issue 1: NCCL InfiniBand Failures (Multi-Node)

**Symptom**: Multi-node jobs (8 nodes × 8 GPUs) hang after NCCL initialization completes. All 7 worker nodes report thousands of IB RDMA errors:
```
NCCL WARN NET/IB : Got completion with error 12, opcode 0, len 0, vendor err 249
NCCL WARN NET/IB : Got completion with error 5, opcode 0, len 0, vendor err 129
NCCL WARN NET/IB : Got completion with error 12, opcode 0, len 0, vendor err 244
```

**Root Cause**: Cluster-wide InfiniBand fabric issue. Intra-node NVLink works fine; only cross-node IB RDMA fails. Reproduced consistently across Jobs 11 and 14.

**Diagnosis**: GPUs show 100% utilization but only 11.4 GB memory — this indicates spin-waiting on stuck NCCL operations, not real compute work.

**Workaround**: Run on single node to avoid IB entirely.

**Status**: Unresolved — requires infrastructure fix from cluster provider.

## Issue 2: cuBLAS bf16/fp16 GEMM Bug on B200 (CRITICAL)

**Symptom**: Every half-precision (bf16 and fp16) matrix multiplication fails with:
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasGemmEx(...)
```

**Root Cause**: The default cuBLAS backend is broken on B200 with CUDA 12.8/driver 580.x. Even a trivial 2×2 bf16 matmul fails. fp32 works fine.

**Reproduction** (standalone, no model needed):
```python
import torch
a = torch.randn(2, 2, dtype=torch.bfloat16, device="cuda:0")
b = torch.randn(2, 2, dtype=torch.bfloat16, device="cuda:0")
c = a @ b  # FAILS with CUBLAS_STATUS_INVALID_VALUE
```

**Fix**: Switch to the cuBLASLt backend, which works correctly:
```python
torch.backends.cuda.preferred_blas_library("cublaslt")
```
This was added to `main.py` at the top, before any model loading.

**Verification**: All Qwen3-235B projection dimensions tested successfully with cublasLt at seq_len=32768 (see `test_cublas3.py`).

## Issue 3: OOM on Single Node with seq_len=32768

**Symptom**: After fixing cuBLAS, training OOMs during the first calibration forward pass at `scaled_dot_product_attention`:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
GPU 0 has a total capacity of 178.35 GiB of which 213.81 MiB is free.
```

**Root Cause**: The model + 109,983 quantizers consume ~170 GB per GPU, leaving only ~13 GB headroom. A forward pass with `seq_len=32768` through the attention mechanism (64 heads, head_dim=128) requires more than 13 GB for activation memory, even with activation checkpointing enabled.

**Memory breakdown per GPU (FSDP FULL_SHARD, 8 GPUs)**:
- Sharded model params + quantizers: ~170 GB
- Available for activations: ~13 GB
- Attention scores for one layer at seq=32768: `64 heads × 32768 × 32768 × 2 bytes ≈ 8.6 GB`
- Plus Q/K/V projections, MoE routing, etc.

**Status**: Unresolved. Possible solutions:
1. **Reduce `max_seq_length`** — e.g., 8192 should fit in 13 GB headroom
2. **Multi-node** — distributing shards across more GPUs reduces per-GPU memory, but requires fixing the IB issue
3. **CPU offloading** — `fsdp_offload_params: true` could free GPU memory, at the cost of speed

## Files Modified

- **`main.py`**: Added `torch.backends.cuda.preferred_blas_library("cublaslt")` workaround for cuBLAS B200 bug
- **`slurm_qat.sh`**: No permanent changes (kept at `--nodes=8` for multi-node use)

## Test Scripts Created

- `test_fsdp_forward.py` — Demonstrates cuBLAS failure with model's q_proj
- `test_cublas.py` — Dimension sweep showing all bf16 GEMMs fail on B200
- `test_cublas2.py` — Discovers cublasLt workaround, tests alternatives
- `test_cublas3.py` — Verifies cublasLt with all Qwen3-235B projection dimensions

## Next Steps

1. **Immediate**: Retry training with a reduced `max_seq_length` (e.g., 8192) on single node
2. **Short-term**: Get IB fabric fixed by cluster provider, then run multi-node with `--nodes=8` and full `max_seq_length=32768`
3. **Nice-to-have**: File a PyTorch/NVIDIA bug report for the cuBLAS bf16 regression on B200 with driver 580.x
