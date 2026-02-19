"""Test which bf16 GEMM dimensions trigger CUBLAS_STATUS_INVALID_VALUE on B200."""
import torch

def test_gemm(m, n, k):
    """Test F.linear with input [m, k] and weight [n, k] -> output [m, n]"""
    try:
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda:0")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda:0")
        c = torch.nn.functional.linear(a, w)
        del a, w, c
        return "OK"
    except Exception as e:
        del a, w
        torch.cuda.empty_cache()
        return f"FAIL"

# Test q_proj: input [seq, 4096] @ weight [8192, 4096]
print("=== Testing q_proj dimensions [seq, 4096] x [8192, 4096] ===")
for seq in [1024, 2048, 4096, 8192, 16384, 32768]:
    r = test_gemm(seq, 8192, 4096)
    print(f"  seq={seq:>6}: {r}")

# Test with fp32 for comparison
print("\n=== Testing fp32 with seq=32768 ===")
a = torch.randn(32768, 4096, dtype=torch.float32, device="cuda:0")
w = torch.randn(8192, 4096, dtype=torch.float32, device="cuda:0")
try:
    c = torch.nn.functional.linear(a, w)
    print(f"  fp32: OK, shape {c.shape}")
except Exception as e:
    print(f"  fp32: FAIL: {e}")
del a, w
torch.cuda.empty_cache()

# Test with fp16 for comparison
print("\n=== Testing fp16 with seq=32768 ===")
a = torch.randn(32768, 4096, dtype=torch.float16, device="cuda:0")
w = torch.randn(8192, 4096, dtype=torch.float16, device="cuda:0")
try:
    c = torch.nn.functional.linear(a, w)
    print(f"  fp16: OK, shape {c.shape}")
except Exception as e:
    print(f"  fp16: FAIL: {e}")
del a, w
torch.cuda.empty_cache()

# Check if total elements matters: m*n*k = 32768*8192*4096 = 1.1T
print("\n=== Testing other large bf16 GEMMs ===")
for m, n, k in [(32768, 4096, 4096), (32768, 8192, 2048), (65536, 4096, 4096), (16384, 8192, 8192)]:
    r = test_gemm(m, n, k)
    print(f"  [{m}, {k}] x [{n}, {k}] -> [{m}, {n}]: {r}")
