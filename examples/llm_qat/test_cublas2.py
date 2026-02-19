"""Test bf16 GEMM workarounds on B200."""
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"cuBLAS version: {torch.backends.cuda.preferred_blas_library()}")

# Test 1: Even tiny bf16 matmul?
print("\n=== Tiny bf16 matmul ===")
a = torch.randn(2, 2, dtype=torch.bfloat16, device="cuda:0")
b = torch.randn(2, 2, dtype=torch.bfloat16, device="cuda:0")
try:
    c = a @ b
    print(f"  2x2 bf16: OK")
except Exception as e:
    print(f"  2x2 bf16: FAIL")

# Test 2: torch.mm vs F.linear
print("\n=== torch.mm bf16 ===")
try:
    c = torch.mm(a, b)
    print(f"  torch.mm 2x2 bf16: OK")
except Exception as e:
    print(f"  torch.mm 2x2 bf16: FAIL")

# Test 3: Try different cuBLAS backends
print("\n=== Testing with different settings ===")

for backend in ['cublaslt', 'cublas']:
    torch.backends.cuda.preferred_blas_library(backend)
    try:
        a = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda:0")
        w = torch.randn(8192, 4096, dtype=torch.bfloat16, device="cuda:0")
        c = torch.nn.functional.linear(a, w)
        print(f"  preferred_blas_library={backend}: OK")
        del a, w, c
    except Exception as e:
        print(f"  preferred_blas_library={backend}: FAIL")
        try:
            del a, w
        except:
            pass
        torch.cuda.empty_cache()

# Test 4: Try with autocast disabled and explicit fp32 compute
print("\n=== bf16 input with fp32 compute (manual cast) ===")
a = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda:0")
w = torch.randn(8192, 4096, dtype=torch.bfloat16, device="cuda:0")
try:
    c = torch.nn.functional.linear(a.float(), w.float()).bfloat16()
    print(f"  cast to fp32 then back: OK, shape {c.shape}")
except Exception as e:
    print(f"  cast to fp32: FAIL: {e}")
del a, w
torch.cuda.empty_cache()

# Test 5: Try with torch.compile
print("\n=== torch.compile ===")
@torch.compile
def compiled_linear(x, w):
    return torch.nn.functional.linear(x, w)

try:
    a = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda:0")
    w = torch.randn(8192, 4096, dtype=torch.bfloat16, device="cuda:0")
    c = compiled_linear(a, w)
    print(f"  torch.compile bf16: OK")
except Exception as e:
    print(f"  torch.compile bf16: FAIL")
