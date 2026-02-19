"""Verify cublasLt workaround with realistic dimensions."""
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

# Test all attention projections with seq=32768
dims = [
    ("q_proj", 32768, 8192, 4096),
    ("k_proj", 32768, 512, 4096),
    ("v_proj", 32768, 512, 4096),
    ("o_proj", 32768, 4096, 8192),
    ("gate_proj", 32768, 1536, 4096),
    ("up_proj", 32768, 1536, 4096),
    ("down_proj", 32768, 4096, 1536),
]

print("Testing with cublasLt backend, seq_len=32768, bf16:")
all_ok = True
for name, m, n, k in dims:
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda:0")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda:0")
    try:
        c = torch.nn.functional.linear(a, w)
        print(f"  {name:>10} [{m},{k}]x[{n},{k}] -> [{m},{n}]: OK")
        del c
    except Exception as e:
        print(f"  {name:>10}: FAIL: {e}")
        all_ok = False
    del a, w
    torch.cuda.empty_cache()

print(f"\n{'All tests passed!' if all_ok else 'Some tests FAILED'}")
