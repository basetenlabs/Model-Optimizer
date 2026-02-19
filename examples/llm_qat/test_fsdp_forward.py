"""Quick test: can FSDP do a forward pass on Qwen3-235B with seq_len=32768?"""
import torch
import transformers
import os

# Test 1: Simple bf16 GEMM on B200 with q_proj dimensions
print("Test 1: cuBLAS bf16 GEMM with q_proj dimensions (4096 x 8192, seq=32768)")
a = torch.randn(32768, 4096, dtype=torch.bfloat16, device="cuda:0")
b = torch.randn(8192, 4096, dtype=torch.bfloat16, device="cuda:0")
try:
    c = torch.nn.functional.linear(a, b)
    print(f"  PASSED: output shape {c.shape}")
    del a, b, c
except Exception as e:
    print(f"  FAILED: {e}")
    del a, b

torch.cuda.empty_cache()

# Test 2: Load just the first layer and test q_proj directly
print("\nTest 2: Loading model and testing q_proj on single GPU...")
model = transformers.AutoModelForCausalLM.from_pretrained(
    "parsed/gamma-qwen3-235b-instruct-matt-merged",
    torch_dtype=torch.bfloat16,
)
q_proj = model.model.layers[0].self_attn.q_proj.to("cuda:0")
print(f"  q_proj weight shape: {q_proj.weight.shape}, dtype: {q_proj.weight.dtype}")

x = torch.randn(1, 32768, 4096, dtype=torch.bfloat16, device="cuda:0")
try:
    out = q_proj(x)
    print(f"  PASSED: output shape {out.shape}")
    del out
except Exception as e:
    print(f"  FAILED: {e}")

del x, q_proj, model
torch.cuda.empty_cache()
print("\nDone.")
