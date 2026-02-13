import torch
import triton
import triton.language as tl
import json
from pathlib import Path
from torch.utils.cpp_extension import load
import time

# # Load CUDA extension
# cuda_module = load(
#     name="multi_head_attention_cuda",
#     sources=[str(Path(__file__).parent / "cuda_kernel.cu")],
#     extra_cuda_cflags=["-O3"],
#     verbose=False
# )


@triton.jit
def multi_head_attention_triton(output, x, Wq, Wk, Wv, 
    B: tl.constexpr, C: tl.constexpr, F: tl.constexpr):
    pass



def multi_head_attention_triton_wrapper(output, x, B, C, F):
    grid = lambda meta: (B, C)
    multi_head_attention_triton[grid](output, x, B, C, F)
    return output


# def multi_head_attention_cuda_wrapper(output, x, gamma, epsilon, B, C, F):
#     return cuda_module.multi_head_attention_cuda(output, x, gamma, epsilon, B, C, F)


def pytorch_baseline(x, Wq, Wk, Wv, num_heads):
    # X is (B, seq_len, embed_dim = d_k * num_heads)
    B, seq_len, embed_dim = x.shape
    d_k = embed_dim // num_heads
    q = x @ Wq # (B, seq_len, embed_dim)
    k = x @ Wk # (B, seq_len, embed_dim)
    v = x @ Wv # (B, seq_len, embed_dim)
    q = q.view(B, seq_len, num_heads, d_k).transpose(1, 2)
    k = k.view(B, seq_len, num_heads, d_k).transpose(1, 2)
    v = v.view(B, seq_len, num_heads, d_k).transpose(1, 2)
    attn = q @ k.transpose(-2, -1) # (B, num_heads, seq_len, seq_len)
    attn = attn / torch.sqrt(d_k)
    attn = torch.softmax(attn, dim=-1)
    output = attn @ v # (B, num_heads, seq_len, d_k)
    output = output.transpose(1, 2).view(B, seq_len, embed_dim)
    return output


def compare_kernels():
    num_repeats = 10
    with open(Path(__file__).parent / "inputs.json", "r") as f:
        input_configs = json.load(f)
    input_arrays = [None] * len(input_configs)
    
    epsilon = 1e-8
    for i, input_config in enumerate(input_configs):
        input_arrays[i] = {}
        for key, value in input_config.items():
            input_arrays[i][key] = torch.randn(*value["shape"]).to(torch.float32).cuda()
    
    for input_array in input_arrays:
        x = input_array["x"]
        gamma = input_array["gamma"]
        B, C, F = x.shape
        
        
        # PyTorch baseline
        start_time = time.time()
        for i in range(num_repeats):
            pytorch_result = pytorch_baseline(x)
            torch.cuda.synchronize()
        end_time = time.time()
        print(f"PyTorch baseline time: {end_time - start_time:.2f} seconds")
        
        # Triton kernel
        # Warmup: compile Triton kernel before timing
        warmup_output = torch.zeros_like(x)
        multi_head_attention_triton_wrapper(warmup_output, x, B, C, F)
        torch.cuda.synchronize()
        triton_output = torch.zeros_like(x)
        start_time = time.time()
        for i in range(num_repeats):
            triton_result = multi_head_attention_triton_wrapper(triton_output, x, B, C, F)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Triton kernel time: {end_time - start_time:.2f} seconds")
        triton_diff = torch.max(torch.abs(triton_result - pytorch_result)).item()
        assert torch.allclose(triton_result, pytorch_result, atol=1e-2, equal_nan=True)
        print(f"✓ Triton kernel passed for shape {x.shape} (max diff: {triton_diff:.2e})")
        
        # # CUDA kernel
        # cuda_output = torch.zeros_like(x)
        # start_time = time.time()
        # for i in range(num_repeats):
        #     cuda_result = multi_head_attention_cuda_wrapper(cuda_output, x, B, C, F)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"CUDA kernel time: {end_time - start_time:.2f} seconds")
        # cuda_diff = torch.max(torch.abs(cuda_result - pytorch_result)).item()
        # assert torch.allclose(cuda_result, pytorch_result, atol=1e-2, equal_nan=True)
        # print(f"✓ CUDA kernel passed for shape {x.shape} (max diff: {cuda_diff:.2e})")
        
        print(f"✓ All kernels match PyTorch baseline!\n")




if __name__ == "__main__":
    compare_kernels()