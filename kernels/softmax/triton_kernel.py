import torch
import triton
import triton.language as tl
import json
from pathlib import Path
from torch.utils.cpp_extension import load
import time

# # Load CUDA extension
# cuda_module = load(
#     name="softmax_cuda",
#     sources=[str(Path(__file__).parent / "cuda_kernel.cu")],
#     extra_cuda_cflags=["-O3"],
#     verbose=False
# )


@triton.jit
def softmax_triton(output, x, B: tl.constexpr, C: tl.constexpr, F: tl.constexpr):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    x_ptr = x + (b_idx * C * F) + (c_idx * F)
    output_ptr = output + (b_idx * C * F) + (c_idx * F)


    x_feat = tl.load(x_ptr + tl.arange(0, F))
    x_max = tl.max(x_feat)
    x_exp = tl.exp(x_feat - x_max)
    x_sum = tl.sum(x_exp)
    result = x_exp / x_sum
    tl.store(output_ptr + tl.arange(0, F), result)


def softmax_triton_wrapper(output, x, B, C, F):
    grid = lambda meta: (B, C)
    softmax_triton[grid](output, x, B, C, F)
    return output


# def softmax_cuda_wrapper(output, x, gamma, epsilon, B, C, F):
#     return cuda_module.softmax_cuda(output, x, gamma, epsilon, B, C, F)


def pytorch_baseline(x):
    # X is (B, C, F)
    x_demax = x - x.max(axis=-1, keepdim=True).values
    x_exp = torch.exp(x_demax)
    x_sum = torch.sum(x_exp, axis=-1, keepdim=True)
    return x_exp / x_sum
    


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
        softmax_triton_wrapper(warmup_output, x, B, C, F)
        torch.cuda.synchronize()
        triton_output = torch.zeros_like(x)
        start_time = time.time()
        for i in range(num_repeats):
            triton_result = softmax_triton_wrapper(triton_output, x, B, C, F)
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
        #     cuda_result = softmax_cuda_wrapper(cuda_output, x, B, C, F)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f"CUDA kernel time: {end_time - start_time:.2f} seconds")
        # cuda_diff = torch.max(torch.abs(cuda_result - pytorch_result)).item()
        # assert torch.allclose(cuda_result, pytorch_result, atol=1e-2, equal_nan=True)
        # print(f"✓ CUDA kernel passed for shape {x.shape} (max diff: {cuda_diff:.2e})")
        
        print(f"✓ All kernels match PyTorch baseline!\n")




if __name__ == "__main__":
    compare_kernels()