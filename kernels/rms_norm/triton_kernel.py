import torch
import triton
import triton.language as tl
import json
from pathlib import Path
from torch.utils.cpp_extension import load
import time

# Load CUDA extension
cuda_module = load(
    name="rms_norm_cuda",
    sources=[str(Path(__file__).parent / "cuda_kernel.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=False
)


@triton.jit
def rms_norm_triton(output, x, gamma, epsilon: tl.constexpr, B: tl.constexpr, C: tl.constexpr, F: tl.constexpr):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    output_ptr = output + (b_idx * C * F) + (c_idx * F)
    x_ptr = x + (b_idx * C * F) + (c_idx * F)
    x_feat = tl.load(x_ptr + tl.arange(0, F))
    gamma_vals = tl.load(gamma + tl.arange(0, F))

    # Calculate mean of squares
    mean_sq = tl.sum(x_feat * x_feat) / F
    # Calculate RMS normalization factor
    rms_norm_factor = tl.rsqrt(mean_sq + epsilon)
    # Apply normalization and gamma
    result = x_feat * rms_norm_factor * gamma_vals
    tl.store(output_ptr + tl.arange(0, F), result)


def rms_norm_triton_wrapper(output, x, gamma, epsilon, B, C, F):
    grid = lambda meta: (B, C)
    rms_norm_triton[grid](output, x, gamma, epsilon, B, C, F)
    return output


def rms_norm_cuda_wrapper(output, x, gamma, epsilon, B, C, F):
    return cuda_module.rms_norm_cuda(output, x, gamma, epsilon, B, C, F)


def pytorch_baseline(x, gamma, epsilon):
    return x / torch.sqrt(torch.mean(x**2, axis=-1, keepdim=True) + epsilon) * gamma


def compare_kernels():
    with open(Path(__file__).parent / "inputs.json", "r") as f:
        input_configs = json.load(f)
    input_arrays = [None] * len(input_configs)
    
    for i, input_config in enumerate(input_configs):
        input_arrays[i] = {}
        for key, value in input_config.items():
            if key == "epsilon":
                # Epsilon should be a small positive value
                input_arrays[i][key] = torch.tensor([1e-8], dtype=torch.float32).cuda()
            else:
                input_arrays[i][key] = torch.randn(*value["shape"]).to(torch.float32).cuda()
    
    for input_array in input_arrays:
        x = input_array["x"]
        gamma = input_array["gamma"]
        epsilon = input_array["epsilon"].item()  # Extract scalar from tensor
        B, C, F = x.shape
        
        
        # PyTorch baseline
        start_time = time.time()
        pytorch_result = pytorch_baseline(x, gamma, epsilon)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"PyTorch baseline time: {end_time - start_time:.2f} seconds")
        
        # Triton kernel
        # Warmup: compile Triton kernel before timing
        warmup_output = torch.zeros_like(x)
        rms_norm_triton_wrapper(warmup_output, x, gamma, epsilon, B, C, F)
        torch.cuda.synchronize()
        triton_output = torch.zeros_like(x)
        start_time = time.time()
        triton_result = rms_norm_triton_wrapper(triton_output, x, gamma, epsilon, B, C, F)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Triton kernel time: {end_time - start_time:.2f} seconds")
        triton_diff = torch.max(torch.abs(triton_result - pytorch_result)).item()
        assert torch.allclose(triton_result, pytorch_result, atol=1e-2, equal_nan=True)
        print(f"✓ Triton kernel passed for shape {x.shape} (max diff: {triton_diff:.2e})")
        
        # CUDA kernel
        cuda_output = torch.zeros_like(x)
        start_time = time.time()
        cuda_result = rms_norm_cuda_wrapper(cuda_output, x, gamma, epsilon, B, C, F)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"CUDA kernel time: {end_time - start_time:.2f} seconds")
        cuda_diff = torch.max(torch.abs(cuda_result - pytorch_result)).item()
        assert torch.allclose(cuda_result, pytorch_result, atol=1e-2, equal_nan=True)
        print(f"✓ CUDA kernel passed for shape {x.shape} (max diff: {cuda_diff:.2e})")
        
        print(f"✓ All kernels match PyTorch baseline!\n")




if __name__ == "__main__":
    compare_kernels()