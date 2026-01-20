#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <math.h>


__global__ void rms_norm_cuda(float* output, float* x, float* gamma, float epsilon, int B, int C, int F) {
    int b_idx = blockIdx.x;
    int c_idx = blockIdx.y;
    int num_threads = blockDim.x;
    float* x_ptr = x + b_idx * C * F + c_idx * F;
    float* output_ptr = output + b_idx * C * F + c_idx * F;
    float* gamma_ptr = gamma;

    // pull squared values from x to shared memory with coalescing
    extern __shared__ float x_shared[];  // Dynamically allocated shared memory
    
    // Initialize shared memory to 0
    x_shared[threadIdx.x] = 0.0f;
    
    for (int i = 0; i < F; i += num_threads) {
        if (i + threadIdx.x < F) {
            float val = x_ptr[i + threadIdx.x];
            x_shared[threadIdx.x] += val * val;
        }
    }
    __syncthreads();
    
    // parallel reduction to calculate sum of squares
    // minimizes shared memory bank conflicts
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            x_shared[threadIdx.x] += x_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float sum_sq = x_shared[0];
    float rms = rsqrtf((sum_sq / float(F)) + epsilon);
    
    // apply normalization and gamma, then write to output
    for (int i = 0; i < F; i += num_threads) {
        if (i + threadIdx.x < F) {
            output_ptr[i + threadIdx.x] = x_ptr[i + threadIdx.x] * rms * gamma_ptr[i + threadIdx.x];
        }
    }
}

torch::Tensor rms_norm_wrapper_cuda(
    torch::Tensor output,
    torch::Tensor x,
    torch::Tensor gamma,
    float epsilon,
    int B,
    int C,
    int F
) {
    dim3 gridDim(B, C, 1);
    dim3 blockDim(256, 1, 1);
    int shared_mem_size = 256 * sizeof(float);  // 256 threads * 4 bytes per float
    
    rms_norm_cuda<<<gridDim, blockDim, shared_mem_size>>>(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        epsilon,
        B,
        C,
        F
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_cuda", &rms_norm_wrapper_cuda, "RMS Norm CUDA kernel");
}
