# from einops import rearrange
import torch
import triton
import triton.language as tl

#************************************
# |      REFERENCE FUNCTIONS        |
#************************************

def linear_attn_cumsum_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes linear attention as:
      output[t] = q[t] @ (sum_{s<=t} (k[s] @ v[s]))

    Args:
    :params q: (B, H, L, D)
    :params k: (B, H, L, D)
    :params v: (B, H, L, D)

    Returns:
    :return: (B, H, L, D)
    """
    kv = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2)) # (B, H, L, D, 1) * (B, H, L, 1, D) -> (B, H, L, D, D)
    kv_cumsum = kv.cumsum(dim=2) # (B, H, L, D, D) -> (B, H, L, D, D)
    out = torch.matmul(q.unsqueeze(-2), kv_cumsum).squeeze(-2)  # (B, H, L, 1, D) * (B, H, L, D, D) -> (B, H, L, D)
    return out


def linear_attn_quadratic_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ Computes linear attention in the form Y = (L * QK^T) * V, where L is the lower triangular causal-mask matrix.

    Args:
    :params q: (B, H, L, D)
    :params k: (B, H, L, D)
    :params v: (B, H, L, D)

    Returns:
    :return: (B, H, L, D)
    """
    qk = torch.matmul(q, k.transpose(-2, -1))
    L, S = qk.shape[-2], qk.shape[-1]
    mask = torch.ones(1, 1, L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)
    qk *= mask
    out = torch.matmul(qk, v)
    return out



#************************************
# |         TODO FUNCTIONS          |
#************************************


@triton.jit
def outer_product_kernel(output, q, k, v, B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    l_idx = tl.program_id(2)


    qkv_offset = (b_idx * H * L * D) + (h_idx * L * D) + (l_idx * D)
    q_ptr = q + qkv_offset
    k_ptr = k + qkv_offset
    v_ptr = v + qkv_offset
    output_ptr = output + (b_idx * H * L * D * D) + (h_idx * L * D * D) + (l_idx * D * D)


    k_feat = tl.load(k_ptr + tl.arange(0, D))
    v_feat = tl.load(v_ptr + tl.arange(0, D))
    kv_outer = k_feat[:, None] * v_feat[None, :]
    
    tl.store(output_ptr + tl.arange(0, D * D), tl.reshape(kv_outer, D * D))





@triton.jit
def cumsum_kernel(output, B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    output_ptr = output + (b_idx * H * L * D * D) + (h_idx * L * D * D)

    kv_block = tl.load(output_ptr + tl.arange(0, L * D * D))
    kv_block = tl.reshape(kv_block, (L, D, D))
    kv_block = tl.cumsum(kv_block, axis=0)
    tl.store(output_ptr + tl.arange(0, L * D * D), tl.reshape(kv_block, L * D * D))



@triton.jit
def inner_product_kernel(output, q, kv, B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    l_idx = tl.program_id(2)


    q_ptr = q + (b_idx * H * L * D) + (h_idx * L * D) + (l_idx * D)
    kv_ptr = kv + (b_idx * H * L * D * D) + (h_idx * L * D * D) + (l_idx * D * D)
    output_ptr = output + (b_idx * H * L * D) + (h_idx * L * D) + (l_idx * D)


    q_block = tl.load(q_ptr + tl.arange(0, D))
    kv_block = tl.load(kv_ptr + tl.arange(0, D * D))
    
    qkv_inner = tl.sum(q_block[:, None] * tl.reshape(kv_block, (D, D)), axis=0)
    
    tl.store(output_ptr + tl.arange(0, D), qkv_inner)



def my_implementation(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Args:
    :params q: (B, H, L, D)
    :params k: (B, H, L, D)
    :params v: (B, H, L, D)

    Returns:
    :return: (B, H, L, D)
    """

    B, H, L, D = q.shape
    kv = torch.zeros((B, H, L, D, D), dtype=torch.float32, device=q.device)

    grid = lambda meta: (B, H, L)
    outer_product_kernel[grid](kv, q, k, v, B, H, L, D)

    grid = lambda meta: (B, H)
    cumsum_kernel[grid](kv, B, H, L, D)

    result = torch.zeros((B, H, L, D), dtype=torch.float32, device=q.device)

    grid = lambda meta: (B, H, L)
    inner_product_kernel[grid](result, q, kv, B, H, L, D)

    return result


@triton.jit
def linear_attn_cumsum_kernel(output, q,k,v,B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    
    kv_cumsum = tl.zeros((D, D), tl.float32)
    for l_idx in range(L):
        qkv_offset = (b_idx * H * L * D) + (h_idx * L * D) + (l_idx * D)
        q_ptr = q + qkv_offset
        k_ptr = k + qkv_offset
        v_ptr = v + qkv_offset
        output_ptr = output + qkv_offset


        k_feat = tl.load(k_ptr + tl.arange(0, D))
        v_feat = tl.load(v_ptr + tl.arange(0, D))
        kv_outer = k_feat[:, None] * v_feat[None, :]
        kv_cumsum += kv_outer

        q_block = tl.load(q_ptr + tl.arange(0, D))
        
        qkv_inner = tl.sum(q_block[:, None] * tl.reshape(kv_cumsum, (D, D)), axis=0)
        
        tl.store(output_ptr + tl.arange(0, D), qkv_inner)




def my_implementation_single_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Args:
    :params q: (B, H, L, D)
    :params k: (B, H, L, D)
    :params v: (B, H, L, D)

    Returns:
    :return: (B, H, L, D)
    """
    B, H, L, D = q.shape
    result = torch.zeros((B, H, L, D), dtype=torch.float32, device=q.device)

    grid = lambda meta: (B, H, L)
    linear_attn_cumsum_kernel[grid](result, q, k, v, B, H, L, D)

    return result


def test_linear_attn_ref():
    B, H, L, D = 2, 8, 32, 16
    q = torch.randn(B, H, L, D, device='cuda')
    k = torch.randn(B, H, L, D, device='cuda')
    v = torch.randn(B, H, L, D, device='cuda')
    
    quadratic_ref = linear_attn_quadratic_ref(q, k, v)


    cumsum_ref = linear_attn_cumsum_ref(q, k, v)
    first_impl = my_implementation(q, k, v)
    single_kernel_impl = my_implementation_single_kernel(q, k, v)


    assert torch.allclose(quadratic_ref, cumsum_ref, atol=1e-2)
    assert torch.allclose(quadratic_ref, first_impl, atol=1e-2)
    assert torch.allclose(quadratic_ref, single_kernel_impl, atol=1e-2)
    print("Implementation same as ref")


if __name__ == "__main__":
    test_linear_attn_ref()
