"""
Multi-Head Attention Profiling Script using PyTorch Profiler

This script implements multi-head attention from scratch in PyTorch and profiles it using PyTorch's built-in profiler.

To run with PyTorch profiler:
    python transformer_pytorch_profile.py

This will:
1. Profile the execution with detailed kernel-level information
2. Generate a TensorBoard trace in logs/
3. Show top operations by CPU/CUDA time in the console

View the results in TensorBoard:
    tensorboard --logdir=logs/
    # Then open http://localhost:6006 in your browser and go to the PyTorch Profiler tab

Alternative: View the Chrome trace directly:
    # The script also exports a Chrome trace JSON file
    # Open chrome://tracing in Chrome and load the .json file from ./profiling/logs/

Key features of PyTorch profiler:
- Records CPU and GPU operations with call stacks
- Shows memory allocations and deallocations
- Provides operator-level breakdown
- Integrates with TensorBoard for visualization
- Can export Chrome trace format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path
from typing import Optional


# Configuration parameters
BATCH_SIZE = 64
SEQ_LEN = 512
EMBED_DIM = 512
NUM_HEADS = 8
DEVICE = 'cuda:0'
NUM_ITERS = 20
WARMUP_ITERS = 10
PROFILE_DIR = Path(__file__).parent / "logs"


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module implemented from scratch.
    
    This implements the attention mechanism from "Attention is All You Need" (Vaswani et al., 2017).
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in linear projections (default: True)
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.0, 
        bias: bool = True
    ) -> None:
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # Scale factor for attention scores
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout layer
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (B, T, C) where B=batch, T=sequence length, C=embed_dim
            key: Key tensor of shape (B, S, C) where S=source sequence length
            value: Value tensor of shape (B, S, C)
            attn_mask: Optional attention mask of shape (T, S) or (B, T, S)
            key_padding_mask: Optional padding mask of shape (B, S) where True indicates padding
            
        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = query.shape
        _, S, _ = key.shape
        
        # Project inputs to Q, K, V
        with record_function("qkv_projection"):
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        
        # Reshape for multi-head attention
        with record_function("reshape_heads"):
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        with record_function("attention_scores"):
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            with record_function("apply_attention_mask"):
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attn_scores = attn_scores + attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            with record_function("apply_padding_mask"):
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Compute attention weights
        with record_function("softmax_dropout"):
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        with record_function("apply_attention"):
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        with record_function("reshape_output"):
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection
        with record_function("output_projection"):
            output = self.out_proj(attn_output)
        
        return output


def main() -> None:
    """Main function to profile multi-head attention using PyTorch profiler."""
    
    # Create output directory
    Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize model and move to device
    model = MultiHeadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=0.1
    ).to(DEVICE)
    
    # Create dummy input data
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    
    # Create optimizer for backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting warmup iterations...")
    # Warmup iterations (not profiled)
    for i in range(WARMUP_ITERS):
        optimizer.zero_grad()
        output = model(x, x, x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
    
    print(f"Warmup complete. Starting profiling for {NUM_ITERS - WARMUP_ITERS} iterations...")
    
    # Profile the execution with TensorBoard handler
    # Schedule: wait=1 (skip first iter), warmup=1 (warmup), active=5 (profile 5 iters), repeat=1 (one file)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(PROFILE_DIR)),
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1)
    ) as prof:
        for i in range(NUM_ITERS - WARMUP_ITERS):
            with record_function(f"iteration_{i}"):
                optimizer.zero_grad()
                
                # Forward pass
                with record_function("forward"):
                    output = model(x, x, x)
                
                # Backward pass
                with record_function("backward"):
                    loss = output.sum()
                    loss.backward()
                
                # Optimizer step
                with record_function("optimizer_step"):
                    optimizer.step()
            
            # Step the profiler to record each iteration
            prof.step()
    
    print("Profiling completed!")
    
    # Print profiling results to console
    print("\n" + "="*80)
    print("GPU PERFORMANCE ANALYSIS - CUDA TIME (FOCUS HERE FOR GPU OPTIMIZATION)")
    print("="*80)
    print("Note: Ignore 'cudaDeviceSynchronize' in CPU time - it's just CPU waiting.")
    print("Focus on CUDA time which shows actual GPU kernel execution time.")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    print("\n" + "="*80)
    print("CPU TIME (includes synchronization overhead - less useful for GPU analysis)")
    print("="*80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    print("\n" + "="*80)
    print("TOP OPERATIONS BY MEMORY USAGE")
    print("="*80)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    
    # TensorBoard trace is automatically saved by on_trace_ready handler
    print(f"\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"\nTensorBoard traces saved to: {PROFILE_DIR}")
    print(f"To view in TensorBoard:")
    print(f"  tensorboard --logdir={PROFILE_DIR}")
    print(f"  Then open http://localhost:6006 in your browser")
    print(f"  Go to the 'PyTorch Profiler' or 'Trace' tab")
    
    print(f"\nChrome trace files are also available in: {PROFILE_DIR}")
    print(f"Open chrome://tracing and load the .pt.trace.json file")
    
    print(f"\nOutput shape: {output.shape}")


if __name__ == "__main__":
    main()
