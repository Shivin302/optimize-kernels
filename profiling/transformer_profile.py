"""
Multi-Head Attention Profiling Script

This script implements multi-head attention from scratch in PyTorch and profiles it using NVIDIA Nsight Systems.

To run with nsys profiling:
    nsys profile -o mha_profile --capture-range=cudaProfilerApi --capture-range-end=stop python profiling/transformer_profile.py

This will:
1. Skip the first 10 warmup iterations (to avoid compilation overhead)
2. Profile iterations 10-19 (10 iterations total)
3. Create a mha_profile.nsys-rep file that can be opened in Nsight Systems GUI

Alternative profiling options:
    # Profile with more detailed CUDA API tracing
    nsys profile -o mha_profile --capture-range=cudaProfilerApi --trace=cuda,nvtx,osrt,cudnn,cublas python profiling/transformer_profile.py
    
    # Profile with kernel statistics
    nsys profile -o mha_profile --capture-range=cudaProfilerApi --stats=true python transformer_profile.py

View the results:
    nsys-ui mha_profile.nsys-rep
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# Configuration parameters
BATCH_SIZE = 64
SEQ_LEN = 512
EMBED_DIM = 512
NUM_HEADS = 8
DEVICE = 'cuda:0'
NUM_ITERS = 20
WARMUP_ITERS = 10


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
        # Shape: (B, T, C)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        # Shape: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # Shape: (B, num_heads, T, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            attn_scores = attn_scores + attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Shape: (B, 1, 1, S)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Compute attention weights
        # Shape: (B, num_heads, T, S)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # Shape: (B, num_heads, T, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        # Shape: (B, T, num_heads, head_dim) -> (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        return output


def main() -> None:
    """Main function to profile multi-head attention."""
    
    # Initialize model and move to device
    model = MultiHeadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout=0.1
    ).to(DEVICE)
    
    # Create dummy input data
    # For self-attention, Q, K, V all come from the same input
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, device=DEVICE)
    
    # Optional: create a causal mask for autoregressive models
    # causal_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE) * float('-inf'), diagonal=1)
    
    # Training loop with profiling
    for i in range(NUM_ITERS):
        # Start profiling after warmup iterations
        if i == WARMUP_ITERS:
            torch.cuda.cudart().cudaProfilerStart()
        
        # Push NVTX range for current iteration
        if i >= WARMUP_ITERS:
            torch.cuda.nvtx.range_push(f"iteration{i}")
        
        # Forward pass
        if i >= WARMUP_ITERS:
            torch.cuda.nvtx.range_push("forward")
        
        output = model(x, x, x)
        
        if i >= WARMUP_ITERS:
            torch.cuda.nvtx.range_pop()
        
        # Backward pass (if training)
        if i >= WARMUP_ITERS:
            torch.cuda.nvtx.range_push("backward")
        
        # Create a dummy loss and compute gradients
        loss = output.sum()
        loss.backward()
        
        if i >= WARMUP_ITERS:
            torch.cuda.nvtx.range_pop()
        
        # Pop iteration range
        if i >= WARMUP_ITERS:
            torch.cuda.nvtx.range_pop()
        
        # Zero gradients for next iteration
        model.zero_grad()
    
    # Stop profiling
    torch.cuda.cudart().cudaProfilerStop()
    
    print(f"Profiling completed successfully!")
    print(f"Output shape: {output.shape}")
    print(f"Warmup iterations: {WARMUP_ITERS}")
    print(f"Profiled iterations: {NUM_ITERS - WARMUP_ITERS}")


if __name__ == "__main__":
    main()
