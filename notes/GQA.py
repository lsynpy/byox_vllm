#!/usr/bin/env python3
"""
Minimal demo of Grouped-Query Attention (GQA) mechanics
"""

import torch
import torch.nn.functional as F

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key and value states for Grouped-Query Attention
    
    Args:
        hidden_states: [batch_size, num_key_value_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each key/value head
    
    Returns:
        [batch_size, num_key_value_heads * n_rep, seq_len, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def standard_mha_demo():
    """Demo of standard Multi-Head Attention"""
    print("=== Standard Multi-Head Attention (MHA) ===")
    
    # Parameters
    batch_size = 1
    seq_len = 4
    num_heads = 4
    head_dim = 8
    
    # Create random Q, K, V with same number of heads
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"Q shape: {Q.shape}")  # [1, 4, 4, 8]
    print(f"K shape: {K.shape}")  # [1, 4, 4, 8]
    print(f"V shape: {V.shape}")  # [1, 4, 4, 8]
    
    # Compute attention for each head
    for i in range(num_heads):
        # Scale factor
        scale = head_dim ** -0.5
        
        # Compute attention scores
        attn_scores = torch.matmul(Q[:, i, :, :], K[:, i, :, :].transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V[:, i, :, :])
        print(f"Head {i} output shape: {output.shape}")  # [1, 4, 8]

def gqa_demo():
    """Demo of Grouped-Query Attention"""
    print("\n=== Grouped-Query Attention (GQA) ===")
    
    # Parameters - Qwen3-0.6B style: 16 Q heads, 8 K/V heads
    batch_size = 1
    seq_len = 4
    num_q_heads = 4  # Simplified from 16
    num_kv_heads = 2  # Simplified from 8
    head_dim = 8
    n_rep = num_q_heads // num_kv_heads  # 2
    
    print(f"Number of Q heads: {num_q_heads}")
    print(f"Number of KV heads: {num_kv_heads}")
    print(f"Each KV head serves {n_rep} Q heads")
    
    # Create random Q with more heads
    Q = torch.randn(batch_size, num_q_heads, seq_len, head_dim)
    print(f"Q shape: {Q.shape}")  # [1, 4, 4, 8]
    
    # Create K and V with fewer heads
    K = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    print(f"K shape: {K.shape}")  # [1, 2, 4, 8]
    print(f"V shape: {V.shape}")  # [1, 2, 4, 8]
    
    # Repeat K and V to match Q heads
    K_repeated = repeat_kv(K, n_rep)
    V_repeated = repeat_kv(V, n_rep)
    print(f"K repeated shape: {K_repeated.shape}")  # [1, 4, 4, 8]
    print(f"V repeated shape: {V_repeated.shape}")  # [1, 4, 4, 8]
    
    # Now compute attention just like standard MHA
    scale = head_dim ** -0.5
    outputs = []
    
    for i in range(num_q_heads):
        # Compute attention scores
        attn_scores = torch.matmul(Q[:, i, :, :], K_repeated[:, i, :, :].transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V_repeated[:, i, :, :])
        outputs.append(output)
        print(f"Head {i} output shape: {output.shape}")  # [1, 4, 8]
    
    # Group heads that share the same K/V
    print("\nGrouping heads that share K/V:")
    for group in range(num_kv_heads):
        start_idx = group * n_rep
        end_idx = (group + 1) * n_rep
        print(f"K/V head {group} serves Q heads {start_idx} to {end_idx-1}")

if __name__ == "__main__":
    standard_mha_demo()
    gqa_demo()