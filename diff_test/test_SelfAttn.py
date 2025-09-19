import os
import sys

sys.path.append(os.path.abspath("."))


import os
import sys

import torch
from transformers import Qwen3Config

from byoxvllm.layers import Attention
from nanovllm.layers.attention import Attention as NanoAttention
from nanovllm.utils.context import set_context

torch.manual_seed(0)  # Fixed seed for reproducibility


def _generate_random_inputs(batch_size, seq_len, num_heads, head_dim, num_kv_heads, device, dtype):
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def byox_inputs(batch_size, seq_len, num_heads, head_dim, num_kv_heads, device, dtype):
    return _generate_random_inputs(batch_size, seq_len, num_heads, head_dim, num_kv_heads, device, dtype)


def nano_inputs(batch_size, seq_len, num_heads, head_dim, num_kv_heads, device, dtype):
    q, k, v = _generate_random_inputs(batch_size, seq_len, num_heads, head_dim, num_kv_heads, device, dtype)

    total_q = batch_size * seq_len

    q = q.permute(0, 2, 1, 3).reshape(total_q, num_heads, head_dim)
    k = k.permute(0, 2, 1, 3).reshape(total_q, num_kv_heads, head_dim)
    v = v.permute(0, 2, 1, 3).reshape(total_q, num_kv_heads, head_dim)

    return q, k, v


def nano_self_attn():
    config = Qwen3Config()
    batch_size = 1
    seq_len = 8

    nano_device = torch.device("cuda")
    nano_dtype = torch.float16

    nano_q, nano_k, nano_v = nano_inputs(
        batch_size,
        seq_len,
        config.num_attention_heads,
        config.head_dim,
        config.num_key_value_heads,
        nano_device,
        nano_dtype,
    )

    # Create nano attention with weights loaded from Qwen3 config
    scale = config.head_dim**-0.5
    nano_attn = NanoAttention(
        num_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        scale=scale,
        num_kv_heads=config.num_key_value_heads,
    )
    nano_attn = nano_attn.to(nano_device)

    set_context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=nano_device),
        cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=nano_device),
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
    )

    nano_output = nano_attn(nano_q, nano_k, nano_v)
    return nano_output.cpu().flatten()


def byox_self_attn():
    config = Qwen3Config()
    batch_size = 1
    seq_len = 8

    # For fair comparison, we'll run byox on CPU
    byox_device = torch.device("cpu")
    byox_dtype = torch.float16

    byox_q, byox_k, byox_v = byox_inputs(
        batch_size,
        seq_len,
        config.num_attention_heads,
        config.head_dim,
        config.num_key_value_heads,
        byox_device,
        byox_dtype,
    )

    # Create byox attention
    byox_attn = Attention(
        num_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        num_kv_heads=config.num_key_value_heads,
    )
    byox_attn = byox_attn.to(byox_device)

    byox_output = byox_attn(byox_q, byox_k, byox_v)
    return byox_output.flatten()


def compare_output():
    nano_output = nano_self_attn()
    byox_output = byox_self_attn()
    print(f"{torch.sum(byox_output != nano_output)} different values in total {byox_output.numel()}")
    print(f"Max difference: {torch.max(torch.abs(byox_output - nano_output))}")
    if torch.equal(byox_output, nano_output):
        print("SelfAttention outputs are exactly the same.")
    else:
        print("SelfAttention outputs are not exactly the same")


if __name__ == "__main__":
    compare_output()
