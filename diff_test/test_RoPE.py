import os
import sys

sys.path.append(os.path.abspath("."))


import torch
from transformers import Qwen3Config

from byoxvllm.layers import RotaryEmbedding as ByoxRotaryEmbedding
from byoxvllm.layers import apply_rotary_pos_emb
from nanovllm.layers.rotary_embedding import RotaryEmbedding as NanoRotaryEmbedding


def test_RoPE():
    torch.manual_seed(0)

    config = Qwen3Config()
    byox_rope = ByoxRotaryEmbedding(
        dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    nano_rope = NanoRotaryEmbedding(
        head_size=config.head_dim,
        rotary_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )

    # 3. Create fake input tensors
    batch_size = 1
    seq_len = 10
    num_heads = config.num_attention_heads
    head_dim = config.head_dim

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    positions = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)

    # 4. Run the input through both RoPE implementations
    # byoxvllm
    byox_cos, byox_sin = byox_rope(dtype=q.dtype, position_ids=positions)
    byox_q, byox_k = apply_rotary_pos_emb(q, k, byox_cos, byox_sin)

    # nanovllm
    # nanovllm expects query and key to be of shape (batch*seq_len, num_heads, head_dim)
    # but the rotary embedding is applied on the last dim, so we can just reshape
    nano_q_in = q.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_dim)
    nano_k_in = k.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_dim)
    nano_pos_in = positions.view(-1)

    nano_q_out, nano_k_out = nano_rope(nano_pos_in, nano_q_in, nano_k_in)

    # reshape back
    nano_q = nano_q_out.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    nano_k = nano_k_out.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # 6. Compare the outputs
    print("Comparing Q")
    print(f"{torch.sum(byox_q != nano_q)} different values in total {byox_q.numel()}")
    print(f"Max difference: {torch.max(torch.abs(byox_q - nano_q))}")
    if torch.allclose(byox_q, nano_q):
        print("Q outputs are close enough.")
    else:
        print("Q outputs are not close enough.")

    print("\nComparing K")
    print(f"{torch.sum(byox_k != nano_k)} different values in total {byox_k.numel()}")
    print(f"Max difference: {torch.max(torch.abs(byox_k - nano_k))}")
    if torch.allclose(byox_k, nano_k):
        print("K outputs are close enough.")
    else:
        print("K outputs are not close enough.")


if __name__ == "__main__":
    test_RoPE()
