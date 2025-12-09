import logging

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func
from torch import nn

from nanovllm.utils.context import get_context
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__, logging.DEBUG)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:  # empty batch
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            logger.debug_once(
                "store_kvcache: k.shape: %s, v.shape: %s, k_cache.shape: %s, "
                "v_cache.shape: %s, slot_mapping: %s",
                k.shape,
                v.shape,
                k_cache.shape,
                v_cache.shape,
                context.slot_mapping.tolist(),
            )
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:  # warmup has no kv cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
            logger.debug_once(
                "call flash_attn prefill: q shape: %s, k shape: %s, v shape: %s, "
                "max_seqlen_q: %s, cu_seqlens_q: %s, "
                "max_seqlen_k: %s, cu_seqlens_k: %s, block_tables: %s"
                "\nflash_attn out: %s",
                q.shape,
                k.shape,
                v.shape,
                context.max_seqlen_q,
                context.cu_seqlens_q.tolist(),
                context.max_seqlen_k,
                context.cu_seqlens_k.tolist(),
                context.block_tables.tolist() if context.block_tables is not None else None,
                o.shape,
            )
        else:  # decode
            q_reshape = q.view(-1, q.shape[-2], q.shape[-1])  # [total_q_tokens, num_heads, head_dim]

            o = flash_attn_varlen_func(
                q_reshape,
                k_cache,
                v_cache,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
            logger.debug_once(
                "call flash_attn decode: q shape: %s, k shape: %s, v shape: %s, "
                "max_seqlen_q: %s, cu_seqlens_q: %s, "
                "max_seqlen_k: %s, cu_seqlens_k: %s, block_tables: %s"
                "\nflash_attn out: %s",
                q_reshape.shape,
                k_cache.shape,
                v_cache.shape,
                context.max_seqlen_q,
                context.cu_seqlens_q.tolist(),
                context.max_seqlen_k,
                context.cu_seqlens_k.tolist(),
                context.block_tables.tolist() if context.block_tables is not None else None,
                o.shape,
            )
        return o
