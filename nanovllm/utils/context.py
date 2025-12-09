from dataclasses import dataclass

import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    def __repr__(self):
        return (
            f"Context(is_prefill={self.is_prefill}, "
            f"max_seqlen_q={self.max_seqlen_q}, "
            f"max_seqlen_k={self.max_seqlen_k}, "
            f"cu_seqlens_q={self.cu_seqlens_q.tolist() if self.cu_seqlens_q is not None else None}, "
            f"cu_seqlens_k={self.cu_seqlens_k.tolist() if self.cu_seqlens_k is not None else None}, "
            f"slot_mapping={self.slot_mapping.tolist() if self.slot_mapping is not None else None}, "
            f"context_lens={self.context_lens.tolist() if self.context_lens is not None else None}, "
            f"block_tables={self.block_tables.tolist() if self.block_tables is not None else None})"
        )


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
