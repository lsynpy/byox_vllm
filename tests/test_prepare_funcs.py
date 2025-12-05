from unittest.mock import Mock

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.utils.context import get_context, reset_context


def make_model_runner():
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 16
    runner._prepare_block_tables = Mock(return_value=torch.tensor([[0, -1, -1, -1]]))
    return runner


def test_prepare_prefill_single_sequence():
    runner = make_model_runner()
    seq = Sequence([1, 2, 3, 4, 5], block_size=runner.block_size)
    seq.block_table = [0]
    seq.num_cached_tokens = 0
    input_ids, positions, _ = runner._prepare_prefill([seq])
    context = get_context()

    assert context.is_prefill
    assert torch.equal(context.cu_seqlens_q.cpu(), torch.tensor([0, 5]))
    assert torch.equal(context.cu_seqlens_k.cpu(), torch.tensor([0, 5]))
    assert context.max_seqlen_q == 5
    assert context.max_seqlen_k == 5
    assert torch.equal(context.slot_mapping.cpu(), torch.tensor([0, 1, 2, 3, 4]))
    assert context.context_lens is None
    assert context.block_tables is None
    assert torch.equal(input_ids.cpu(), torch.tensor([1, 2, 3, 4, 5]))
    assert torch.equal(positions.cpu(), torch.tensor([0, 1, 2, 3, 4]))
    reset_context()


def test_prepare_prefill_multiple_sequences():
    runner = make_model_runner()
    seq0 = Sequence([1, 2, 3], block_size=runner.block_size)
    seq1 = Sequence([4, 5, 6, 7, 8], block_size=runner.block_size)
    seq0.block_table = [0]
    seq1.block_table = [1]
    seq0.num_cached_tokens = 0
    seq1.num_cached_tokens = 0
    input_ids, positions, _ = runner._prepare_prefill([seq0, seq1])
    context = get_context()

    assert context.is_prefill
    assert torch.equal(context.cu_seqlens_q.cpu(), torch.tensor([0, 3, 8]))
    assert torch.equal(context.cu_seqlens_k.cpu(), torch.tensor([0, 3, 8]))
    assert context.max_seqlen_q == 5
    assert context.max_seqlen_k == 5
    assert torch.equal(context.slot_mapping.cpu(), torch.tensor([0, 1, 2, 16, 17, 18, 19, 20]))
    assert context.context_lens is None
    assert context.block_tables is None
    assert torch.equal(input_ids.cpu(), torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64))
    assert torch.equal(positions.cpu(), torch.tensor([0, 1, 2, 0, 1, 2, 3, 4], dtype=torch.int64))
    reset_context()


def test_prepare_prefill_with_cached_tokens():
    runner = make_model_runner()
    seq = Sequence([1, 2, 3, 4, 5, 6], block_size=runner.block_size)
    seq.num_cached_tokens = 3
    seq.block_table = [0, 1]
    input_ids, positions, _ = runner._prepare_prefill([seq])
    context = get_context()

    assert context.is_prefill
    assert torch.equal(context.cu_seqlens_q.cpu(), torch.tensor([0, 3]))
    assert torch.equal(context.cu_seqlens_k.cpu(), torch.tensor([0, 6]))
    assert context.max_seqlen_q == 3
    assert context.max_seqlen_k == 6
    assert torch.equal(context.slot_mapping.cpu(), torch.tensor([0, 1, 2, 3, 4, 5]))
    assert context.context_lens is None
    assert torch.equal(context.block_tables.cpu(), torch.tensor([[0, -1, -1, -1]]))
    assert torch.equal(input_ids.cpu(), torch.tensor([4, 5, 6], dtype=torch.int64))
    assert torch.equal(positions.cpu(), torch.tensor([3, 4, 5], dtype=torch.int64))
    reset_context()


def test_prepare_decode_single_sequence():
    runner = make_model_runner()
    seq = Sequence([1, 2, 3, 4, 5], block_size=runner.block_size)
    seq.status = SequenceStatus.RUNNING
    seq.block_table = [0, -1, -1, -1]
    seq.num_cached_tokens = 0
    runner._prepare_block_tables = Mock(return_value=torch.tensor([[0, -1, -1, -1]], dtype=torch.int32))
    input_ids, positions, _ = runner._prepare_decode([seq])
    context = get_context()

    assert not context.is_prefill
    assert context.cu_seqlens_q is None
    assert context.cu_seqlens_k is None
    assert context.max_seqlen_q == 0
    assert context.max_seqlen_k == 0
    assert torch.equal(context.slot_mapping.cpu(), torch.tensor([-12]))
    assert torch.equal(context.context_lens.cpu(), torch.tensor([5]))
    assert torch.equal(context.block_tables.cpu(), torch.tensor([[0, -1, -1, -1]]))
    assert torch.equal(input_ids.cpu(), torch.tensor([5], dtype=torch.int64))
    assert torch.equal(positions.cpu(), torch.tensor([4], dtype=torch.int64))
    reset_context()


def test_prepare_decode_multiple_sequences():
    runner = make_model_runner()
    seq0 = Sequence([1, 2, 3], block_size=runner.block_size)
    seq1 = Sequence([4, 5, 6, 7], block_size=runner.block_size)
    seq0.block_table = [0, -1, -1, -1]
    seq1.block_table = [1, -1, -1, -1]
    seq0.num_cached_tokens = 0
    seq1.num_cached_tokens = 0
    runner._prepare_block_tables = Mock(
        return_value=torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1]], dtype=torch.int32)
    )
    input_ids, positions, _ = runner._prepare_decode([seq0, seq1])
    context = get_context()

    assert not context.is_prefill
    assert context.cu_seqlens_q is None
    assert context.cu_seqlens_k is None
    assert context.max_seqlen_q == 0
    assert context.max_seqlen_k == 0
    assert torch.equal(context.slot_mapping.cpu(), torch.tensor([-14, -13]))
    assert torch.equal(context.context_lens.cpu(), torch.tensor([3, 4]))
    assert torch.equal(context.block_tables.cpu(), torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1]]))
    assert torch.equal(input_ids.cpu(), torch.tensor([3, 7], dtype=torch.int64))
    assert torch.equal(positions.cpu(), torch.tensor([2, 3], dtype=torch.int64))
    reset_context()


def test_prepare_prefill_multiple_with_cached_tokens():
    runner = make_model_runner()
    seq0 = Sequence([1, 2, 3, 4, 5], block_size=runner.block_size)
    seq1 = Sequence([6, 7, 8, 9, 10, 11], block_size=runner.block_size)
    seq0.num_cached_tokens = 2
    seq1.num_cached_tokens = 1
    seq0.block_table = [0]
    seq1.block_table = [1]
    input_ids, positions, _ = runner._prepare_prefill([seq0, seq1])
    context = get_context()

    assert context.is_prefill
    assert torch.equal(context.cu_seqlens_q.cpu(), torch.tensor([0, 3, 8]))
    assert torch.equal(context.cu_seqlens_k.cpu(), torch.tensor([0, 5, 11]))
    assert context.max_seqlen_q == 5
    assert context.max_seqlen_k == 6
    assert torch.equal(context.slot_mapping.cpu(), torch.tensor([0, 1, 2, 3, 4, 16, 17, 18, 19, 20, 21]))
    assert context.context_lens is None
    assert torch.equal(context.block_tables.cpu(), torch.tensor([[0, -1, -1, -1]]))
    assert torch.equal(input_ids.cpu(), torch.tensor([3, 4, 5, 7, 8, 9, 10, 11], dtype=torch.int64))
    assert torch.equal(positions.cpu(), torch.tensor([2, 3, 4, 1, 2, 3, 4, 5], dtype=torch.int64))
    reset_context()


def test_prepare_decode_with_cached_tokens():
    runner = make_model_runner()
    seq = Sequence([1, 2, 3, 4, 5], block_size=runner.block_size)
    seq.status = SequenceStatus.RUNNING
    seq.block_table = [0, -1, -1, -1]
    seq.num_cached_tokens = 2
    runner._prepare_block_tables = Mock(return_value=torch.tensor([[0, -1, -1, -1]], dtype=torch.int32))
    input_ids, positions, _ = runner._prepare_decode([seq])
    context = get_context()

    assert not context.is_prefill
    assert context.cu_seqlens_q is None
    assert context.cu_seqlens_k is None
    assert context.max_seqlen_q == 0
    assert context.max_seqlen_k == 0
    assert torch.equal(context.slot_mapping.cpu(), torch.tensor([-12]))
    assert torch.equal(context.context_lens.cpu(), torch.tensor([5]))
    assert torch.equal(context.block_tables.cpu(), torch.tensor([[0, -1, -1, -1]]))
    assert torch.equal(input_ids.cpu(), torch.tensor([5], dtype=torch.int64))
    assert torch.equal(positions.cpu(), torch.tensor([4], dtype=torch.int64))
    reset_context()


def test_prepare_decode_multiple_with_cached_tokens():
    runner = make_model_runner()
    seq0 = Sequence([1, 2, 3, 4], block_size=runner.block_size)
    seq1 = Sequence([5, 6, 7, 8, 9], block_size=runner.block_size)
    seq0.status = SequenceStatus.RUNNING
    seq1.status = SequenceStatus.RUNNING
    seq0.block_table = [0, -1, -1, -1]
    seq1.block_table = [1, -1, -1, -1]
    seq0.num_cached_tokens = 2
    seq1.num_cached_tokens = 3
    runner._prepare_block_tables = Mock(
        return_value=torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1]], dtype=torch.int32)
    )
    input_ids, positions, _ = runner._prepare_decode([seq0, seq1])
    context = get_context()

    assert not context.is_prefill
    assert context.cu_seqlens_q is None
    assert context.cu_seqlens_k is None
    assert context.max_seqlen_q == 0
    assert context.max_seqlen_k == 0
    assert torch.equal(context.slot_mapping.cpu(), torch.tensor([-13, -12]))
    assert torch.equal(context.context_lens.cpu(), torch.tensor([4, 5]))
    assert torch.equal(context.block_tables.cpu(), torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1]]))
    assert torch.equal(input_ids.cpu(), torch.tensor([4, 9], dtype=torch.int64))
    assert torch.equal(positions.cpu(), torch.tensor([3, 4], dtype=torch.int64))
    reset_context()
