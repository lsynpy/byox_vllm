import os
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def test_seq_waiting_for_allocation():
    # block manager cannot allocate seq2, so seq2 will wait for prefill, seq1 continue decoding
    block_size = 256
    num_blocks = 10
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    config = Config(model=path, kvcache_block_size=block_size, num_kvcache_blocks=num_blocks)
    scheduler = Scheduler(config)

    common_token_ids = [i for i in range(3) for _ in range(block_size)]  # Complete 3 blocks
    unique_token_ids = [3] * 4  # Incomplete 1 block (4 tokens)
    all_token_ids = common_token_ids + unique_token_ids

    seq1 = Sequence(all_token_ids, block_size)
    scheduler.add(seq1)
    seqs, is_prefill = scheduler.schedule()
    assert seqs == [seq1]
    assert is_prefill

    seq2 = Sequence([i for i in range(7) for _ in range(block_size)], block_size)
    scheduler.add(seq2)
    seqs, is_prefill = scheduler.schedule()
    assert seqs == [seq1]
    assert not is_prefill


def test_max_batched_tokens_exceed():
    # single seq exceed max_batched_tokens, so no seq is scheduled
    block_size = 256
    num_blocks = 10
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    config = Config(
        model=path,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
        max_num_batched_tokens=1000,
        max_model_len=1000,
    )
    scheduler = Scheduler(config)
    seq1 = Sequence([i for i in range(7) for _ in range(block_size)], block_size)
    scheduler.add(seq1)
    seqs, is_prefill = scheduler.schedule()  # only seq1 in queue, rotate has no effect
    assert seqs == []
    assert not is_prefill

    # 2 new seqs
    seq2 = Sequence([i for i in range(3) for _ in range(block_size)], block_size)
    scheduler.add(seq2)
    seq3 = Sequence([i for i in range(3, 4) for _ in range(block_size)], block_size)
    scheduler.add(seq3)
    # seq1 is still at start of queue

    assert scheduler.waiting == deque([seq1, seq2, seq3])
    seqs, is_prefill = scheduler.schedule()  # will rotate seq1 to the end of queue
    assert seqs == []
    assert not is_prefill
    assert scheduler.waiting == deque([seq2, seq3, seq1])
    # seq2 do prefilling, then seq3 exceed, and rotate
    seqs, is_prefill = scheduler.schedule()
    seq2.append_token(100)  # simulate prefilling phase to generate one token

    assert seqs == [seq2]
    assert is_prefill
    assert scheduler.waiting == deque([seq1, seq3])
    seqs, is_prefill = scheduler.schedule()
    assert seqs == [seq2]
    assert not is_prefill
    assert scheduler.waiting == deque([seq3, seq1])


def test_preemption():
    block_size = 256
    num_blocks = 10
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    config = Config(model=path, kvcache_block_size=block_size, num_kvcache_blocks=num_blocks)
    scheduler = Scheduler(config)

    # seq1 has 7 full blocks and 1 partial block, is running
    seq1 = Sequence([i for i in range(7) for _ in range(block_size)], block_size)
    scheduler.add(seq1)
    scheduler.schedule()
    seq1.append_token(100)

    assert scheduler.running == deque([seq1])
    assert scheduler.block_manager.free_block_ids == deque([7, 8, 9])

    # seq2 has 2 full blocks, is running
    seq2 = Sequence([i for i in range(7, 9) for _ in range(block_size)], block_size)
    scheduler.add(seq2)
    scheduler.schedule()
    seq2.append_token(101)

    assert scheduler.running == deque([seq1, seq2])
    assert scheduler.block_manager.free_block_ids == deque([9])

    # seq3 has 2 full blocks, and a new token, preemption will happen
    scheduler.schedule()
