import os
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.scheduler import DecodeType, Scheduler
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

    seq0 = Sequence(all_token_ids, block_size)
    scheduler.add(seq0)
    output = scheduler.schedule()
    seqs = output.scheduled_seqs
    is_prefill = output.decode_type == DecodeType.PREFILL
    assert seqs == [seq0]
    assert is_prefill

    seq1 = Sequence([i for i in range(7) for _ in range(block_size)], block_size)
    scheduler.add(seq1)
    output = scheduler.schedule()
    seqs = output.scheduled_seqs
    is_prefill = output.decode_type == DecodeType.PREFILL
    assert seqs == [seq0]
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
    seq0 = Sequence([i for i in range(7) for _ in range(block_size)], block_size)
    scheduler.add(seq0)
    output = scheduler.schedule()  # only seq0 in queue, rotate has no effect
    seqs = output.scheduled_seqs if output.scheduled_seqs else []
    is_prefill = output.decode_type == DecodeType.PREFILL
    assert seqs == []
    assert not is_prefill

    # 2 new seqs
    seq1 = Sequence([i for i in range(3) for _ in range(block_size)], block_size)
    scheduler.add(seq1)
    seq2 = Sequence([i for i in range(3, 4) for _ in range(block_size)], block_size)
    scheduler.add(seq2)
    # seq0 is still at start of queue

    assert scheduler.waiting == deque([seq0, seq1, seq2])
    output = scheduler.schedule()  # will rotate seq0 to the end of queue
    seqs = output.scheduled_seqs if output.scheduled_seqs else []
    is_prefill = output.decode_type == DecodeType.PREFILL
    assert seqs == []
    assert not is_prefill
    assert scheduler.waiting == deque([seq1, seq2, seq0])
    # seq1 do prefilling, then seq2 exceed, and rotate
    output = scheduler.schedule()
    seqs = output.scheduled_seqs if output.scheduled_seqs else []
    is_prefill = output.decode_type == DecodeType.PREFILL
    seq1.append_token(100)  # simulate prefilling phase to generate one token

    assert seqs == [seq1]
    assert is_prefill
    assert scheduler.waiting == deque([seq0, seq2])
    output = scheduler.schedule()
    seqs = output.scheduled_seqs if output.scheduled_seqs else []
    is_prefill = output.decode_type == DecodeType.PREFILL
    assert seqs == [seq1]
    assert not is_prefill
    assert scheduler.waiting == deque([seq2, seq0])


def test_preemption():
    block_size = 256
    num_blocks = 10
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    config = Config(model=path, kvcache_block_size=block_size, num_kvcache_blocks=num_blocks)
    scheduler = Scheduler(config)

    # seq0 has 7 full blocks and 1 partial block, is running
    seq0 = Sequence([i for i in range(7) for _ in range(block_size)], block_size)
    scheduler.add(seq0)
    scheduler.schedule()
    seq0.append_token(100)

    assert scheduler.running == deque([seq0])
    assert scheduler.block_manager.free_block_ids == deque([7, 8, 9])

    # seq1 has 2 full blocks, is running
    seq1 = Sequence([i for i in range(7, 9) for _ in range(block_size)], block_size)
    scheduler.add(seq1)
    scheduler.schedule()
    seq1.append_token(101)

    assert scheduler.running == deque([seq0, seq1])
    assert scheduler.block_manager.free_block_ids == deque([9])
    assert seq0.block_table == [0, 1, 2, 3, 4, 5, 6]
    assert seq1.block_table == [7, 8]

    # seq1 has 2 full blocks, and a new token, preemption will happen
    scheduler.schedule()
    # schedule() will loop twice.
    # First iteration seq0 is scheduled for decode, and will allocate a new block.
    # Second iteration, seq1 cannot append, preemption happened here.
    # In the second iteration, after pop seq1, the running queue is [] (seq0 popped in first iteration).
    # So seq1 itself will be preempted, and move to waiting queue, its blocks will be freed.
    # After schedule(), seq0 will be put back to running queue.
    assert scheduler.running == deque([seq0])
    assert scheduler.waiting == deque([seq1])
    assert scheduler.block_manager.free_block_ids == deque([8, 7])
    assert seq0.block_table == [0, 1, 2, 3, 4, 5, 6, 9]
    assert seq1.block_table == []
