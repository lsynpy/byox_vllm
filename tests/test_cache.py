from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def test_prefill():
    block_size = 16
    num_blocks = 10
    manager = BlockManager(num_blocks, block_size)

    common_token_ids = [i for i in range(3) for _ in range(16)]  # Complete 3 blocks [0..0,1..1,2..2]
    unique_token_ids = [3] * 4  # Incomplete 1 block (4 tokens)
    all_token_ids = common_token_ids + unique_token_ids

    ########################################################################################
    # 1st seq: cache miss
    seq0 = Sequence(all_token_ids, block_size)
    manager.allocate(seq0)

    assert seq0.num_blocks == 4
    assert seq0.block_table == [0, 1, 2, 3]
    assert seq0.num_cached_tokens == 0
    assert seq0.num_cached_blocks == 0
    assert seq0.num_completion_tokens == 0

    assert len(manager.used_block_ids) == 4  # 3 common + 1 unique
    assert manager.used_block_ids == {0, 1, 2, 3}
    assert manager.free_block_ids == deque([4, 5, 6, 7, 8, 9])

    # Check full block metadata
    parent_block_hash = -1
    for block_id in (0, 1, 2):
        block_tokens = tuple(all_token_ids[(block_id) * 16 : (block_id + 1) * 16])
        # block_hash = hash_block_tokens(hash_fn, parent_block_hash, block_tokens)
        block_hash = manager._compute_hash(block_tokens, parent_block_hash)
        assert block_hash == manager.blocks[block_id].hash
        assert manager.blocks[block_id].ref_count == 1
        parent_block_hash = block_hash

    # Check partial block metadata
    for block_id in (3,):
        assert manager.blocks[block_id].hash is -1
        assert manager.blocks[block_id].ref_count == 1

    ########################################################################################
    # 2nd seq: cache hit in the common prefix when the original block is still in use.
    unique_token_ids = [3] * 5  # Incomplete 1 block (5 tokens)
    seq2 = Sequence(common_token_ids + unique_token_ids, block_size)
    manager.allocate(seq2)

    assert seq2.num_blocks == 4
    assert seq2.block_table == [0, 1, 2, 4]
    assert seq2.num_cached_tokens == 16 * 3  # 3 full blocks cached
    assert seq2.num_cached_blocks == 3
    assert seq2.num_completion_tokens == 0

    # At this point, we should have 5 free blocks left.
    assert len(manager.used_block_ids) == 5  # 3 common + 2 unique
    assert manager.used_block_ids == {0, 1, 2, 3, 4}
    assert manager.free_block_ids == deque([5, 6, 7, 8, 9])

    # Check full block metadata
    for block_id in (0, 1, 2):
        assert manager.blocks[block_id].ref_count == 2

    # Check partial block metadata
    for block_id in (4,):
        assert manager.blocks[block_id].hash is -1
        assert manager.blocks[block_id].ref_count == 1

    manager.deallocate(seq0)
    assert manager.free_block_ids == deque([5, 6, 7, 8, 9, 3])
    manager.deallocate(seq2)
    assert manager.free_block_ids == deque([5, 6, 7, 8, 9, 3, 4, 2, 1, 0])

    ########################################################################################
    # 3rd seq: cache hit in the common prefix when the original block is already free.
    unique_token_ids = [3] * 6
    seq3 = Sequence(common_token_ids + unique_token_ids, block_size)
    manager.allocate(seq3)

    assert seq3.num_blocks == 4
    assert seq3.block_table == [0, 1, 2, 5]
    assert seq3.num_cached_tokens == 16 * 3  # 3 full blocks cached
    assert seq3.num_cached_blocks == 3
    assert seq3.num_completion_tokens == 0

    assert len(manager.used_block_ids) == 4  # 3 common + 1 unique
    assert manager.used_block_ids == {0, 1, 2, 5}
    assert manager.free_block_ids == deque([6, 7, 8, 9, 3, 4])
