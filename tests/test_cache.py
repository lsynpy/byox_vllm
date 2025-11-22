from collections import deque
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.logging import logger


def test_prefill():
    block_size = 16
    num_blocks = 11
    manager = BlockManager(num_blocks, block_size)

    common_token_ids = [i for i in range(3) for _ in range(16)]  # Complete 3 blocks
    unique_token_ids = [3] * 7  # Incomplete 1 block (7 tokens)
    all_token_ids = common_token_ids + unique_token_ids

    seq0 = Sequence(all_token_ids, block_size)
    manager.allocate(seq0)
    assert len(manager.used_block_ids) == 4  # 3 common + 1 unique
    assert manager.used_block_ids == {0, 1, 2, 3}
    assert seq0.num_cached_blocks == 0
    assert seq0.num_completion_tokens == 0
    assert manager.blocks is not None
    assert manager.free_block_ids == deque([4,5,6,7,8,9,10])

