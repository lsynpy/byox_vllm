def test_no_new_blocks():
    """
    for block_size = 256, 4 seqs, each seq 1 block.
    step 1:

    draft forward inputs:
      input_ids: [279, 1156, 5779, 10250, 5109, 25, 220,
                6722, 315, 9625, 374, 12095,
                5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 1052,
                220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220]
      positions: [0, 1, 2, 3, 4, 5, 6,
                0, 1, 2, 3, 4,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    slot_mapping: [0, 1, 2, 3, 4, 5, 6,
                256, 257, 258, 259, 260,
                512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523,
                768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778]
    draft forward. sampled draft token ids: [220, 12095, 1052, 220]

    step 2:

    draft forward inputs:
        input_ids: [220, 12095, 1052, 220]
        positions: [7, 5, 12, 11]
    slot_mapping: [7, 261, 524, 779]
    """


def test_new_blocks():
    """
    When do N draft forward, there will be situation new blocks need for draft kv cache.
    But target kv cache is not exist yet, so draft forward need to create new blocks.
    """
