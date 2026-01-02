# Eagle data flow

## iter 1

```cpp
schedule result:
  scheduled_new_reqs: [NewRequestData(req_id=0,prompt_token_ids=[852, 279, 1156, 5779, 10250, 5109, 25],block_ids=([1],),), NewRequestData(req_id=1,prompt_token_ids=[785, 6722, 315, 9625, 374],block_ids=([2],),)]
  scheduled_cached_reqs: CachedRequestData(req_ids=[],num_computed_tokens=[],)
  num_scheduled_tokens: {'0': 7, '1': 5}
  total_num_scheduled_tokens: 12
  scheduled_spec_decode_tokens: {}
  num_common_prefix_blocks: [0]
  finished_req_ids: set()

_prepare_input() for normal get:
  logits_indices: [6, 11]
  num_sampled_tokens: [1 1]

_model_forward() on:
  input_ids: [852, 279, 1156, 5779, 10250, 5109, 25, 785, 6722, 315, 9625, 374]
  positions: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]

call flash_attn with params:
  q.shape: torch.Size([12, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 12
  cu_seqlens_q: tensor([ 0,  7, 12], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 7
  seqused_k: (7, 5)
  max_seqlen_k: 7
  block_table: tensor([[1, 0],
        [2, 0]], device='cuda:0', dtype=torch.int32)

flash_attn out: torch.Size([12, 16, 128])

_model_forward() get: hidden_states: torch.Size([12, 2048])

use logits_indices get:
  sample_hidden_states: torch.Size([2, 2048])
  logits: torch.Size([2, 151936])

prepare_next_token_ids_padded:
  next_token_ids: [220, 12095]
  valid_sampled_tokens_count: [1, 1]

propose inputs:
  target_token_ids: [852, 279, 1156, 5779, 10250, 5109, 25, 785, 6722, 315, 9625, 374]
  target_positions: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]
  target_hidden_states: torch.Size([12, 6144])
  next_token_ids: [220, 12095]
  last_token_indices: None

set last_token_indices to [6, 11]

--------------------------------------------------
draft forward inputs:
  input_ids: [279, 1156, 5779, 10250, 5109, 25, 220, 6722, 315, 9625, 374, 12095]
  positions: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4]
  hidden_states: torch.Size([12, 2048])
draft forward:
  draft_token_ids: [220, 374]
--------------------------------------------------
draft forward inputs:
  input_ids: [220, 374]
  positions: [7, 5]
  hidden_states: torch.Size([2, 2048])
call flash_attn with params:
  q.shape: torch.Size([2, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 2
  cu_seqlens_q: tensor([0, 1, 2], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (8, 6)
  max_seqlen_k: 7
  block_table: tensor([[1, 0],
        [2, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([2, 16, 128])
draft forward 0:
  draft_token_ids: [220, 12095]
--------------------------------------------------
draft forward inputs:
  input_ids: [220, 12095]
  positions: [8, 6]
  hidden_states: torch.Size([2, 2048])
call flash_attn with params:
  q.shape: torch.Size([2, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 2
  cu_seqlens_q: tensor([0, 1, 2], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (9, 7)
  max_seqlen_k: 7
  block_table: tensor([[1, 0],
        [2, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([2, 16, 128])
draft forward 1: draft_token_ids: [220, 12095]
--------------------------------------------------
proposed draft_token_ids: [[220, 220, 220], [374, 12095, 12095]]
appended new tokens to req-0. tokens: [220]
appended new tokens to req-1. tokens: [12095]
```

## iter 2

```cpp
schedule result:
  scheduled_new_reqs: [
    NewRequestData(req_id=2,prompt_token_ids=[12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11],block_ids=([3],),),
    NewRequestData(req_id=3,prompt_token_ids=[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25],block_ids=([4],),)]
  scheduled_cached_reqs: CachedRequestData(req_ids=['0', '1'],num_computed_tokens=[7, 5],)
  num_scheduled_tokens: {'0': 4, '1': 4, '2': 12, '3': 11}
  total_num_scheduled_tokens: 31
  scheduled_spec_decode_tokens: {'0': [220, 220, 220], '1': [374, 12095, 12095]}
  num_common_prefix_blocks: [0]
  finished_req_ids: set()
_prepare_input() for SD get: logits_indices: [0, 1, 2, 3, 4, 5, 6, 7, 19, 30], num_sampled_tokens: [4 4 1 1]
_model_forward() on:
  input_ids: [220, 220, 220, 220, 12095, 374, 12095, 12095, 12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25]
  positions: [7, 8, 9, 10, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
call flash_attn with params:
  q.shape: torch.Size([31, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 31
  cu_seqlens_q: tensor([ 0,  4,  8, 20, 31], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 12
  seqused_k: (11, 9, 12, 11)
  max_seqlen_k: 12
  block_table: tensor([[1, 0],
        [2, 0],
        [3, 0],
        [4, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([31, 16, 128])
_model_forward() get: hidden_states: torch.Size([31, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([10, 2048])
  logits: torch.Size([10, 151936])
rejection sampling: draft_token_ids: [220, 220, 220, 374, 12095, 12095], output_token_ids: [[17, -1, -1, -1], [13, -1, -1, -1], [1052, -1, -1, -1], [220, -1, -1, -1]]
prepare_next_token_ids_padded:
  next_token_ids: [17, 13, 1052, 220]
  valid_sampled_tokens_count: [1, 1, 1, 1]
prepare_inputs_padded before kernel:
  cu_num_draft_tokens: [3, 6, 6, 6]
  valid_sampled_tokens_count: [1, 1, 1, 1]
  query_start_loc: [0, 4, 8, 20, 31]
  token_indices_to_sample: [10, 10, 10, 10]
  num_reqs: 4
prepare_inputs_padded after kernel:
  cu_num_draft_tokens: [3, 6, 6, 6]
  valid_sampled_tokens_count: [1, 1, 1, 1]
  query_start_loc: [0, 4, 8, 20, 31]
  token_indices_to_sample: [0, 4, 19, 30]
  num_reqs: 4
propose inputs:
  target_token_ids: [220, 220, 220, 220, 12095, 374, 12095, 12095, 12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25]
  target_positions: [7, 8, 9, 10, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  target_hidden_states: torch.Size([31, 6144])
  next_token_ids: [17, 13, 1052, 220]
  last_token_indices: [0, 4, 19, 30]
--------------------------------------------------
draft forward inputs:
  input_ids: [17, 220, 220, 12095, 13, 12095, 12095, 12522, 5193, 264, 882, 304, 264, 4268, 3041, 11, 3041, 3123, 11, 1052, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220]
  positions: [7, 8, 9, 10, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  hidden_states: torch.Size([31, 2048])
draft forward, get draft_token_ids: [11, 9625, 1052, 16]
--------------------------------------------------
draft forward inputs:
  input_ids: [11, 9625, 1052, 16]
  positions: [8, 6, 12, 11]
  hidden_states: torch.Size([4, 2048])
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (12, 10, 13, 12)
  max_seqlen_k: 12
  block_table: tensor([[1, 0],
        [2, 0],
        [3, 0],
        [4, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
draft forward 0, get draft_token_ids: [220, 13, 1052, 15]
--------------------------------------------------
draft forward inputs:
  input_ids: [220, 13, 1052, 15]
  positions: [9, 7, 13, 12]
  hidden_states: torch.Size([4, 2048])
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (13, 11, 14, 13)
  max_seqlen_k: 12
  block_table: tensor([[1, 0],
        [2, 0],
        [3, 0],
        [4, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
draft forward 1, get draft_token_ids: [18, 12095, 11, 271]
--------------------------------------------------
proposed draft_token_ids: [[11, 220, 18], [9625, 13, 12095], [1052, 1052, 11], [16, 15, 271]]
appended new tokens to req-0, tokens: [17]
appended new tokens to req-1, tokens: [13]
appended new tokens to req-2, tokens: [1052]
appended new tokens to req-3, tokens: [220]

```

## iter 3

```cpp
schedule result:
  scheduled_new_reqs: []
  scheduled_cached_reqs: CachedRequestData(req_ids=['0', '1', '2', '3'],num_computed_tokens=[8, 6, 12, 11],)
  num_scheduled_tokens: {'0': 4, '1': 4, '2': 4, '3': 4}
  total_num_scheduled_tokens: 16
  scheduled_spec_decode_tokens: {'0': [11, 220, 18], '1': [9625, 13, 12095], '2': [1052, 1052, 11], '3': [16, 15, 271]}
  num_common_prefix_blocks: [0]
  finished_req_ids: set()
_prepare_input() for SD get: logits_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], num_sampled_tokens: [4 4 4 4]
_model_forward() on:
  input_ids: [17, 11, 220, 18, 13, 9625, 13, 12095, 1052, 1052, 1052, 11, 220, 16, 15, 271]
  positions: [8, 9, 10, 11, 6, 7, 8, 9, 12, 13, 14, 15, 11, 12, 13, 14]
call flash_attn with params:
  q.shape: torch.Size([16, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 16
  cu_seqlens_q: tensor([ 0,  4,  8, 12, 16], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 4
  seqused_k: (12, 10, 16, 15)
  max_seqlen_k: 16
  block_table: tensor([[1, 0],
        [2, 0],
        [3, 5],
        [4, 6]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([16, 16, 128])
_model_forward() get: hidden_states: torch.Size([16, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([16, 2048])
  logits: torch.Size([16, 151936])
rejection sampling: draft_token_ids: [11, 220, 18, 9625, 13, 12095, 1052, 1052, 11, 16, 15, 271], output_token_ids: [[11, 220, 18, 11], [576, -1, -1, -1], [572, -1, -1, -1], [16, 11, -1, -1]]
prepare_next_token_ids_padded:
  next_token_ids: [11, 576, 572, 11]
  valid_sampled_tokens_count: [4, 1, 1, 2]
prepare_inputs_padded before kernel:
  cu_num_draft_tokens: [3, 6, 9, 12]
  valid_sampled_tokens_count: [4, 1, 1, 2]
  query_start_loc: [0, 4, 8, 12, 16]
  token_indices_to_sample: [5, 5, 5, 5]
  num_reqs: 4
prepare_inputs_padded after kernel:
  cu_num_draft_tokens: [3, 6, 9, 12]
  valid_sampled_tokens_count: [4, 1, 1, 2]
  query_start_loc: [0, 4, 8, 12, 16]
  token_indices_to_sample: [3, 4, 8, 13]
  num_reqs: 4
propose inputs:
  target_token_ids: [17, 11, 220, 18, 13, 9625, 13, 12095, 1052, 1052, 1052, 11, 220, 16, 15, 271]
  target_positions: [8, 9, 10, 11, 6, 7, 8, 9, 12, 13, 14, 15, 11, 12, 13, 14]
  target_hidden_states: torch.Size([16, 6144])
  next_token_ids: [11, 576, 572, 11]
  last_token_indices: [3, 4, 8, 13]
--------------------------------------------------
draft forward inputs:
  input_ids: [11, 220, 18, 11, 576, 13, 12095, 1052, 572, 1052, 11, 220, 16, 11, 271, 11]
  positions: [8, 9, 10, 11, 6, 7, 8, 9, 12, 13, 14, 15, 11, 12, 13, 14]
  hidden_states: torch.Size([16, 2048])
draft forward. get draft_token_ids: [220, 374, 264, 220]
--------------------------------------------------
draft forward inputs:
  input_ids: [220, 374, 264, 220]
  positions: [12, 7, 13, 13]
  hidden_states: torch.Size([4, 2048])
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (13, 11, 17, 16)
  max_seqlen_k: 16
  block_table: tensor([[1, 0],
        [2, 0],
        [3, 5],
        [4, 6]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
draft forward 0. get draft_token_ids: [19, 374, 264, 16]
--------------------------------------------------
--------------------------------------------------
draft forward inputs:
  input_ids: [19, 374, 264, 16]
  positions: [13, 8, 14, 14]
  hidden_states: torch.Size([4, 2048])
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (14, 12, 18, 17)
  max_seqlen_k: 16
  block_table: tensor([[1, 0],
        [2, 0],
        [3, 5],
        [4, 6]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
draft forward 1. get draft_token_ids: [11, 315, 1052, 11]
--------------------------------------------------
proposed draft_token_ids: [[220, 19, 11], [374, 374, 315], [264, 264, 1052], [220, 16, 11]]
appended new tokens to req-0, tokens: [11, 220, 18, 11]
appended new tokens to req-1, tokens: [576]
appended new tokens to req-2, tokens: [572]
appended new tokens to req-3, tokens: [16, 11]
```

## iter 4

```cpp
schedule result:
  scheduled_new_reqs: []
  scheduled_cached_reqs: CachedRequestData(req_ids=['0', '1', '2', '3'],num_computed_tokens=[12, 7, 13, 13],)
  num_scheduled_tokens: {'0': 4, '1': 4, '2': 4, '3': 4}
  total_num_scheduled_tokens: 16
  scheduled_spec_decode_tokens: {'0': [220, 19, 11], '1': [374, 374, 315], '2': [264, 264, 1052], '3': [220, 16, 11]}
  num_common_prefix_blocks: [0]
  finished_req_ids: set()
_prepare_input() for SD get: logits_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], num_sampled_tokens: [4 4 4 4]
_model_forward() on:
  input_ids: [11, 220, 19, 11, 576, 374, 374, 315, 572, 264, 264, 1052, 11, 220, 16, 11]
  positions: [12, 13, 14, 15, 7, 8, 9, 10, 13, 14, 15, 16, 13, 14, 15, 16]
call flash_attn with params:
  q.shape: torch.Size([16, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 16
  cu_seqlens_q: tensor([ 0,  4,  8, 12, 16], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 4
  seqused_k: (16, 11, 17, 17)
  max_seqlen_k: 17
  block_table: tensor([[1, 7],
        [2, 0],
        [3, 5],
        [4, 6]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([16, 16, 128])
_model_forward() get: hidden_states: torch.Size([16, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([16, 2048])
  logits: torch.Size([16, 151936])
rejection sampling: draft_token_ids: [220, 19, 11, 374, 374, 315, 264, 264, 1052, 220, 16, 11], output_token_ids: [[220, 20, -1, -1], [6722, -1, -1, -1], [264, 3908, -1, -1], [220, 16, 16, -1]]
prepare_next_token_ids_padded:
  next_token_ids: [20, 6722, 3908, 16]
  valid_sampled_tokens_count: [2, 1, 2, 3]
prepare_inputs_padded before kernel:
  cu_num_draft_tokens: [3, 6, 9, 12]
  valid_sampled_tokens_count: [2, 1, 2, 3]
  query_start_loc: [0, 4, 8, 12, 16]
  token_indices_to_sample: [220, 279, 572, 220]
  num_reqs: 4
prepare_inputs_padded after kernel:
  cu_num_draft_tokens: [3, 6, 9, 12]
  valid_sampled_tokens_count: [2, 1, 2, 3]
  query_start_loc: [0, 4, 8, 12, 16]
  token_indices_to_sample: [1, 4, 9, 14]
  num_reqs: 4
propose inputs:
  target_token_ids: [11, 220, 19, 11, 576, 374, 374, 315, 572, 264, 264, 1052, 11, 220, 16, 11]
  target_positions: [12, 13, 14, 15, 7, 8, 9, 10, 13, 14, 15, 16, 13, 14, 15, 16]
  target_hidden_states: torch.Size([16, 6144])
  next_token_ids: [20, 6722, 3908, 16]
  last_token_indices: [1, 4, 9, 14]
--------------------------------------------------
draft forward inputs:
  input_ids: [220, 20, 11, 576, 6722, 374, 315, 572, 264, 3908, 1052, 11, 220, 16, 16, 11]
  positions: [12, 13, 14, 15, 7, 8, 9, 10, 13, 14, 15, 16, 13, 14, 15, 16]
  hidden_states: torch.Size([16, 2048])
draft forward. get draft_token_ids: [11, 315, 3908, 11]
--------------------------------------------------
draft forward inputs:
  input_ids: [11, 315, 3908, 11]
  positions: [14, 8, 15, 16]
  hidden_states: torch.Size([4, 2048])
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (17, 12, 18, 18)
  max_seqlen_k: 17
  block_table: tensor([[1, 7],
        [2, 0],
        [3, 5],
        [4, 6]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
draft forward 0. get draft_token_ids: [220, 279, 5193, 220]
--------------------------------------------------
draft forward inputs:
  input_ids: [220, 279, 5193, 220]
  positions: [15, 9, 16, 17]
  hidden_states: torch.Size([4, 2048])
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (18, 13, 19, 19)
  max_seqlen_k: 17
  block_table: tensor([[1, 7],
        [2, 0],
        [3, 5],
        [4, 6]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
draft forward 1. get draft_token_ids: [22, 315, 264, 220]
--------------------------------------------------
proposed draft_token_ids: [[11, 220, 22], [315, 279, 315], [3908, 5193, 264], [11, 220, 220]]
appended new tokens to req-0, tokens: [220, 20]
appended new tokens to req-1, tokens: [6722]
appended new tokens to req-2, tokens: [264, 3908]
appended new tokens to req-3, tokens: [220, 16, 16]

```

## iter 5

```cpp
schedule result:
  scheduled_new_reqs: []
  scheduled_cached_reqs: CachedRequestData(req_ids=['1', '2', '3'],num_computed_tokens=[8, 15, 16],)
  num_scheduled_tokens: {'1': 4, '2': 4, '3': 4}
  total_num_scheduled_tokens: 12
  scheduled_spec_decode_tokens: {'1': [315, 279, 315], '2': [3908, 5193, 264], '3': [11, 220, 220]}
  num_common_prefix_blocks: [0]
  finished_req_ids: {'0'}
_prepare_input() for SD get: logits_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], num_sampled_tokens: [4 4 4]
_model_forward() on:
  input_ids: [16, 11, 220, 220, 6722, 315, 279, 315, 3908, 3908, 5193, 264]
  positions: [16, 17, 18, 19, 8, 9, 10, 11, 15, 16, 17, 18]
call flash_attn with params:
  q.shape: torch.Size([12, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 12
  cu_seqlens_q: tensor([ 0,  4,  8, 12], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 4
  seqused_k: (20, 12, 19)
  max_seqlen_k: 20
  block_table: tensor([[4, 6],
        [2, 0],
        [3, 5]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([12, 16, 128])
_model_forward() get: hidden_states: torch.Size([12, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([12, 2048])
  logits: torch.Size([12, 151936])
rejection sampling: draft_token_ids: [11, 220, 220, 315, 279, 315, 3908, 5193, 264], output_token_ids: [[11, 220, 16, -1], [315, 279, 3639, -1], [3743, -1, -1, -1]]
prepare_next_token_ids_padded:
  next_token_ids: [16, 3639, 3743]
  valid_sampled_tokens_count: [3, 3, 1]
prepare_inputs_padded before kernel:
  cu_num_draft_tokens: [3, 6, 9]
  valid_sampled_tokens_count: [3, 3, 1]
  query_start_loc: [0, 4, 8, 12]
  token_indices_to_sample: [7, 7, 7]
  num_reqs: 3
prepare_inputs_padded after kernel:
  cu_num_draft_tokens: [3, 6, 9]
  valid_sampled_tokens_count: [3, 3, 1]
  query_start_loc: [0, 4, 8, 12]
  token_indices_to_sample: [2, 6, 8]
  num_reqs: 3
propose inputs:
  target_token_ids: [16, 11, 220, 220, 6722, 315, 279, 315, 3908, 3908, 5193, 264]
  target_positions: [16, 17, 18, 19, 8, 9, 10, 11, 15, 16, 17, 18]
  target_hidden_states: torch.Size([12, 6144])
  next_token_ids: [16, 3639, 3743]
  last_token_indices: [2, 6, 8]
--------------------------------------------------
draft forward inputs:
  input_ids: [11, 220, 16, 6722, 315, 279, 3639, 3908, 3743, 5193, 264, 11]
  positions: [16, 17, 18, 19, 8, 9, 10, 11, 15, 16, 17, 18]
  hidden_states: torch.Size([12, 2048])
draft forward. get draft_token_ids: [16, 374, 6941]
--------------------------------------------------
draft forward inputs:
  input_ids: [16, 374, 6941]
  positions: [19, 11, 16]
  hidden_states: torch.Size([3, 2048])
call flash_attn with params:
  q.shape: torch.Size([3, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 3
  cu_seqlens_q: tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (21, 13, 20)
  max_seqlen_k: 20
  block_table: tensor([[4, 6],
        [2, 0],
        [3, 5]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([3, 16, 128])
draft forward 0. get draft_token_ids: [16, 374, 6941]
--------------------------------------------------
draft forward inputs:
  input_ids: [16, 374, 6941]
  positions: [20, 12, 17]
  hidden_states: torch.Size([3, 2048])
call flash_attn with params:
  q.shape: torch.Size([3, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 3
  cu_seqlens_q: tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (22, 14, 21)
  max_seqlen_k: 20
  block_table: tensor([[4, 6],
        [2, 0],
        [3, 5]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([3, 16, 128])
draft forward 1. get draft_token_ids: [11, 304, 6941]
--------------------------------------------------
proposed draft_token_ids: [[16, 16, 11], [374, 374, 304], [6941, 6941, 6941]]
appended new tokens to req-1, tokens: [315, 279, 3639]
appended new tokens to req-2, tokens: [3743]
appended new tokens to req-3, tokens: [11, 220]
```

## iter 6

```cpp
schedule result:
  scheduled_new_reqs: []
  scheduled_cached_reqs: CachedRequestData(req_ids=['1', '2'],num_computed_tokens=[11, 16],)
  num_scheduled_tokens: {'1': 4, '2': 4}
  total_num_scheduled_tokens: 8
  scheduled_spec_decode_tokens: {'1': [374, 374, 304], '2': [6941, 6941, 6941]}
  num_common_prefix_blocks: [0]
  finished_req_ids: {'3'}
_prepare_input() for SD get: logits_indices: [0, 1, 2, 3, 4, 5, 6, 7], num_sampled_tokens: [4 4]
_model_forward() on:
  input_ids: [3743, 6941, 6941, 6941, 3639, 374, 374, 304]
  positions: [16, 17, 18, 19, 11, 12, 13, 14]
call flash_attn with params:
  q.shape: torch.Size([8, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 8
  cu_seqlens_q: tensor([0, 4, 8], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 4
  seqused_k: (20, 15)
  max_seqlen_k: 20
  block_table: tensor([[3, 5],
        [2, 8]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([8, 16, 128])
_model_forward() get: hidden_states: torch.Size([8, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([8, 2048])
  logits: torch.Size([8, 151936])
rejection sampling: draft_token_ids: [6941, 6941, 6941, 374, 374, 304], output_token_ids: [[6941, 444, -1, -1], [4180, -1, -1, -1]]
prepare_next_token_ids_padded:
  next_token_ids: [444, 4180]
  valid_sampled_tokens_count: [2, 1]
prepare_inputs_padded before kernel:
  cu_num_draft_tokens: [3, 6]
  valid_sampled_tokens_count: [2, 1]
  query_start_loc: [0, 4, 8]
  token_indices_to_sample: [10, 10]
  num_reqs: 2
prepare_inputs_padded after kernel:
  cu_num_draft_tokens: [3, 6]
  valid_sampled_tokens_count: [2, 1]
  query_start_loc: [0, 4, 8]
  token_indices_to_sample: [1, 4]
  num_reqs: 2
propose inputs:
  target_token_ids: [3743, 6941, 6941, 6941, 3639, 374, 374, 304]
  target_positions: [16, 17, 18, 19, 11, 12, 13, 14]
  target_hidden_states: torch.Size([8, 6144])
  next_token_ids: [444, 4180]
  last_token_indices: [1, 4]
--------------------------------------------------
draft forward inputs:
  input_ids: [6941, 444, 6941, 3639, 4180, 374, 304, 3908]
  positions: [16, 17, 18, 19, 11, 12, 13, 14]
  hidden_states: torch.Size([8, 2048])
draft forward. get draft_token_ids: [444, 374]
--------------------------------------------------
draft forward inputs:
  input_ids: [444, 374]
  positions: [18, 12]
  hidden_states: torch.Size([2, 2048])
call flash_attn with params:
  q.shape: torch.Size([2, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 2
  cu_seqlens_q: tensor([0, 1, 2], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (21, 16)
  max_seqlen_k: 20
  block_table: tensor([[3, 5],
        [2, 8]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([2, 16, 128])
draft forward 0. get draft_token_ids: [13, 304]
--------------------------------------------------
draft forward inputs:
  input_ids: [13, 304]
  positions: [19, 13]
  hidden_states: torch.Size([2, 2048])
call flash_attn with params:
  q.shape: torch.Size([2, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 2
  cu_seqlens_q: tensor([0, 1, 2], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (22, 17)
  max_seqlen_k: 20
  block_table: tensor([[3, 5],
        [2, 8]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([2, 16, 128])
draft forward 1. get draft_token_ids: [13, 279]
--------------------------------------------------
proposed draft_token_ids: [[444, 13, 13], [374, 304, 279]]
appended new tokens to req-1, tokens: [4180]
appended new tokens to req-2, tokens: [6941, 444]

```

## iter 7

```cpp
schedule result:
  scheduled_new_reqs: []
  scheduled_cached_reqs: CachedRequestData(req_ids=['2'],num_computed_tokens=[18],)
  num_scheduled_tokens: {'2': 4}
  total_num_scheduled_tokens: 4
  scheduled_spec_decode_tokens: {'2': [444, 13, 13]}
  num_common_prefix_blocks: [2]
  finished_req_ids: {'1'}
_prepare_input() for SD get: logits_indices: [0, 1, 2, 3], num_sampled_tokens: [4]
_model_forward() on:
  input_ids: [444, 444, 13, 13]
  positions: [18, 19, 20, 21]
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 4
  seqused_k: (22,)
  max_seqlen_k: 22
  block_table: tensor([[3, 5]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
_model_forward() get: hidden_states: torch.Size([4, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([4, 2048])
  logits: torch.Size([4, 151936])
rejection sampling: draft_token_ids: [444, 13, 13], output_token_ids: [[10524, -1, -1, -1]]
prepare_next_token_ids_padded:
  next_token_ids: [10524]
  valid_sampled_tokens_count: [1]
prepare_inputs_padded before kernel:
  cu_num_draft_tokens: [3]
  valid_sampled_tokens_count: [1]
  query_start_loc: [0, 4]
  token_indices_to_sample: [19]
  num_reqs: 1
prepare_inputs_padded after kernel:
  cu_num_draft_tokens: [3]
  valid_sampled_tokens_count: [1]
  query_start_loc: [0, 4]
  token_indices_to_sample: [0]
  num_reqs: 1
propose inputs:
  target_token_ids: [444, 444, 13, 13]
  target_positions: [18, 19, 20, 21]
  target_hidden_states: torch.Size([4, 6144])
  next_token_ids: [10524]
  last_token_indices: [0]
--------------------------------------------------
draft forward inputs:
  input_ids: [10524, 13, 13, 3639]
  positions: [18, 19, 20, 21]
  hidden_states: torch.Size([4, 2048])
draft forward. get draft_token_ids: [13]
--------------------------------------------------
draft forward inputs:
  input_ids: [13]
  positions: [19]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (23,)
  max_seqlen_k: 22
  block_table: tensor([[3, 5]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 0. get draft_token_ids: [13]
--------------------------------------------------
draft forward inputs:
  input_ids: [13]
  positions: [20]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (24,)
  max_seqlen_k: 22
  block_table: tensor([[3, 5]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 1. get draft_token_ids: [13]
--------------------------------------------------
proposed draft_token_ids: [[13, 13, 13]]
appended new tokens to req-2, tokens: [10524]
```
