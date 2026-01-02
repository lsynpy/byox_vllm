# Eagle Data Flow for Single Batch

## iter 1

```cpp
schedule result:
  scheduled_new_reqs: [NewRequestData(req_id=0,prompt_token_ids=[852, 279, 1156, 5779, 10250, 5109, 25],block_ids=([1],),)]
  scheduled_cached_reqs: CachedRequestData(req_ids=[],num_computed_tokens=[],)
  num_scheduled_tokens: {'0': 7}
  total_num_scheduled_tokens: 7
  scheduled_spec_decode_tokens: {}
  num_common_prefix_blocks: [1]
  finished_req_ids: set()
_prepare_input() for normal get:
  logits_indices: [6]
  num_sampled_tokens: [1]
_model_forward() on:
  input_ids: [852, 279, 1156, 5779, 10250, 5109, 25]
  positions: [0, 1, 2, 3, 4, 5, 6]
call flash_attn with params:
  q.shape: torch.Size([7, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 7
  cu_seqlens_q: tensor([0, 7], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 7
  seqused_k: (7,)
  max_seqlen_k: 7
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([7, 16, 128])
eagle target model forward get:
  hidden_states: torch.Size([7, 2048])
  aux_hidden_states: 3 x torch.Size([7, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([1, 2048])
  logits: torch.Size([1, 151936])
prepare_next_token_ids_padded:
  next_token_ids: [220]
  valid_sampled_tokens_count: [1]
propose inputs:
  target_token_ids: [852, 279, 1156, 5779, 10250, 5109, 25]
  target_positions: [0, 1, 2, 3, 4, 5, 6]
  target_hidden_states: torch.Size([7, 6144])
  next_token_ids: [220]
  last_token_indices: None
set last_token_indices to [6]
--------------------------------------------------
draft forward inputs:
  input_ids: [279, 1156, 5779, 10250, 5109, 25, 220]
  positions: [0, 1, 2, 3, 4, 5, 6]
  hidden_states: torch.Size([7, 2048])
draft forward. get draft_token_ids: [220]
--------------------------------------------------
draft forward inputs:
  input_ids: [220]
  positions: [7]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (8,)
  max_seqlen_k: 7
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 0. get draft_token_ids: [279]
--------------------------------------------------
draft forward inputs:
  input_ids: [279]
  positions: [8]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (9,)
  max_seqlen_k: 7
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 1. get draft_token_ids: [279]
--------------------------------------------------
proposed draft_token_ids: [[220, 279, 279]]
appended new tokens to req-0, tokens: [220]
```

## iter 2

```cpp
schedule result:
  scheduled_new_reqs: []
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[7],)
  num_scheduled_tokens: {'0': 4}
  total_num_scheduled_tokens: 4
  scheduled_spec_decode_tokens: {'0': [220, 279, 279]}
  num_common_prefix_blocks: [1]
  finished_req_ids: set()
_prepare_input() for SD get: logits_indices: [0, 1, 2, 3], num_sampled_tokens: [4]
_model_forward() on:
  input_ids: [220, 220, 279, 279]
  positions: [7, 8, 9, 10]
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 4
  seqused_k: (11,)
  max_seqlen_k: 11
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
eagle target model forward get:
  hidden_states: torch.Size([4, 2048])
  aux_hidden_states: 3 x torch.Size([4, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([4, 2048])
  logits: torch.Size([4, 151936])
rejection sampling: draft_token_ids: [220, 279, 279], output_token_ids: [[17, -1, -1, -1]]
prepare_next_token_ids_padded:
  next_token_ids: [17]
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
  target_token_ids: [220, 220, 279, 279]
  target_positions: [7, 8, 9, 10]
  target_hidden_states: torch.Size([4, 6144])
  next_token_ids: [17]
  last_token_indices: [0]
--------------------------------------------------
draft forward inputs:
  input_ids: [17, 279, 279, 10250]
  positions: [7, 8, 9, 10]
  hidden_states: torch.Size([4, 2048])
draft forward. get draft_token_ids: [11]
--------------------------------------------------
draft forward inputs:
  input_ids: [11]
  positions: [8]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (12,)
  max_seqlen_k: 11
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 0. get draft_token_ids: [279]
--------------------------------------------------
draft forward inputs:
  input_ids: [279]
  positions: [9]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (13,)
  max_seqlen_k: 11
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 1. get draft_token_ids: [279]
--------------------------------------------------
proposed draft_token_ids: [[11, 279, 279]]
appended new tokens to req-0, tokens: [17]
```

## iter 3

```cpp
schedule result:
  scheduled_new_reqs: []
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[8],)
  num_scheduled_tokens: {'0': 4}
  total_num_scheduled_tokens: 4
  scheduled_spec_decode_tokens: {'0': [11, 279, 279]}
  num_common_prefix_blocks: [1]
  finished_req_ids: set()
_prepare_input() for SD get: logits_indices: [0, 1, 2, 3], num_sampled_tokens: [4]
_model_forward() on:
  input_ids: [17, 11, 279, 279]
  positions: [8, 9, 10, 11]
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 4
  seqused_k: (12,)
  max_seqlen_k: 12
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
eagle target model forward get:
  hidden_states: torch.Size([4, 2048])
  aux_hidden_states: 3 x torch.Size([4, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([4, 2048])
  logits: torch.Size([4, 151936])
rejection sampling: draft_token_ids: [11, 279, 279], output_token_ids: [[11, 220, -1, -1]]
prepare_next_token_ids_padded:
  next_token_ids: [220]
  valid_sampled_tokens_count: [2]
prepare_inputs_padded before kernel:
  cu_num_draft_tokens: [3]
  valid_sampled_tokens_count: [2]
  query_start_loc: [0, 4]
  token_indices_to_sample: [19]
  num_reqs: 1
prepare_inputs_padded after kernel:
  cu_num_draft_tokens: [3]
  valid_sampled_tokens_count: [2]
  query_start_loc: [0, 4]
  token_indices_to_sample: [1]
  num_reqs: 1
propose inputs:
  target_token_ids: [17, 11, 279, 279]
  target_positions: [8, 9, 10, 11]
  target_hidden_states: torch.Size([4, 6144])
  next_token_ids: [220]
  last_token_indices: [1]
--------------------------------------------------
draft forward inputs:
  input_ids: [11, 220, 279, 10250]
  positions: [8, 9, 10, 11]
  hidden_states: torch.Size([4, 2048])
draft forward. get draft_token_ids: [18]
--------------------------------------------------
draft forward inputs:
  input_ids: [18]
  positions: [10]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (13,)
  max_seqlen_k: 12
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 0. get draft_token_ids: [11]
--------------------------------------------------
draft forward inputs:
  input_ids: [11]
  positions: [11]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (14,)
  max_seqlen_k: 12
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 1. get draft_token_ids: [220]
--------------------------------------------------
proposed draft_token_ids: [[18, 11, 220]]
appended new tokens to req-0, tokens: [11, 220]
```

## iter 4

```cpp
schedule result:
  scheduled_new_reqs: []
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[10],)
  num_scheduled_tokens: {'0': 4}
  total_num_scheduled_tokens: 4
  scheduled_spec_decode_tokens: {'0': [18, 11, 220]}
  num_common_prefix_blocks: [2]
  finished_req_ids: set()
_prepare_input() for SD get: logits_indices: [0, 1, 2, 3], num_sampled_tokens: [4]
_model_forward() on:
  input_ids: [220, 18, 11, 220]
  positions: [10, 11, 12, 13]
call flash_attn with params:
  q.shape: torch.Size([4, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 4
  cu_seqlens_q: tensor([0, 4], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 4
  seqused_k: (14,)
  max_seqlen_k: 14
  block_table: tensor([[1, 2]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([4, 16, 128])
eagle target model forward get:
  hidden_states: torch.Size([4, 2048])
  aux_hidden_states: 3 x torch.Size([4, 2048])
use logits_indices get:
  sample_hidden_states: torch.Size([4, 2048])
  logits: torch.Size([4, 151936])
rejection sampling: draft_token_ids: [18, 11, 220], output_token_ids: [[18, 11, 220, 20]]
prepare_next_token_ids_padded:
  next_token_ids: [20]
  valid_sampled_tokens_count: [4]
prepare_inputs_padded before kernel:
  cu_num_draft_tokens: [3]
  valid_sampled_tokens_count: [4]
  query_start_loc: [0, 4]
  token_indices_to_sample: [20]
  num_reqs: 1
prepare_inputs_padded after kernel:
  cu_num_draft_tokens: [3]
  valid_sampled_tokens_count: [4]
  query_start_loc: [0, 4]
  token_indices_to_sample: [3]
  num_reqs: 1
propose inputs:
  target_token_ids: [220, 18, 11, 220]
  target_positions: [10, 11, 12, 13]
  target_hidden_states: torch.Size([4, 6144])
  next_token_ids: [20]
  last_token_indices: [3]
--------------------------------------------------
draft forward inputs:
  input_ids: [18, 11, 220, 20]
  positions: [10, 11, 12, 13]
  hidden_states: torch.Size([4, 2048])
draft forward. get draft_token_ids: [11]
--------------------------------------------------
draft forward inputs:
  input_ids: [11]
  positions: [14]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (15,)
  max_seqlen_k: 14
  block_table: tensor([[1, 2]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 0. get draft_token_ids: [220]
--------------------------------------------------
draft forward inputs:
  input_ids: [220]
  positions: [15]
  hidden_states: torch.Size([1, 2048])
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128])
  k.shape: torch.Size([156, 16, 8, 128])
  v.shape: torch.Size([156, 16, 8, 128])
  number_actual_tokens: 1
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32)
  max_seqlen_q: 1
  seqused_k: (16,)
  max_seqlen_k: 14
  block_table: tensor([[1, 2]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 1. get draft_token_ids: [220]
--------------------------------------------------
proposed draft_token_ids: [[11, 220, 220]]
appended new tokens to req-0, tokens: [18, 11, 220, 20]
```
