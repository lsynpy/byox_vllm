## 1. Quick Start

```sh
uv pip install -e .
pytest
python example.py
python bench.py
```

## Eagle data flow

```py
schedule result:
  scheduled_new_reqs: [NewRequestData(req_id=0,prompt_token_ids=[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25])],
  scheduled_cached_reqs: CachedRequestData(req_ids=[],num_computed_tokens=[],),
  num_scheduled_tokens: {'0': 11},
  total_num_scheduled_tokens: 11,
  scheduled_spec_decode_tokens: {}

_prepare_input() for normal get:
  logits_indices: [10], num_sampled_tokens: [1]

_model_forward() on:
  input_ids: [852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25], positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

call flash_attn with params:
  q.shape: torch.Size([11, 16, 128]),
  k.shape: torch.Size([172, 16, 8, 128]),
  v.shape: torch.Size([172, 16, 8, 128]),
  number_actual_tokens: 11,
  cu_seqlens_q: tensor([ 0, 11], device='cuda:0', dtype=torch.int32),
  max_seqlen_q: 11,
  seqused_k: (11,),
  max_seqlen_k: 11,
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([11, 16, 128])

_model_forward() get: hidden_states: torch.Size([11, 2048])

use logits_indices get:
  sample_hidden_states: torch.Size([1, 2048]), logits: torch.Size([1, 151936])

prepare_next_token_ids_padded:
  next_token_ids: [220], valid_sampled_tokens_count: [1]

propose inputs:
  target_token_ids: [852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25]
  target_positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  target_hidden_states: torch.Size([11, 6144])
  next_token_ids: [220]
  last_token_indices: None
--------------------------------------------------
 draft forward inputs:
  input_ids: [220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220],
  positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  hidden_states: torch.Size([11, 2048]),
  inputs_embeds: None
 draft forward, get draft_token_ids: [16]
--------------------------------------------------
draft forward inputs:
  input_ids: [16],
  positions: [11],
  hidden_states: torch.Size([1, 2048]),
  inputs_embeds: None
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128]),
  k.shape: torch.Size([172, 16, 8, 128]),
  v.shape: torch.Size([172, 16, 8, 128]),
  number_actual_tokens: 1,
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32),
  max_seqlen_q: 1,
  seqused_k: (12,),
  max_seqlen_k: 11,
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
flash_attn out:
  torch.Size([1, 16, 128])
draft forward 0, get draft_token_ids: [15]
--------------------------------------------------
draft forward inputs:
  input_ids: [15],
  positions: [12],
  hidden_states: torch.Size([1, 2048]),
  inputs_embeds: None
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128]),
  k.shape: torch.Size([172, 16, 8, 128]),
  v.shape: torch.Size([172, 16, 8, 128]),
  number_actual_tokens: 1,
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32),
  max_seqlen_q: 1,
  seqused_k: (13,),
  max_seqlen_k: 11,
  block_table: tensor([[1, 0]], device='cuda:0', dtype=torch.int32)
 flash_attn out:
  torch.Size([1, 16, 128])
 draft forward 1, get draft_token_ids: [271]

 proposed draft_token_ids: [[16, 15, 271]]

 appended new tokens to req-0, tokens: [220]
```

```py
schedule result:
  scheduled_new_reqs: [],
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[11],),
  num_scheduled_tokens: {'0': 4},
  total_num_scheduled_tokens: 4,
  scheduled_spec_decode_tokens: {'0': [16, 15, 271]}

_prepare_input() for SD get:
  logits_indices: [0, 1, 2, 3],
  num_sampled_tokens: [4]

_model_forward() on:
  input_ids: [220, 16, 15, 271],
  positions: [11, 12, 13, 14]

call flash_attn with params:
  q.shape: torch.Size([4, 16, 128]),
  k.shape: torch.Size([172, 16, 8, 128]),
  v.shape: torch.Size([172, 16, 8, 128]),
  number_actual_tokens: 4,
  cu_seqlens_q: tensor([0, 4], device='cuda:0', dtype=torch.int32),
  max_seqlen_q: 4,
  seqused_k: (15,),
  max_seqlen_k: 15,
  block_table: tensor([[1, 2]], device='cuda:0', dtype=torch.int32)
flash_attn out:
  torch.Size([4, 16, 128])

_model_forward() get:
  hidden_states: torch.Size([4, 2048])

use logits_indices get:
  sample_hidden_states: torch.Size([4, 2048]),
  logits: torch.Size([4, 151936])

rejection sampling:
  draft_token_ids: [16, 15, 271],
  output_token_ids: [[16, 11, -1, -1]]

prepare_next_token_ids_padded:
  next_token_ids: [11],
  valid_sampled_tokens_count: [2]

propose inputs:
  target_token_ids: [220, 16, 15, 271]
  target_positions: [11, 12, 13, 14]
  target_hidden_states: torch.Size([4, 6144])
  next_token_ids: [11]
  last_token_indices: [1]
--------------------------------------------------
draft forward inputs:
  input_ids: [16, 11, 271, 5109],
  positions: [11, 12, 13, 14],
  hidden_states: torch.Size([4, 2048]),
  inputs_embeds: None
draft forward, get draft_token_ids: [220]
--------------------------------------------------
draft forward inputs:
  input_ids: [220],
  positions: [13],
  hidden_states: torch.Size([1, 2048]),
  inputs_embeds: None
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128]),
  k.shape: torch.Size([172, 16, 8, 128]),
  v.shape: torch.Size([172, 16, 8, 128]),
  number_actual_tokens: 1,
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32),
  max_seqlen_q: 1,
  seqused_k: (16,),
  max_seqlen_k: 15,
  block_table: tensor([[1, 2]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 0, get draft_token_ids: [16]
--------------------------------------------------
draft forward inputs:
  input_ids: [16],
  positions: [14],
  hidden_states: torch.Size([1, 2048]),
  inputs_embeds: None
call flash_attn with params:
  q.shape: torch.Size([1, 16, 128]),
  k.shape: torch.Size([172, 16, 8, 128]),
  v.shape: torch.Size([172, 16, 8, 128]),
  number_actual_tokens: 1,
  cu_seqlens_q: tensor([0, 1], device='cuda:0', dtype=torch.int32),
  max_seqlen_q: 1,
  seqused_k: (17,),
  max_seqlen_k: 15,
  block_table: tensor([[1, 2]], device='cuda:0', dtype=torch.int32)
flash_attn out: torch.Size([1, 16, 128])
draft forward 1, get draft_token_ids: [11]

proposed draft_token_ids: [[220, 16, 11]]

appended new tokens to req-0, tokens: [16, 11]
```
