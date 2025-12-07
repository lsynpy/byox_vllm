## 1. Quick Start

```sh
uv pip install -e .
pytest
python example.py
python bench.py
```

## 2. vLLM ngram data flow

```py
input = "List 10 numbers only contains digit 1:",
tokens = ['852<List>', '220< >', '16<1>', '15<0>', '5109< numbers>', '1172< only>', '5610<contains>', '15723< digit>', '220< >', '16<1>', '25<:>']

--- 1. prefill on first 11 tokens,      0 draft input, 0 accept, 1 append, 0 draft output---
schedule result:
  scheduled_new_reqs: [NewRequestData(req_id=0,prompt_token_ids=[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25],block_ids=([1],),num_computed_tokens=0,)],
  scheduled_cached_reqs: CachedRequestData(req_ids=[],num_computed_tokens=[],)
  num_scheduled_tokens: {'0': 11},
  total_num_scheduled_tokens: 11,
  scheduled_spec_decode_tokens: {}

_prepare_input() for normal get:
  logits_indices: [10],
  num_sampled_tokens: [1]

_model_forward() on:
  input_ids: [852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25],
  positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

_model_forward() get:
  hidden_states: torch.Size([11, 1024]),

use logits_indices get:
  sample_hidden_states: torch.Size([1, 1024]),
  logits: torch.Size([1, 151936])

token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220< >,]]
proposed draft_token_ids: [[]]

appended new tokens to req-0, tokens: [220]

--- 2. decoding for 12th token,         0 draft input, 0 accept, 1 append, 2 draft output---
schedule result:
  scheduled_new_reqs: [],
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[11],),
  num_scheduled_tokens: {'0': 1},
  total_num_scheduled_tokens: 1,
  scheduled_spec_decode_tokens: {}

_prepare_input() for normal get:
  logits_indices: [0],
  num_sampled_tokens: [1]

_model_forward() on:
  input_ids: [220, 220, 16],  # pad to 3 for cuda graph batch size, 220 is the new generated token, 220, 16 is just stale data in self.input_ids.gpu
  positions: [11, 1, 2] # same as input_ids, self.positions.gpu = [11, 1, 2, ...], 11 is the new token position, 1, 2 is stale

_model_forward() get:
  hidden_states: torch.Size([3, 1024]),

use logits_indices get:
  sample_hidden_states: torch.Size([1, 1024]),
  logits: torch.Size([1, 151936])

token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, ]]
proposed draft_token_ids: [[15, 5109]]

appended new tokens to req-0, tokens: [16]

--- 3. decoding for 13th token,        2 draft input, 0 accept, 1 append, 0 draft output---
schedule result:
  scheduled_new_reqs: [],
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[12],),
  num_scheduled_tokens: {'0': 3},
  total_num_scheduled_tokens: 3,
  scheduled_spec_decode_tokens: {'0': [15, 5109]}

_prepare_input() for SD get:
  logits_indices: [0, 1, 2],
  num_sampled_tokens: [3]

_model_forward() on:
  input_ids: [16, 15, 5109],
  positions: [12, 13, 14]

_model_forward() get:
  hidden_states: torch.Size([3, 1024])

use logits_indices get:
  sample_hidden_states: torch.Size([3, 1024]),
  logits: torch.Size([3, 151936])

rejection sampling:
  draft_token_ids: [15, 5109],
  output_token_ids: [[11, -1, -1]]

token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 5109]]
proposed draft_token_ids: [[]]

appended new tokens to req-0, tokens: [11]

--- 4. decoding for 14th token,        0 draft input, 0 accept, 1 append, 0 draft output---
appended new tokens to req-0, tokens: [220]

--- 5. decoding for 15th token,        0 draft input, 0 accept, 1 append, 2 draft output---
token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16]]
proposed draft_token_ids: [[15, 5109]]

appended new tokens to req-0, tokens: [16]

--- 6. decoding for 16th token,        2 draft input, 0 accept, 1 append, 0 draft output---
rejection sampling:
  draft_token_ids: [15, 5109],
  output_token_ids: [[16, -1, -1]]

token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16, 16, 5109]]
proposed draft_token_ids: [[]]

appended new tokens to req-0, tokens: [16]

--- 7. decoding for 17th token,        0 draft input, 0 accept, 1 append, 2 draft output---
token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16, 16]]

proposed draft_token_ids: [[220, 16]]
appended new tokens to req-0, tokens: [11]

--- 8. decoding for 18,19,20th token,  2 draft input, 2 accept, 3 append, 2 draft output---
schedule result:
  scheduled_new_reqs: [],
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[17],),
  num_scheduled_tokens: {'0': 3},
  total_num_scheduled_tokens: 3,
  scheduled_spec_decode_tokens: {'0': [220, 16]}

_prepare_input() for SD get:
  logits_indices: [0, 1, 2],
  num_sampled_tokens: [3]

_model_forward() on:
  input_ids: [11, 220, 16],
  positions: [17, 18, 19]

_model_forward() get:
  hidden_states: torch.Size([3, 1024])

use logits_indices get:
  sample_hidden_states: torch.Size([3, 1024]),
  logits: torch.Size([3, 151936])
rejection sampling:
  draft_token_ids: [220, 16],
  output_token_ids: [[220, 16, 16]]
token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16, 16, 11, 220, 16, 16]]

proposed draft_token_ids: [[11, 220]]
appended new tokens to req-0, tokens: [220, 16, 16]

--- 9. decoding for 21th token,        2 draft input, 0 accept, 1 append, 2 draft output---
rejection sampling:
  draft_token_ids: [11, 220],
  output_token_ids: [[16, -1, -1]]

token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16, 16, 11, 220, 16, 16, 16, 220]]

proposed draft_token_ids: [[11, 220]]
appended new tokens to req-0, tokens: [16]

--- 10. decoding for 22,23,24th token, 2 draft input, 2 accept, 3 append, 2 draft output---

schedule result:
  scheduled_new_reqs: [],
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[21],),
  num_scheduled_tokens: {'0': 3},
  total_num_scheduled_tokens: 3,
  scheduled_spec_decode_tokens: {'0': [11, 220]}

_prepare_input() for SD get:
  logits_indices: [0, 1, 2],
  num_sampled_tokens: [3]

_model_forward() on:
  input_ids: [16, 11, 220],
  positions: [21, 22, 23]

_model_forward() get:
  hidden_states: torch.Size([3, 1024])

use logits_indices get:
  sample_hidden_states: torch.Size([3, 1024]),
  logits: torch.Size([3, 151936])

rejection sampling:
  draft_token_ids: [11, 220],
  output_token_ids: [[11, 220, 16]]

token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16, 16, 11, 220, 16, 16, 16, 11, 220, 16]]
proposed draft_token_ids: [[16, 16]]

appended new tokens to req-0, tokens: [11, 220, 16]

--- 11. decoding for 25,26,27th token, 2 draft input, 2 accept, 3 append, 2 draft output---
schedule result:
  scheduled_new_reqs: [],
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[24],),
  num_scheduled_tokens: {'0': 3},
  total_num_scheduled_tokens: 3,
  scheduled_spec_decode_tokens: {'0': [16, 16]}

_prepare_input() for SD get:
  logits_indices: [0, 1, 2],
  num_sampled_tokens: [3]

_model_forward() on:
  input_ids: [16, 16, 16],
  positions: [24, 25, 26],

_model_forward() get:
  hidden_states: torch.Size([3, 1024])

use logits_indices get:
  sample_hidden_states: torch.Size([3, 1024]),
  logits: torch.Size([3, 151936])

rejection sampling:
  draft_token_ids: [16, 16],
  output_token_ids: [[16, 16, 16]]

token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16, 16, 11, 220, 16, 16, 16, 11, 220, 16, 16, 16, 16]]
proposed draft_token_ids: [[11, 220]]

appended new tokens to req-0, tokens: [16, 16, 16]

--- 12. decoding for 28,29,30th token, 2 draft input, 2 accept, 3 append, 2 draft output---
rejection sampling:
  draft_token_ids: [11, 220],
  output_token_ids: [[11, 220, 16]]

token_ids_cpu: [[852, 220, 16, 15, 5109, 1172, 5610, 15723, 220, 16, 25, 220, 16, 11, 220, 16, 16, 11, 220, 16, 16, 16, 11, 220, 16, 16, 16, 16, 11, 220, 16]]
proposed draft_token_ids: [[16]]

appended new tokens to req-0, tokens: [11, 220, 16]

--- 13. decoding for 31th token,       0 draft input, 0 accept, 1 append, 0 draft output---
schedule result:
  scheduled_new_reqs: [],
  scheduled_cached_reqs: CachedRequestData(req_ids=['0'],num_computed_tokens=[30],),
  num_scheduled_tokens: {'0': 1},
  total_num_scheduled_tokens: 1,
  scheduled_spec_decode_tokens: {}

_prepare_input() for normal get:
  logits_indices: [0],
  num_sampled_tokens: [1]

_model_forward() on:
  input_ids: [16, 11, 220],
  positions: [30, 28, 29]

_model_forward() get:
  hidden_states: torch.Size([3, 1024])

logits_indices get:
  sample_hidden_states: torch.Size([1, 1024]),
  logits: torch.Size([1, 151936])

appended new tokens to req-0, tokens: [16]
```

### 2.1. vLLM ngram summary

seq origin len: 11
max tokens: 32
total ouput tokens: 32 - 11 = 21
total steps: 13
number of steps with draft tokens: 7
number of steps without draft tokens: 6
number of draft tokens: 7 `*` 2 = 14
number of accepted tokens: 8
mean acceptance length: `8 / 7` + 1 = 2.14
acceptance at token 0: `4 / 7` = 0.57
acceptance at token 1: `4 / 7` = 0.57
