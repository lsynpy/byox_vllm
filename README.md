## Quick Start

```sh
uv pip install -e .
pytest
python example.py
python bench.py
```

## Todos for Speculative Decoding

- [x] init NgramProposer when init runner, and propose once to trigger numba JIT compilation
- [x] after init, model runner is running in busy loop, waiting task from scheduler
  - [x] call llm.generate(), scheduler will schedule(), and get schedule results for model runner
  - [x] model runner do forward pass, get h_states and logits
  - [x] do sampling,
    - [x] if not ngram, do normal sample
    - [ ] if ngram, first normal sample to get a new token (bonus token)
    - [ ] then append bonus token to input_tokens
    - [ ] do ngram on input_tokens -> draft_token_ids
      - [ ] if non empty draft_token_ids, do rejection sampling, may append new tokens to output
      - [ ] if empty draft_token_ids, append bonus token to output
  - [ ] return output to engine, engine then call scheduler to update this output
    - [ ] scheduler then append new tokens to seq
- [ ] engine in busy loop to process input then run step(), so next iteration is goon
