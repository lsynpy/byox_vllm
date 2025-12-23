import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class SpeculativeConfig:
    method: str | None = None
    num_speculative_tokens: int = None
    # Ngram
    prompt_lookup_max: int | None = None
    prompt_lookup_min: int | None = None
    # Eagle
    draft_path: str = None
    draft_hf_config: AutoConfig | None = None

    def __post_init__(self):
        if self.draft_path:
            assert os.path.isdir(self.draft_path)
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_path)


@dataclass
class Config:
    model_path: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_cudagraph_batch_size: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    speculative_config: SpeculativeConfig = None

    def __post_init__(self):
        assert os.path.isdir(self.model_path)
        assert (
            self.kvcache_block_size % 256 == 0
        )  # flash-attn requires block size to be divisible by 256
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

        if self.speculative_config is not None and isinstance(self.speculative_config, dict):
            self.speculative_config = SpeculativeConfig(**self.speculative_config)
