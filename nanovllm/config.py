import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class SpeculativeConfig:
    method: str | None = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    num_speculative_tokens: int = None
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    # Ngram proposer configuration
    prompt_lookup_max: int | None = None
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = None
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""


@dataclass
class Config:
    model: str
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
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0  # flash-attn requires block size to be divisible by 256
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

        if self.speculative_config is not None and isinstance(self.speculative_config, dict):
            self.speculative_config = SpeculativeConfig(**self.speculative_config)
