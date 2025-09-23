from dataclasses import dataclass, field

from transformers import AutoConfig


@dataclass
class Config:
    model: str
    kvcache_block_size: int = 256
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096
    gpu_memory_utilization: float = 0.9
    num_kvcache_blocks: int = 1024
    hf_config: AutoConfig = field(init=False)
    eos_token_id: int = field(init=False)

    def __post_init__(self):
        self.hf_config = AutoConfig.from_pretrained(self.model)
