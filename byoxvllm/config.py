from dataclasses import dataclass, field

from transformers import AutoConfig


@dataclass
class Config:
    model_path: str
    hf_config: AutoConfig = field(init=False)

    def __post_init__(self):
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
