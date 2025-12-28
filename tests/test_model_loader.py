import os

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.config import Config, SpeculativeConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_eagle3 import Eagle3Qwen3ForCausalLM
from nanovllm.utils.loader import default_weight_loader, load_model


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_distributed():
    if not dist.is_available() or not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="tcp://127.0.0.1:29501", world_size=1, rank=0
        )
    yield
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def test_load_pytorch():
    draft_path = os.path.expanduser("~/huggingface/Qwen3-1.7B_eagle3")

    speculative_config = SpeculativeConfig(
        method="eagle3",
        num_speculative_tokens=5,
        draft_path=draft_path,
        draft_hf_config=AutoConfig.from_pretrained(draft_path),
    )
    config = Config(model_path=draft_path, speculative_config=speculative_config)

    model = Eagle3Qwen3ForCausalLM(config)
    load_model(model, draft_path)


def test_load_safetensors():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B")

    config = AutoConfig.from_pretrained(path)
    model = Qwen3ForCausalLM(config)
    load_model(model, path)


def test_default_weight_loader():
    param = torch.nn.Parameter(torch.zeros(5, 3))
    loaded_weight = torch.ones(5, 3)

    default_weight_loader(param, loaded_weight)

    assert torch.allclose(param.data, loaded_weight), "Default weight loader didn't copy data correctly"
