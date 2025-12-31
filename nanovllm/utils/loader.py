import logging
import os
from collections.abc import Iterable
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

from nanovllm.models.qwen3_eagle3 import Eagle3Qwen3ForCausalLM
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__, logging.DEBUG)


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _load_weights(model: nn.Module, weight_dict: dict, packed_modules_mapping: dict):
    for weight_name, loaded_weight in weight_dict.items():
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


def load_model(model: nn.Module, path: str):
    if isinstance(model, Eagle3Qwen3ForCausalLM):
        all_weights = []
        pytorch_model_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            for key, value in state_dict.items():
                all_weights.append((key, value))

        load_eagle3_model_weights(model, all_weights)
    else:
        packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as f:
                weight_dict = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118
                _load_weights(model, weight_dict, packed_modules_mapping)


def load_eagle3_model_weights(model: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    params_dict = dict(model.named_parameters())
    loaded_params: set[str] = set()

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    for name, loaded_weight in weights:
        if "t2d" in name:
            continue

        is_packed_module = False
        param_name = None
        shard_id = None

        for k, (v, sid) in packed_modules_mapping.items():
            if k in name:
                param_name = name.replace(k, v)
                shard_id = sid
                is_packed_module = True
                break

        if not is_packed_module:
            if "lm_head" in name:
                if name in params_dict:
                    param_name = name
            else:
                if name in params_dict:
                    param_name = name
                elif ("model." + name) in params_dict:
                    param_name = "model." + name

        if is_packed_module:
            if not param_name.startswith("model."):
                param_name = "model." + param_name
            if not param_name.endswith(".weight"):
                param_name = param_name + ".weight"

        if param_name and param_name in params_dict:
            param = params_dict[param_name]

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if is_packed_module:
                weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader(param, loaded_weight)

            loaded_params.add(param_name)

    return loaded_params
