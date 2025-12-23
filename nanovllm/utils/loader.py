import os
from collections.abc import Iterable
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

from nanovllm.models.llama_eagle3 import Eagle3Qwen3ForCausalLM


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

    for name, loaded_weight in weights:
        if "t2d" in name:
            continue
        if "d2t" in name:
            name = name.replace("d2t", "draft_id_to_target_id")
        elif "lm_head" not in name:
            name = "model." + name

        if name in params_dict:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

    return loaded_params
