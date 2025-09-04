import os
import sys
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from byoxvllm.config import Config
from byoxvllm.qwen3 import Qwen3ForCausalLM


def load_model_weights(model, model_path):
    weight_files = glob(os.path.join(model_path, "*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    state_dict = {}
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    config = Config(model_path)
    hf_config = config.hf_config

    print("Creating model...")
    model = Qwen3ForCausalLM(hf_config)

    print("Loading weights...")
    model = load_model_weights(model, model_path)

    model = model.to("cpu")
    model.eval()

    prompt = "Hello, my name is"
    print(f"Prompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Input IDs: {input_ids}")
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand(input_ids.shape[0], -1)
        logits = model(input_ids, position_ids)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

    generated_token = tokenizer.decode(next_token_id[0].item())
    print(f"Generated token: {generated_token}")
    print(f"Generated token ID: {next_token_id[0].item()}")

    full_output = tokenizer.decode(torch.cat([input_ids[0], next_token_id[0].unsqueeze(0)], dim=-1))
    print(f"Full output: {full_output}")


if __name__ == "__main__":
    main()
