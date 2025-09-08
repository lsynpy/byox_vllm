import os
import sys
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from byoxvllm.config import Config
from byoxvllm.qwen3 import Qwen3ForCausalLM
from nanovllm import LLM, SamplingParams


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


def run_nanovllm(prompt):
    # Set random seeds for deterministic results
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)  # Enable deterministic operations (may impact performance)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for deterministic CUDA operations

    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Use the prompt directly without chat template
    prompts = [prompt]

    outputs = llm.generate(prompts, sampling_params, False)

    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


def run_byoxvllm(prompt):
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config = Config(model_path)
    hf_config = config.hf_config
    model = Qwen3ForCausalLM(hf_config)
    model = load_model_weights(model, model_path)
    model = model.to(torch.bfloat16)  # Convert model to bfloat16
    model = model.to("cpu")
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Input IDs - shape: {input_ids.shape}, values: {input_ids[0, :3]}")

    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand(input_ids.shape[0], -1)
        logits = model(input_ids, position_ids)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

    generated_token = tokenizer.decode(next_token_id[0].item())
    print(f"Generated token: {generated_token}")
    print(f"Generated token ID - shape: {next_token_id.shape}, value: {next_token_id[0].item()}")


if __name__ == "__main__":
    prompt = "list all prime numbers within 100"
    print(f"Prompt: {prompt!r}")

    if len(sys.argv) > 1:
        print("\n=== Running nanovllm ===")
        run_nanovllm(prompt)
    else:
        print("\n=== Running byoxvllm ===")
        run_byoxvllm(prompt)
