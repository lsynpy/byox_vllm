import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from byoxvllm.engine import LLMEngine
from byoxvllm.sampling_params import SamplingParams
from nanovllm import LLM

# Set random seeds for deterministic results
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)  # Enable deterministic operations (may impact performance)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for deterministic CUDA operations
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # supress waring


def run_byoxvllm(prompt, t=0.6, num_tokens=1):
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B")

    # Use the LLMEngine which handles all the complexity
    engine = LLMEngine(model_path)

    # Simple generation with default parameters
    sampling_params = SamplingParams(temperature=t, max_tokens=num_tokens)
    outputs = engine.generate([prompt], sampling_params)

    for output in outputs:
        print(f"Generated token: {output['text']!r} <{output['token_ids']}>")


def run_nanovllm(prompt, t=0.6, num_tokens=1):
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=t, max_tokens=num_tokens)

    prompts = [prompt]

    outputs = llm.generate(prompts, sampling_params, False)

    for output in outputs:
        print(f"Generated token: {output['text']!r} <{output['token_ids']}>")


if __name__ == "__main__":
    prompt = "The capital of France is"
    print(f"Prompt: {prompt!r}")
    N = 1
    t = 0.6

    if len(sys.argv) > 1:
        print("\n=== Running nanovllm ===")
        run_nanovllm(prompt, t=t, num_tokens=N)
    else:
        print("\n=== Running byoxvllm ===")
        run_byoxvllm(prompt, t=t, num_tokens=N)
