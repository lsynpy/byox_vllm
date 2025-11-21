import gc
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from byoxvllm.engine import LLMEngine
from byoxvllm.sampling_params import SamplingParams
from nanovllm import LLM

# Determinism
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def attach_hooks(model, module_paths: list[str]):
    hidden_states = []
    hook_handles = []

    def make_hook(name):
        def hook_fn(mod, _, out):
            output = out[0] if isinstance(out, tuple) else out
            hidden_states.append(
                {
                    "name": name,
                    "class": type(mod).__name__,
                    "output": output.detach().cpu(),
                }
            )

        return hook_fn

    root = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model

    for path in module_paths:
        try:
            mod = root
            for part in path.split("."):
                mod = getattr(mod, part)
            h = mod.register_forward_hook(make_hook(path))
            hook_handles.append(h)
        except AttributeError as e:
            print(f"[Warning] Failed to hook '{path}': {e}")

    return hidden_states, hook_handles


def get_module_paths(model):
    root = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
    paths = ["embed_tokens", "norm"]
    num_layers = len(root.layers)
    for i in range(num_layers):
        for sub in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]:
            paths.append(f"layers.{i}.{sub}")
    return paths


def run_byoxvllm(prompt, t=0.6, num_tokens=1, dump_path=None):
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    engine = LLMEngine(model_path)
    model = engine.model_runner.model
    paths = get_module_paths(model)
    hidden_states, handles = attach_hooks(model, paths)

    sampling_params = SamplingParams(temperature=t, max_tokens=num_tokens)
    outputs = engine.generate([prompt], sampling_params)

    for output in outputs:
        print(f"[byoxvllm] Generated token: {output['text']!r}\n <{output['token_ids']}>")

    if dump_path:
        torch.save(hidden_states, dump_path)
        # print(f"[byoxvllm] dumped {len(hidden_states)} layers to {dump_path}")

    for h in handles:
        h.remove()
    del engine, model, hidden_states, handles, sampling_params, outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_nanovllm(prompt, t=0.6, num_tokens=1, dump_path=None):
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    model = llm.model_runner.model
    paths = get_module_paths(model)
    hidden_states, handles = attach_hooks(model, paths)

    sampling_params = SamplingParams(temperature=t, max_tokens=num_tokens)
    outputs = llm.generate([prompt], sampling_params, False)

    for output in outputs:
        print(f"[nanovllm] Generated token: {output['text']!r}\n <{output['token_ids']}>")

    if dump_path:
        torch.save(hidden_states, dump_path)
        # print(f"[nanovllm] dumped {len(hidden_states)} layers to {dump_path}")

    for h in handles:
        h.remove()
    del llm, model, hidden_states, handles, sampling_params, outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def compare_hidden_states(path1, path2, rtol=1e-3, atol=1e-3):
    print("\n === Comparing === \n")
    h1 = torch.load(path1, map_location="cpu")
    h2 = torch.load(path2, map_location="cpu")
    assert len(h1) == len(h2), f"Mismatched number of hooks: {len(h1)} vs {len(h2)}"

    for a, b in zip(h1, h2):
        name_a, _ = a["name"], b["name"]
        class_a, class_b = a["class"], b["class"]
        print(" " * name_a.count("."), end="")
        print(f"{name_a} ({class_a} <-> {class_b}) |", end=" ")
        try:
            torch.testing.assert_close(a["output"], b["output"], rtol=rtol, atol=atol)
            print("PASS ✅")
        except AssertionError:
            print("FAILED ❗️")


if __name__ == "__main__":
    prompt = "The capital of France is"
    t = 0.6
    num_tokens = 30
    dump_dir = "./hidden_dumps"
    os.makedirs(dump_dir, exist_ok=True)

    path_byox = os.path.join(dump_dir, "byoxvllm.pt")
    path_nano = os.path.join(dump_dir, "nanovllm.pt")

    # run_byoxvllm(prompt, t=t, num_tokens=num_tokens, dump_path=path_byox)
    run_nanovllm(prompt, t=t, num_tokens=num_tokens, dump_path=path_nano)
    # compare_hidden_states(path_byox, path_nano)
