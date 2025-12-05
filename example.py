import logging
import os

from nanovllm import LLM, SamplingParams, set_global_log_level
from nanovllm.utils.logging import logger


def main():
    set_global_log_level(logging.DEBUG)
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    prompts = [
        "List the first ten prime numbers:",
        "The capital of France is",
        "Once upon a time in a land far, far away,",
    ]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        logger.info(f"Prompt: {prompt!r}")
        logger.info(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
