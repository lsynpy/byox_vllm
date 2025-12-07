import logging
import os

from nanovllm import LLM, SamplingParams
from nanovllm.utils.logging import get_logger, set_default_log_level

logger = get_logger(__name__)


def main():
    set_default_log_level(logging.DEBUG)
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=4)
    prompts = [
        "List the first ten prime numbers:",
        "The capital of France is",
        "Once upon a time in a land far, far away,",
        "List 10 numbers only contains digit 1:",
    ]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        logger.info("Prompt: %r", prompt)
        logger.info("Completion: %s", output["text"])


if __name__ == "__main__":
    main()
