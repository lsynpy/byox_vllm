import logging
import os

from nanovllm import LLM, SamplingParams, set_global_log_level
from nanovllm.utils.logging import logger

NUM_SPEC_TOKENS = 2
PROMPT_LOOKUP_MAX = 5
PROMPT_LOOKUP_MIN = 2


def main():
    set_global_log_level(logging.DEBUG)
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    prompts = [
        "List the first ten prime numbers:",
    ]
    speculative_config = {
        "method": "ngram",
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "prompt_lookup_max": PROMPT_LOOKUP_MAX,
        "prompt_lookup_min": PROMPT_LOOKUP_MIN,
    }
    sampling_params = SamplingParams(temperature=0, max_tokens=32)

    llm = LLM(
        path,
        gpu_memory_utilization=0.4,
        speculative_config=speculative_config,
        enforce_eager=True,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        logger.info(f"Prompt: {prompt!r}")
        logger.info(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
