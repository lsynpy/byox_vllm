import logging
import os

from nanovllm import LLM, SamplingParams
from nanovllm.utils.logging import get_logger, set_default_log_level

logger = get_logger(__name__)

NUM_SPEC_TOKENS = 2
PROMPT_LOOKUP_MAX = 5
PROMPT_LOOKUP_MIN = 2


def main():
    set_default_log_level(logging.DEBUG)
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    prompts = [
        # "List the first ten prime numbers:",
        "List the 10 numbers only contains digit 1:",
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
        gpu_memory_utilization=0.3,
        speculative_config=speculative_config,
        enforce_eager=True,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        logger.info("Prompt: %r", prompt)
        logger.info("Completion: %s", output["text"])


if __name__ == "__main__":
    main()
