import logging
import os
import warnings

# Suppress numba warnings and logs
try:
    from numba import config

    config.DISABLE_JIT = False  # Keep JIT enabled but suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="numba")
    warnings.filterwarnings("ignore", message=".*TBB threading layer.*")
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.ERROR)
except ImportError:
    pass

from nanovllm import LLM, SamplingParams
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__)

NUM_SPEC_TOKENS = 2
PROMPT_LOOKUP_MAX = 5
PROMPT_LOOKUP_MIN = 2


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    prompts = [
        # "List the first ten prime numbers:",
        # "The capital of France is",
        # "Once upon a time in a land far, far away,",
        "List 10 numbers only contains digit 1:",
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
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        speculative_config=speculative_config,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        logger.info("Prompt: %r", prompt)
        logger.info("Completion: %s", output["text"])


if __name__ == "__main__":
    main()
