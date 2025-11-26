import logging
import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams, set_global_log_level
from nanovllm.utils.logging import logger

NUM_SPEC_TOKENS = 2
PROMPT_LOOKUP_MAX = 5
PROMPT_LOOKUP_MIN = 2


def main():
    set_global_log_level(logging.INFO)
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)

    speculative_config = {
        "method": "ngram",
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "prompt_lookup_max": PROMPT_LOOKUP_MAX,
        "prompt_lookup_min": PROMPT_LOOKUP_MIN,
    }
    sampling_params = SamplingParams(temperature=0.6, max_tokens=512)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params, speculative_config=speculative_config)

    for prompt, output in zip(prompts, outputs):
        logger.info(f"Prompt: {prompt!r}")
        logger.info(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
