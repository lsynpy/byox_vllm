import logging
import os
import time
from random import randint, seed

from vllm import LLM as vLLM
from vllm import SamplingParams as vSP

from nanovllm import LLM as nLLM
from nanovllm import SamplingParams as nSP
from nanovllm import get_logger

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bench(name, LLM, SamplingParams):
    seed(0)
    num_seqs = 128
    max_input_len = 128
    max_ouput_len = 128

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=256, gpu_memory_utilization=0.7)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]

    if name == "vllm":
        prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["warmup"], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    logger.info("[%s] %d tok, %.2fs, %.2f tok/s", name, total_tokens, t, total_tokens / t)


def main():
    from nanovllm.utils.logging import set_default_log_level

    set_default_log_level(logging.INFO)
    bench("vllm", vLLM, vSP)
    bench("nanovllm", nLLM, nSP)


if __name__ == "__main__":
    main()
