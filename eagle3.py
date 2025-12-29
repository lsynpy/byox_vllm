import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TRITON_INTERPRET"] = "1"

from nanovllm.llm import LLM
from nanovllm.sample.sampling_params import SamplingParams
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__)

NUM_SPEC_TOKENS = 5
OUTPUT_LEN = 32


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
    draft_path = os.path.expanduser("~/huggingface/Qwen3-1.7B_eagle3")

    prompts = [
        "List the first ten prime numbers:",
        "The capital of France is",
        "Once upon a time in a land far, far away,",
        "List 10 numbers only contains digit 1:",
    ]
    speculative_config = {
        "method": "eagle3",
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "draft_path": draft_path,
    }
    sampling_params = SamplingParams(temperature=0, max_tokens=OUTPUT_LEN)

    llm = LLM(
        path,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
        speculative_config=speculative_config,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        logger.info("Prompt: %r", prompt)
        logger.info("Completion: %s", output["text"])


if __name__ == "__main__":
    main()
