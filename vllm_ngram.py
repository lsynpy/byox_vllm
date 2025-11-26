from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

# Hardcoded parameters
MODEL_DIR = "Qwen/Qwen3-0.6B"
NUM_SPEC_TOKENS = 2
PROMPT_LOOKUP_MAX = 5
PROMPT_LOOKUP_MIN = 2
TEMPERATURE = 0
OUTPUT_LEN = 256

# Sample prompts
PROMPTS = [
    # "Hello, my name is",
    # "The capital of France is",
    "List the first ten prime numbers:",
    # "The meaning of life is",
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Encode prompts
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in PROMPTS]

    # Configure ngram speculative decoding
    speculative_config = {
        "method": "ngram",
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "prompt_lookup_max": PROMPT_LOOKUP_MAX,
        "prompt_lookup_min": PROMPT_LOOKUP_MIN,
    }

    # Initialize LLM
    llm = LLM(
        model=MODEL_DIR,
        trust_remote_code=True,
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        enforce_eager=False,
        gpu_memory_utilization=0.8,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 5},
        disable_chunked_mm_input=True,
    )

    # Generate
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=OUTPUT_LEN)
    outputs = llm.generate(
        [{"prompt_token_ids": x} for x in prompt_ids],
        sampling_params=sampling_params,
    )

    for i, output in enumerate(outputs):
        print("-" * 50)
        print(f"prompt: {PROMPTS[i]}")
        print(f"generated text: {output.outputs[0].text}")
        print("-" * 50)

    try:
        metrics = llm.get_metrics()
    except AssertionError:
        print("Metrics are not supported in the V0 engine.")
        return

    total_num_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * NUM_SPEC_TOKENS

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # Print acceptance at each token position
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
