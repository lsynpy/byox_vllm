import os
import sys

sys.path.append(os.path.abspath("."))


import torch

from byoxvllm.layers import RMSNorm as ByoxRMSNorm
from byoxvllm.qwen3 import Qwen3Config
from nanovllm.layers.layernorm import RMSNorm as NanoRMSNorm


def test_RMSNorm():
    torch.manual_seed(0)
    # 1. Create configurations
    config = Qwen3Config()

    # 2. Create RMSNorm instances
    byox_RMSNorm = ByoxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    nano_RMSNorm = NanoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 3. Create fake weights and load them
    weight = torch.randn(config.hidden_size)
    byox_RMSNorm.weight.data = weight.clone()
    nano_RMSNorm.weight.data = weight.clone()

    # 4. Create a fake input tensor
    fake_input = torch.randn(1, 10, config.hidden_size)

    # 5. Run the input through both RMSNorm layers
    byox_output = byox_RMSNorm(fake_input)
    nano_output = nano_RMSNorm(fake_input)

    # 6. Compare the outputs
    print(f"{torch.sum(byox_output != nano_output)} different values in total {byox_output.numel()}")
    print(f"Max difference: {torch.max(torch.abs(byox_output - nano_output))}")
    if torch.equal(byox_output, nano_output):
        print("RMSNorm outputs are exactly the same.")
    else:
        print("Outputs are not exactly the same")


if __name__ == "__main__":
    test_RMSNorm()
