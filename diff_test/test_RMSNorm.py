import os
import sys

sys.path.append(os.path.abspath("."))


import torch

from byoxvllm.qwen3 import Qwen3Config
from byoxvllm.rmsnorm_kernel.norm import RMSNorm as ByoxRMSNorm
from nanovllm.layers.layernorm import RMSNorm as NanoRMSNorm


def test_RMSNorm():
    torch.manual_seed(0)
    config = Qwen3Config()

    byox_RMSNorm = ByoxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    nano_RMSNorm = NanoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    weight = torch.randn(config.hidden_size)
    byox_RMSNorm.weight.data = weight.clone()
    nano_RMSNorm.weight.data = weight.clone()

    fake_input = torch.randn(10, 100, config.hidden_size, dtype=torch.bfloat16)

    byox_output = byox_RMSNorm(fake_input)
    nano_output = nano_RMSNorm(fake_input)

    torch.testing.assert_close(byox_output, nano_output, rtol=1e-3, atol=1e-3)
    print("pass")


if __name__ == "__main__":
    test_RMSNorm()
