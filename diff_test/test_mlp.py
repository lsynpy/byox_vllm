import os
import sys

sys.path.append(os.path.abspath("."))

import torch
import torch.distributed as dist

from byoxvllm.qwen3 import Qwen3MLP as ByoxMLP
from nanovllm.models.qwen3 import Qwen3MLP as NanoMLP


def test_mlp():
    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
    x = torch.randn(10, 100, 1024, dtype=torch.bfloat16).cuda()

    nano_mlp = NanoMLP(1024, 3072, "silu").cuda().bfloat16()
    byox_mlp = ByoxMLP(1024, 3072).cuda().bfloat16()

    gate_up_weight = nano_mlp.gate_up_proj.weight.data
    gate_weight, up_weight = torch.chunk(gate_up_weight, 2, dim=0)
    byox_mlp.gate_proj.weight.data.copy_(gate_weight)
    byox_mlp.up_proj.weight.data.copy_(up_weight)
    byox_mlp.down_proj.weight.data.copy_(nano_mlp.down_proj.weight.data)

    nano_o = nano_mlp(x)
    byox_o = byox_mlp(x)

    dist.destroy_process_group()
    torch.testing.assert_close(nano_o, byox_o, rtol=1e-3, atol=1e-3)
    print("pass")


if __name__ == "__main__":
    test_mlp()
