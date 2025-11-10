import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from norm_ext import rmsnorm


def llama_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def test_rmsnorm():
    batch_size = 100
    hidden_size = 768
    x = torch.randn(batch_size, hidden_size).to(0).to(torch.float16)
    w = torch.randn(hidden_size).to(0).to(torch.float16)
    eps = 1e-6
    y = torch.empty_like(x)
    rmsnorm(y, x, w, eps)
    y_ref = llama_rms_norm(x, w)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)
    print("Test passed!")


if __name__ == "__main__":
    test_rmsnorm()
