import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Build .so before run test. `python setup.py build_ext --inplace`
from norm_ext import fused_add_rmsnorm, rmsnorm


def rmsnorm_torch(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def fused_add_rmsnorm_torch(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(orig_dtype)
    return x, residual


def test_rmsnorm():
    batch_size = 100
    hidden_size = 768
    x = torch.randn(batch_size, hidden_size).to(0).to(torch.float16)
    w = torch.randn(hidden_size).to(0).to(torch.float16)
    eps = 1e-6
    y = torch.empty_like(x)
    rmsnorm(y, x, w, eps)
    y_torch = rmsnorm_torch(x, w)

    torch.testing.assert_close(y_torch, y, rtol=1e-3, atol=1e-3)
    print("Test rmsnorm passed!")


def test_fused_rmsnorm():
    batch_size = 100
    hidden_size = 768
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size).to(0).to(torch.float16)
    w = torch.randn(hidden_size).to(0).to(torch.float16)
    residual = torch.randn_like(x)
    x_torch, residual_torch = fused_add_rmsnorm_torch(x.clone(), residual.clone(), w, eps)
    x_fused = x.clone()
    residual_fused = residual.clone()
    fused_add_rmsnorm(x_fused, residual_fused, w, eps)

    torch.testing.assert_close(x_fused, x_torch, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_torch, rtol=1e-3, atol=1e-3)
    print("Test fused_add_rmsnorm passed!")


if __name__ == "__main__":
    test_rmsnorm()
    test_fused_rmsnorm()
