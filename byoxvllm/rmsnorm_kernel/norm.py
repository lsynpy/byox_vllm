import torch
import torch.nn as nn
from einops import rearrange

from .norm_ext import fused_add_rmsnorm as fused_add_rmsnorm_cuda
from .norm_ext import rmsnorm as rmsnorm_cuda

ENABLE_RMSNORM_KERNEL = False


@torch.compile
def rms_norm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x


@torch.compile
def fused_add_rms_norm_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x, residual


def rmsnorm_impl(x, weight, eps=1e-6):
    if ENABLE_RMSNORM_KERNEL:
        x_flat = rearrange(x, "... d -> (...) d")  # kernel only support (b, d) input
        out_flat = torch.empty_like(x_flat)
        rmsnorm_cuda(out_flat, x_flat, weight, eps)
        return out_flat.view_as(x)
    return rms_norm_torch(x, weight, eps)


def fused_add_rmsnorm_impl(x, residual, weight, eps=1e-6):
    if ENABLE_RMSNORM_KERNEL:
        x_flat = rearrange(x, "... d -> (...) d").contiguous()  # x is modified in-place in kernel
        r_flat = rearrange(residual, "... d -> (...) d").contiguous()
        fused_add_rmsnorm_cuda(x_flat, r_flat, weight, eps)
        x.copy_(x_flat.view_as(x))
        residual.copy_(r_flat.view_as(residual))
        return x, residual
    return fused_add_rms_norm_torch(x, residual, weight, eps)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is None:
            return rmsnorm_impl(x, self.weight, self.eps)
        else:
            return fused_add_rmsnorm_impl(x, residual, self.weight, self.eps)
