import torch
import torch.nn as nn

from .norm_ext import fused_add_rmsnorm, rmsnorm

ENABLE_RMSNORM_KERNEL = True


@torch.compile
def rms_norm_torch(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = weight.to(torch.float32) * hidden_states
    return hidden_states.to(input_dtype)


@torch.compile
def fused_add_rms_norm_torch(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    hidden_states += residual
    residual = hidden_states

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = weight.to(torch.float32) * hidden_states
    return hidden_states.to(input_dtype), residual.to(input_dtype)


def rmsnorm_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    if ENABLE_RMSNORM_KERNEL:
        out = torch.empty_like(hidden_states)
        rmsnorm(out, hidden_states, weight, eps)
        return out
    else:
        return rms_norm_torch(hidden_states, weight, eps)


def fused_add_rmsnorm_impl(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    if ENABLE_RMSNORM_KERNEL:
        fused_add_rmsnorm(hidden_states, residual, weight, eps)
        return hidden_states, hidden_states
    else:
        return fused_add_rms_norm_torch(hidden_states, residual, weight, eps)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is None:
            return rmsnorm_impl(hidden_states, self.weight, self.eps)
        else:
            return fused_add_rmsnorm_impl(hidden_states, residual, self.weight, self.eps)
