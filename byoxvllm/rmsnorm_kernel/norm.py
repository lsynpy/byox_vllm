from typing import Optional

import torch
import torch.nn as nn

from .norm_ext import fused_add_rmsnorm, rmsnorm

ENABLE_RMSNORM_KERNEL = True


def rms_norm_torch(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure PyTorch RMSNorm implementation as fallback."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = weight.to(torch.float32) * hidden_states
    return hidden_states.to(input_dtype)


def fused_add_rms_norm_torch(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure PyTorch fused add and RMSNorm implementation as fallback."""
    # Add residual connection
    hidden_states += residual
    residual = hidden_states

    # Apply RMSNorm
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = weight.to(torch.float32) * hidden_states
    return hidden_states.to(input_dtype), residual.to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if ENABLE_RMSNORM_KERNEL:
            print("using rmsnorm cuda")
            out = torch.empty_like(hidden_states)
            rmsnorm(out, hidden_states, self.weight, self.variance_epsilon)
            return out
        else:
            return rms_norm_torch(hidden_states, self.weight, self.variance_epsilon)


class FusedAddRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if residual is None:
            residual = torch.zeros_like(hidden_states)

        if ENABLE_RMSNORM_KERNEL:
            # For fused operation, both tensors are modified in-place
            print("using fused rmsnorm cuda")
            hidden_states = fused_add_rmsnorm(hidden_states, residual, self.weight, self.variance_epsilon)
            # Return the modified hidden_states and residual
            return hidden_states, hidden_states  # Return same tensor since it's modified in-place
        else:
            return fused_add_rms_norm_torch(hidden_states, residual, self.weight, self.variance_epsilon)
