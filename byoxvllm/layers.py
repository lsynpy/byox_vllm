import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    @torch.compile
    def rmsnorm(self, x):
        """
        r = x
        x = input_layernorm(x)
        x = self_attn(x)
        x = x + r
        r = x
        x = post_layernorm(x)
        x = mlp(x)
        x = x + r
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(orig_dtype)

    @torch.compile
    def fused_add_rmsnorm(self, x, residual):
        """
        x, r = input_layernorm(x, r)
        x = self_attn(x)
        x, r = post_layernorm(x, r)
        x = mlp(x)
        return x, r
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(orig_dtype), residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self.rmsnorm(x)
        else:
            return self.fused_add_rmsnorm(x, residual)


class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 1, 2)
        sin = sin.repeat(1, 1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query, key


def rotate_half(x):
    x1 = x.float()[..., : x.shape[-1] // 2]
    x2 = x.float()[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile
def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(q.dtype)
