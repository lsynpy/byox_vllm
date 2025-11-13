import torch
import torch.nn.functional as F
from torch import nn


class MergedLinear(nn.Module):
    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes

        # Create a single weight matrix for all outputs
        total_output_size = sum(output_sizes)
        self.weight = nn.Parameter(torch.empty(total_output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(total_output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.bias = None

        # Initialize weights and biases
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int = None):
        # For MergedLinear, we need to handle loading weights for different shards
        if loaded_shard_id is not None:
            # Handle shard-specific loading
            param_data = param.data
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
            param_data = param_data.narrow(0, shard_offset, shard_size)
            param_data.copy_(loaded_weight)
        else:
            # Load the entire weight matrix
            param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform linear transformation
        output = F.linear(x, self.weight, self.bias)
        return output


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
