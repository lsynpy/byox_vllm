# Rotary Positional Embedding (RoPE)

## Overview

Rotary Positional Embedding (RoPE) is a method for encoding positional information in transformer models. Unlike traditional positional embeddings that are added to the input, RoPE injects positional information through rotation operations in the attention mechanism.

## Mathematical Definition

RoPE applies a rotation matrix to the query and key vectors based on their positions. For a vector $x$ at position $m$, RoPE applies the following transformation:

$$\text{RoPE}(x_m) = R_{\theta,m} \cdot x_m$$

Where $R_{\theta,m}$ is a rotation matrix that depends on the position $m$ and trainable parameters $\theta$.

For a 2D vector $[x, y]$, the rotation is defined as:

$$R_{\theta,m} \cdot \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \cdot \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x\cos(m\theta) - y\sin(m\theta) \\ x\sin(m\theta) + y\cos(m\theta) \end{bmatrix}$$

## Implementation in This Repository

### RoPE Implementation

In `byoxvllm/layers.py`:

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

### Applying RoPE

The rotation is applied using the `apply_rotary_pos_emb` function:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

## Advantages

1. **Relative Positional Information**: RoPE naturally captures relative positional relationships between tokens.
2. **Extrapolation**: RoPE can generalize to sequence lengths not seen during training.
3. **Parameter Efficiency**: No additional parameters are needed for positional embeddings.
4. **Compatibility**: Works well with existing transformer architectures.

## Usage in Qwen3

Qwen3 uses RoPE for encoding positional information in its attention mechanism, which helps the model understand the order of tokens in the input sequence.
