# QK-Norm in Qwen3

## Overview

QK-Norm is a technique used in Qwen3 where RMSNorm (Root Mean Square Normalization) is applied to the query (Q) and key (K) vectors in the attention mechanism. This normalization helps stabilize training and can improve model performance.

## Implementation

In the Qwen3 implementation, QK-Norm is applied as follows:

```python
# In the attention mechanism
self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

# During forward pass
q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
```

## How It Works

1. After the linear projections for Q and K, but before applying rotary positional embeddings
2. RMSNorm is applied to each head of the Q and K tensors separately
3. The normalization is done across the head dimension

## Benefits

1. **Stabilizes Training**: Normalization helps maintain stable gradients during training
2. **Improves Convergence**: Can lead to faster convergence during training
3. **Better Generalization**: Normalized representations can generalize better to unseen data

## Difference Between byoxvllm and nanovllm

### nanovllm Implementation

- Explicitly implements QK-Norm with separate RMSNorm layers for Q and K
- Applies normalization after linear projections but before RoPE

### byoxvllm Implementation

- Does not currently implement QK-Norm
- This is noted as a TODO item in the notes/todo.md file

## Mathematical Representation

For a query vector $q_i$ and key vector $k_i$ for head $i$:

$$q_{norm_i} = \text{RMSNorm}(q_i)$$
$$k_{norm_i} = \text{RMSNorm}(k_i)$$

Where RMSNorm is defined as:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma$$

With:

- $\epsilon$: A small constant for numerical stability
- $\gamma$: Learnable scale parameter

## Usage in Qwen3
Qwen3 uses QK-Norm as part of its attention mechanism to improve the stability and performance of the model. This is one of the architectural differences between the full Qwen3 implementation and the simplified byoxvllm version.
