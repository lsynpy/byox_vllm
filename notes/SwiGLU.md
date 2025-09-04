# SwiGLU Activation Function

## Overview

SwiGLU (Sigmoid-weighted Gated Linear Unit) is an activation function used in modern transformer-based models, including Qwen3. It combines the Gated Linear Unit (GLU) mechanism with the Swish/SiLU activation function.

## Mathematical Definition

$$\text{SwiGLU}(x, W, V, W_2) = (\sigma(\text{SiLU}(xW)) \otimes (xV))W_2$$

Where:

- $\sigma$ is the sigmoid function
- $\text{SiLU}$ (Sigmoid Linear Unit) is defined as: $\text{SiLU}(x) = x \cdot \sigma(x)$
- $\otimes$ is element-wise multiplication
- $W$, $V$, $W_2$ are weight matrices

## Implementation in This Repository

### nanovllm Implementation

In `nanovllm/layers/activation.py`:

```python
class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```

This implementation assumes the input tensor has been pre-split into two parts, applying SiLU to the first half and multiplying it element-wise with the second half.

### byoxvllm Implementation

In `byoxvllm/qwen3.py` (Qwen3MLP class):

```python
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

This implementation explicitly uses separate linear projections for the gate and value components, then applies the SiLU activation to the gate projection before multiplying with the value projection.

## Advantages

1. **Gating Mechanism**: The gating mechanism allows the network to control the flow of information, which can improve training stability and model performance.
2. **Non-linearity**: Combines the benefits of both the gating mechanism and the non-linear activation function.
3. **Performance**: Has been shown to improve performance in large language models compared to traditional activation functions like ReLU.

## Usage in Qwen3

Qwen3 uses SwiGLU as the activation function in its MLP (Multi-Layer Perceptron) layers, following the architecture described in the Qwen3 technical report.
