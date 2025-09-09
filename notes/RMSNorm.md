# RMSNorm (Root Mean Square Normalization)

## Overview

RMSNorm is a simplified normalization technique that normalizes activations using only the root mean square (RMS) statistic, without centering (no mean subtraction).

## Formula

```cpp
RMSNorm(x) = x / RMS(x) * γ
where RMS(x) = sqrt(mean(x²) + ε)
```

## Key Features

- **No mean calculation**: Unlike LayerNorm, RMSNorm doesn't subtract the mean
- **Faster computation**: ~15-20% speedup over LayerNorm
- **Similar performance**: Achieves comparable results to LayerNorm in practice
- **Learnable scaling**: Uses learnable parameter γ (weight) for each dimension

## Implementation Details

- `Fused residual connection` for efficiency
- `Cast to float32` for precision
