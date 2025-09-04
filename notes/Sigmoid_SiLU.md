# 1. Standard Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}$$

- Maps $ \mathbb{R} \to (0,1) $
- Used as an **activation function** in neural networks, particularly for binary classification outputs.
- Outputs can be interpreted as probabilities (e.g., $ P(y=1 \mid x) $, conditional probability that the label y=1 given the input features x).

## 1.1 Symmetry

$$\sigma(x) + \sigma(-x) = 1$$

- Symmetric about point $ (0, 0.5) $ (180° rotational symmetry).

## 1.2 Generalized Sigmoid

$$f_N(x) = \frac{e^x}{e^x + N}$$

- Same shape as $ \sigma(x) $, but **horizontally shifted**.

### 1.2.1 Derive and prove

$$\frac{1}{N} = e^{-\ln N} \Rightarrow \frac{e^x}{N} = e^x \cdot e^{-\ln N} = e^{x - \ln N}$$

$$f_N(x) = \frac{e^x}{e^x + N} = \frac{e^x / N}{e^x / N + 1} = \frac{e^{x - \ln N}}{e^{x - \ln N} + 1}$$

Thus:

$$f_N(x) = \sigma(x - \ln N)$$

- $\sigma(x)$ is symmetric at x =0
- So $f_N(x)$ is symmetric about $ (\ln N, 0.5) $.

### 1.3 Gradient

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

#### 1.3.1 Derive and prove

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
Let $u = 1 + e^{-x}$, so $\sigma(x) = u^{-1}$. Then
$$\sigma'(x) = -u^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$$
Express in Terms of $\sigma(x)$:
$$\sigma(x)(1 - \sigma(x)) = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} = \frac{e^{-x}}{(1 + e^{-x})^2}$$

### 1.4 Reminder of math Rules

$\frac{1}{e^a} = e^{-a}$, $N=e^{\ln N}$

## 1.5 Why Sigmoid Was Popular

- **Smooth and differentiable**: Works with gradient-based optimization (e.g., backpropagation).
- **Output in (0, 1)**: Ideal for modeling probability-like outputs.
- **Historical use**: Widely used in logistic regression and early neural networks.

**Example**: In binary classification, the final layer often uses:
$$\hat{y} = \sigma(w^T x + b)$$
to output a value interpreted as $ P(y = 1 \mid x) $.

## 1.6 ️ Why It’s Rarely Used in Hidden Layers Today

- **Vanishing gradients**: $ \sigma'(x) \to 0 $ for large $ |x| $
- **Not zero-centered**: output is always positive, causing gradient updates to be biased in one direction
- **Slow convergence**: Outperformed by ReLU, Tanh, etc.

## 1.7 Where Sigmoid Is Still Used

| Context                                  | Use Case                                                |
| ---------------------------------------- | ------------------------------------------------------- |
| **Output layer** (binary classification) | Paired with binary cross-entropy loss                   |
| **LSTM gates**                           | Controls information flow (forget, input, output gates) |
| **Interpretable models**                 | When output must resemble a probability                 |

## 1.8 Modern Alternatives for Hidden Layers

| Activation      | Formula                               | Advantage                                               |
| --------------- | ------------------------------------- | ------------------------------------------------------- |
| **ReLU**        | $ \max(0, x) $                        | Fast, sparse, avoids vanishing gradient (for $ x > 0 $) |
| **Tanh**        | $ \frac{e^x - e^{-x}}{e^x + e^{-x}} $ | Zero-centered, better gradient flow                     |
| **Swish(SILU)** | $ x \cdot \sigma(x) $                 | Smooth, self-gated, performs well in deep models        |

# 2. Swish(SILU)

SILU = Sigmoid Linear Unit
$$\text{SILU}(x) = \frac{x}{1 + e^{-x}} = x \cdot \sigma(x)$$
![SILU](SiLU.png)

## 2.1 Key Properties

| Property           | Description                                     |
| ------------------ | ----------------------------------------------- |
| **Domain**         | $ x \in (-\infty, \infty) $                     |
| **Range**          | $ (-\infty, \infty) $, but bounded growth       |
| **Smooth**         | ✅ Yes — fully differentiable, no kinks         |
| **Zero at origin** | $ \text{Swish}(0) = 0 $                         |
| **Self-gating**    | Sigmoid acts as a learnable gate on input $ x $ |

## 2.2️ Behavior by Region

| Input $ x $       | Output Behavior                                |
| ----------------- | ---------------------------------------------- |
| $ x \to +\infty $ | $ \text{Swish}(x) \approx x $ (linear)         |
| $ x \to -\infty $ | $ \text{Swish}(x) \to 0 $ (exponentially fast) |
| $ x = 0 $         | $ \text{Swish}(0) = 0 $                        |
| $ x < 0 $         | Small negative values → smoother than ReLU     |

Unlike ReLU, Swish has a **smooth, gradual transition** around zero.

## 2.3 Derivative

The derivative is smooth and well-behaved:

Let $ s = \sigma(x) $, then:

$$\frac{d}{dx} \text{Swish}(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = s(1 + x(1 - s)j)$$

$$\boxed{\text{Swish}'(x) = \sigma(x) \left(1 + x (1 - \sigma(x))\right)}$$

This smooth gradient helps with optimization in deep networks.

## 2.4 Advantages Over ReLU

| Benefit                           | Explanation                                                        |
| --------------------------------- | ------------------------------------------------------------------ |
| **Smooth everywhere**             | No kink at $ x=0 $ → better gradient flow                          |
| **Non-zero negative activations** | Allows small negative outputs → more expressive                    |
| **Self-gating mechanism**         | Network learns how much of $ x $ to pass                           |
| **Empirically superior**          | Often outperforms ReLU in deep models (e.g., ResNet, EfficientNet) |
| **No hyperparameters**            | Unlike Leaky ReLU, no need to tune negative slope                  |

## 2.5 Comparison with Other Activations

| Function        | Formula               | Smooth? | Zero-Centered? | Vanishing Gradient? |
| --------------- | --------------------- | ------- | -------------- | ------------------- |
| **ReLU**        | $ \max(0, x) $        | ❌      | ❌             | ❌ (for $ x < 0 $)  |
| **Leaky ReLU**  | $ \max(\alpha x, x) $ | ❌      | ❌             | ✅ Less severe      |
| **Sigmoid**     | $ \sigma(x) $         | ✅      | ❌             | ❌ (strong)         |
| **Tanh**        | $ \tanh(x) $          | ✅      | ✅             | ❌ (moderate)       |
| **Swish(SiLU)** | $ x \cdot \sigma(x) $ | ✅      | ❌             | ✅ Mild             |

## 💡 Intuition: "Self-Gated Linear Unit"

SILU can be interpreted as:

> "Pass a fraction of the input, where the fraction is learned via sigmoid."

$$\text{SILU} = \underbrace{x}_{\text{input}} \times \underbrace{\sigma(x)}_{\text{adaptive gate in (0,1)}}$$

This mimics gating mechanisms in LSTMs or attention.

# 3. ReLU Activation Function

$$
\text{ReLU}(x) = \max(0, x) =
\begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

## 3.1 Key Properties

- **Domain**: $ \mathbb{R} $
- **Range**: $ [0, \infty) $
- **Nonlinear**, **piecewise linear**
- **Not zero-centered** (outputs ≥ 0)
- **Sparse**: sets negative inputs to 0

## 3.2 Advantages

✅ Computationally efficient
✅ No vanishing gradient for $ x > 0 $
✅ Accelerates convergence in deep networks

## 3.3 Disadvantages

❌ **Dying ReLU**: neurons can get stuck (output 0 forever)
❌ **Not smooth** at $ x = 0 $ (kink)
❌ **Not zero-centered** → can cause zig-zag gradients

## 3.4 Variants

| Variant                 | Formula                                      | Purpose                  |
| ----------------------- | -------------------------------------------- | ------------------------ |
| Leaky ReLU              | $ \max(\alpha x, x) $, $ \alpha > 0 $        | Fixes dying ReLU         |
| Parametric ReLU (PReLU) | $ \max(\alpha x, x) $, $ \alpha $ learnable  | Adaptive negative slope  |
| ELU                     | $ x $ if $ x > 0 $, else $ \alpha(e^x - 1) $ | Smoother, zero-mean-like |

## 3.5 Usage

- **Most common activation** in hidden layers of deep neural networks
- Rarely used in output layers (except regression)
