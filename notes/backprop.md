# Backpropagation Derivations (Day 3)

Notation: a batch of `N` samples, rows = samples. `dZ` means `∂L/∂Z`. The upstream gradient arriving at a layer is `grad_output = ∂L/∂(layer output)`.

## The rule that drives everything

Each layer receives `∂L/∂output` and must produce:

1. `∂L/∂input` — handed to the previous layer, and
2. `∂L/∂params` — stored for the optimizer.

Chaining layers in reverse order is the chain rule applied once per layer.

---

## Dense layer: `Z = X W + b`

`X: (N, D_in)`, `W: (D_in, D_out)`, `b: (1, D_out)`, `Z: (N, D_out)`.

Element-wise, `Z[n, j] = Σᵢ X[n, i] W[i, j] + b[j]`. Differentiate `L` with respect to each quantity:

```
∂L/∂W[i, j] = Σₙ dZ[n, j] · X[n, i]    →   dW = Xᵀ · dZ          (D_in, D_out)
∂L/∂b[j]    = Σₙ dZ[n, j]              →   db = Σ_rows(dZ)       (1, D_out)
∂L/∂X[n, i] = Σⱼ dZ[n, j] · W[i, j]    →   dX = dZ · Wᵀ          (N, D_in)
```

**Shape check:** every gradient has the same shape as the thing it is a gradient of. If a shape is wrong, the formula is wrong.

**Why sum for `b`?** The bias is shared by all `N` samples, so its gradient collects a contribution from each one.

---

## Activations (element-wise)

For `A = f(Z)` applied element-wise: `dZ = dA ⊙ f′(Z)`, where `⊙` is element-wise multiplication.

| Activation | `f′` | What the layer caches |
|---|---|---|
| ReLU | `1` if `z > 0` else `0` | input `Z` |
| Sigmoid | `s(1 − s)` | output `s` |
| Tanh | `1 − t²` | output `t` |

**Derivation of the sigmoid derivative**

```
s(z) = (1 + e⁻ᶻ)⁻¹
s′(z) = e⁻ᶻ (1 + e⁻ᶻ)⁻²
      = [1 / (1 + e⁻ᶻ)] · [e⁻ᶻ / (1 + e⁻ᶻ)]
      = s · (1 − s)
```

**Tanh:** `t′ = 1 − tanh²(z)` follows from `tanh = sinh / cosh` and `cosh² − sinh² = 1`.

**ReLU at exactly 0:** the derivative is undefined; by convention we use 0.

---

## Losses

Both losses average over **all** elements (`size = N · D_out`). The `1/size` factor therefore enters the gradient *once, at the very start of backprop*, and layers never need to know the batch size.

### Mean squared error

```
L = (1/size) Σ (ŷ − y)²
∂L/∂ŷ = 2 (ŷ − y) / size
```

> Day 1 used `L = ½(a − y)²`, which has gradient `(a − y)`. The library's MSE has no ½, so its gradients are exactly **2×** the Day 1 hand values. `tests/test_backward.py::test_matches_day1_hand_example` checks this.

### Binary cross-entropy

```
L = −(1/size) Σ [ y log p + (1 − y) log(1 − p) ]
∂L/∂p = (p − y) / (p (1 − p)) / size
```

Derivation: `∂/∂p [ y log p + (1−y) log(1−p) ] = y/p − (1−y)/(1−p) = (y − p) / (p(1−p))`; negate it for the loss.

**Numerical safety:** `p` is clipped to `[ε, 1 − ε]` (default `ε = 1e-12`) so `log(0)` and division by zero cannot occur.

### Sigmoid + BCE simplifies

Chaining the BCE gradient through the sigmoid derivative:

```
∂L/∂z = (p − y) / (p(1 − p)) · p(1 − p) / size = (p − y) / size
```

The `p(1 − p)` factors cancel — the same cancellation seen later with softmax + cross-entropy (Day 7). `tests/test_losses.py::test_sigmoid_plus_bce_gradient_simplifies` verifies it.

---

## Putting it together: a 2-layer network

```
forward:   X → Dense₁ → Z₁ → Tanh → A₁ → Dense₂ → Z₂ → Sigmoid → ŷ → Loss

backward:  g = Loss.backward()          # ∂L/∂ŷ
           g = Sigmoid.backward(g)      # ∂L/∂Z₂
           g = Dense₂.backward(g)       # ∂L/∂A₁   (stores dW₂, db₂)
           g = Tanh.backward(g)         # ∂L/∂Z₁
           g = Dense₁.backward(g)       # ∂L/∂X    (stores dW₁, db₁)
```

Run `python examples/day3_backprop.py` to see non-zero gradients for every parameter and a loss decrease after one manual step.

## How correctness is checked

`tests/test_numerical_gradients.py` compares analytic gradients to centered finite differences for every hidden activation × both losses (relative error < 1e-6). Day 5 turns this into a reusable gradient checker for the whole library.

**Gotcha found while testing:** with the default init (std 0.01) the gradients are ~1e-5, so finite-difference rounding noise dominates the relative error. Gradient checks must use realistically scaled weights.

## Self-check questions

- Why is `db` a sum over rows but `dX` is not? *(`b` is shared across samples; each `X` row is its own sample.)*
- Why does the loss, not the layer, divide by `N`? *(The layer's formulas stay batch-size independent and the 1/N enters exactly once.)*
- Why do the sigmoid and BCE gradients combine so cleanly? *(`σ′ = p(1 − p)` cancels the denominator of the BCE gradient.)*
