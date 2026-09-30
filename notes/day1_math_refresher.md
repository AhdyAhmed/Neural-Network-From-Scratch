# Day 1 — Math Refresher & Hand-Worked Example

The three ideas behind everything in this project: **matrix multiplication**, **the chain rule**, and **gradient descent**.

---

## 1. Matrix multiplication (the forward pass)

A dense layer for a batch of `N` samples:

```
Z = X · W + b
X: (N, D_in)     W: (D_in, D_out)     b: (1, D_out)     Z: (N, D_out)
```

Rule of thumb: **inner dimensions must match, outer dimensions give the result shape.**
`(N, D_in) · (D_in, D_out) → (N, D_out)`. The bias `b` is broadcast across all `N` rows.

## 2. Partial derivatives & the chain rule (the backward pass)

If `L` depends on `a`, `a` depends on `z`, and `z` depends on `w`:

```
∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w
```

Backpropagation is just this rule applied layer by layer, reusing the products already computed for later layers.

## 3. Gradient descent (the learning step)

```
θ ← θ − η · ∂L/∂θ
```

`η` is the learning rate. Move each parameter a small step **against** the gradient and the loss goes down.

---

## Worked example: one sigmoid neuron

**Model**

```
z = w₁x₁ + w₂x₂ + b
a = σ(z) = 1 / (1 + e⁻ᶻ)
L = ½ (a − y)²
```

**Values**

| x₁ | x₂ | w₁ | w₂ | b | y |
|---|---|---|---|---|---|
| 1 | 2 | 0.5 | −0.5 | 0.1 | 1 |

### Forward pass

```
z = (0.5)(1) + (−0.5)(2) + 0.1 = −0.4
a = σ(−0.4) = 1 / (1 + e^0.4) ≈ 0.401312
L = ½ (0.401312 − 1)² ≈ 0.179213
```

### Backward pass

Local derivatives:

```
∂L/∂a = a − y              = 0.401312 − 1     = −0.598688
∂a/∂z = a(1 − a)           = 0.401312 · 0.598688 ≈ 0.240261
∂z/∂w₁ = x₁ = 1      ∂z/∂w₂ = x₂ = 2      ∂z/∂b = 1
```

Chain them together:

```
∂L/∂z  = ∂L/∂a · ∂a/∂z = −0.598688 · 0.240261 ≈ −0.143841

∂L/∂w₁ = ∂L/∂z · x₁ = −0.143841 · 1 ≈ −0.143841
∂L/∂w₂ = ∂L/∂z · x₂ = −0.143841 · 2 ≈ −0.287682
∂L/∂b  = ∂L/∂z · 1  ≈ −0.143841
```

### One gradient-descent step (η = 0.5)

```
w₁ ← 0.5  − 0.5 · (−0.143841) ≈  0.571921
w₂ ← −0.5 − 0.5 · (−0.287682) ≈ −0.356159
b  ← 0.1  − 0.5 · (−0.143841) ≈  0.171921
```

New loss ≈ **0.121091** (down from 0.179213). The neuron moved toward the target `y = 1`.

### Verify it yourself

```bash
python examples/day1_hand_example.py
pytest tests/test_day1_hand_example.py
```

The script recomputes everything and confirms the analytic gradients match **numerical gradients** (centered finite differences) to about 1e-11 — the same technique used on Day 5 to check the full library.

---

## Key takeaways

1. The gradient of a weight is the **upstream gradient × the local input** (`∂L/∂w = ∂L/∂z · x`).
2. The gradient of a bias is just the **upstream gradient**.
3. Sigmoid's derivative is expressed with its own output: `σ′ = a(1 − a)` — no need to recompute anything.
4. Always verify analytic gradients numerically. It catches almost every backprop bug.

## Self-check questions

- What shape is `∂L/∂W` for a layer with `W: (D_in, D_out)`? *(Same as `W`.)*
- Why does `∂L/∂b` sum over the batch dimension? *(`b` is shared by every sample.)*
- What happens to `∂a/∂z` when the sigmoid saturates (`z` very large or very negative)? *(It approaches 0 → vanishing gradients.)*
