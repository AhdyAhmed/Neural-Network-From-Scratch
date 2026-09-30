# Day 2 — Forward Pass: Dense Layer & Activations

**Goal:** data can flow forward through the network. Backward passes come on Day 3.

## The Layer contract

Every component implements the same interface, so a network is just a list of layers:

| Method | Day 2 status |
|---|---|
| `forward(x, training=True)` | Implemented for `Dense`, `ReLU`, `Sigmoid`, `Tanh` |
| `backward(grad_output)` | Raises `NotImplementedError` (Day 3) |
| `params()` | Returns `(value, gradient)` pairs for the optimizer (used on Day 4) |

Each layer **caches** what its backward pass will need (the input for `Dense`/`ReLU`, the output for `Sigmoid`/`Tanh`). Doing this now means Day 3 only adds the math.

## Dense layer

```
Z = X @ W + b
X: (N, D_in)   W: (D_in, D_out)   b: (1, D_out)   Z: (N, D_out)
```

- Rows are samples, columns are features.
- `b` has shape `(1, D_out)` and is broadcast across the batch.
- Input shape is validated so bugs fail loudly with a clear message.

### Initialization (temporary)

Weights ~ `N(0, 0.01²)`, biases = 0. This is enough to make things run, but too small for deep networks: activations shrink layer by layer, so an untrained network outputs almost exactly 0.5 after a sigmoid (see `examples/day2_forward_pass.py`). **Day 9** replaces this with He / Xavier initialization and measures the difference.

## Activations

| Activation | Forward | Range |
|---|---|---|
| ReLU | `max(0, x)` | `[0, ∞)` |
| Sigmoid | `1 / (1 + e⁻ˣ)` | `(0, 1)` |
| Tanh | `tanh(x)` | `(−1, 1)` |

### Numerically stable sigmoid

`exp(-x)` overflows for large negative `x` (e.g. `x = -1000`). The fix is to use two equivalent forms:

```
x >= 0:  1 / (1 + exp(-x))
x <  0:  exp(x) / (1 + exp(x))
```

Both only ever exponentiate a non-positive number, so nothing overflows. `tests/test_activations.py` enforces this by turning floating-point overflow warnings into errors.

## Verification

```bash
pytest                                  # 35 tests
python examples/day2_forward_pass.py    # shape trace through a 3-layer network
```

A useful check: `Dense(2→1) + Sigmoid` with the Day 1 weights reproduces the hand-calculated `a = 0.401312`.

## Self-check questions

- Why does a `Dense(5, 3)` layer reject an input of shape `(2, 4)`? *(Inner dimensions must match: 4 ≠ 5.)*
- Why cache the *output* for sigmoid but the *input* for ReLU? *(σ′ = a(1 − a) uses the output; ReLU′ needs to know where x > 0.)*
- What goes wrong with a naive sigmoid at `x = -1000`? *(`exp(1000)` overflows to `inf`.)*
