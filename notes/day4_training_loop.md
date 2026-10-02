# Day 4 — Sequential Model, SGD & the Training Loop

**Goal:** train a network end-to-end for the first time.

## The training loop

Every epoch repeats the same four steps:

```
1. forward    y_pred = model.forward(X)
2. loss       L = loss.forward(y_pred, y)
3. backward   model.backward(loss.backward())     # fills dW, db in every Dense layer
4. update     optimizer.step(model.params())      # theta <- theta - lr * grad
```

`Sequential.train_step(x, y)` is exactly these four lines; `fit()` just calls it once per epoch and records the loss.

## Components

| Piece | Responsibility |
|---|---|
| `Sequential` | Runs layers forward, reverse-order backward, collects parameters |
| `compile(loss, optimizer)` | Attaches a loss and an optimizer (required before `fit`/`evaluate`) |
| `SGD(lr)` | `value -= lr * grad`, **in place** on the `(value, gradient)` pairs from `params()` |
| `fit(x, y, epochs)` | Full-batch training loop returning `{"loss": [...], "val_loss": [...]}` |
| `predict(x)` | Forward pass with `training=False` (matters for Dropout on Day 11) |
| `evaluate(x, y)` | Loss on given data, changes nothing |
| `summary()` | Layer list and parameter counts |

### Why in-place updates?

`Dense.params()` returns references to `W`, `b` and their gradient arrays. `SGD` modifies `W` and `b` directly (`value -= ...`), and `backward()` overwrites the gradients in place (Day 3). No copies, and nothing gets out of sync.

### Full-batch vs mini-batch

Day 4 uses the **whole dataset** for every update (full-batch gradient descent). Day 7 adds shuffling and mini-batches, which is what "stochastic" in SGD really refers to.

## Milestone: learn `y = 2x + 1`

```bash
python examples/day4_linear_regression.py
```

A single `Dense(1 → 1)` with MSE and `SGD(lr=0.3)` recovers the line in ~20 epochs:

```
Learned:  y = 1.997 * x + 0.996   (true: 2 * x + 1)
```

![Day 4 result](../assets/day4_linear_regression.png)

**Why does the loss stop at ~0.00235 instead of 0?** The data has Gaussian noise with σ = 0.05, so the best possible MSE is the noise variance, σ² = 0.0025. A loss that plateaus *at the noise floor* means the model has learned everything learnable. Going lower would mean overfitting the noise.

## Choosing the learning rate

| lr | Behaviour |
|---|---|
| Too small | Loss falls very slowly |
| Good | Fast, smooth decrease |
| Too large | Loss oscillates or explodes (`inf`/`NaN`) |

For linear regression on inputs in `[-1, 1]`, `lr = 0.3` is stable. Try `lr = 2.0` in the example to see divergence.

## What the tests prove

- `SGD` follows the update rule, works in place, and minimizes a quadratic.
- Training loss decreases monotonically for a small learning rate.
- The model recovers slope 2 and intercept 1 within 0.01.
- A hidden-layer tanh network also fits the line.
- A sigmoid + binary cross-entropy model reaches > 95% accuracy on separable 2D data.
- `predict` and `evaluate` never modify parameters.
- Training is deterministic when seeds are fixed.
- Errors are raised for missing `compile()`, mismatched sample counts, and invalid `epochs`.

## Self-check questions

- What would break if `SGD.step` rebound `value = value - lr * grad` instead of using `-=`? *(The layer's `W` would never change; only a local variable would.)*
- Why can't training loss go below the noise variance on this data? *(The noise is unpredictable from `x`.)*
- Why does `predict` pass `training=False`? *(Layers like Dropout must behave differently at inference time.)*
