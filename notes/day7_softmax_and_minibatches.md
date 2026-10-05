# Day 7 — Softmax, Cross-Entropy & Mini-Batching

**Goal:** multi-class classification and efficient training.

```bash
python examples/three_class_blobs.py     # saves assets/day7_three_class.png
```

![Three-class result](../assets/day7_three_class.png)

---

## 1. Softmax

Turns a row of arbitrary scores (logits) into probabilities:

```
s_k = exp(z_k) / Σ_j exp(z_j)
```

**Numerical stability.** `exp(1000)` overflows. Subtracting the row maximum changes nothing (softmax is shift-invariant) but keeps every exponent ≤ 0:

```python
shifted = z - z.max(axis=1, keepdims=True)
```

**Backward (full Jacobian).** For one row, `∂s_k/∂z_j = s_k(δ_kj − s_j)`. Given an upstream gradient `g`, that collapses to a vector formula:

```
dz = s ⊙ (g − Σ_k g_k s_k)
```

Every row of `dz` sums to 0, which makes sense: changing logits cannot change the fact that probabilities sum to 1. (Tested.)

## 2. Categorical cross-entropy

```
L = −(1/N) Σ_n Σ_k y[n,k] · log p[n,k]
```

Targets `y` are one-hot, built with `nn.utils.one_hot(labels, num_classes)`. Note the average is over **samples** (`N`), unlike MSE/BCE which average over every element. Passing integer labels by mistake raises an error that points you to `one_hot`.

## 3. The fused gradient: `(p − y) / N`

Chaining the two derivatives, almost everything cancels:

```
∂L/∂z_k = Σ_j (∂L/∂p_j)(∂p_j/∂z_k) = Σ_j (−y_j/p_j) · p_j(δ_jk − p_k) = p_k − y_k
```

(divided by `N` for the batch average). This is the same cancellation seen on Day 3 with sigmoid + BCE.

`Sequential.backward_from_loss()` uses it automatically when the model ends in `Softmax` and the loss is `CategoricalCrossEntropy`: it feeds `(p − y)/N` straight into the layer *before* the softmax and skips `Softmax.backward`. `train_step` and the gradient checker both go through this path, so the fused route is gradient-checked end to end.

### Fusion is more than a speed-up

The unfused route computes `−y/p` first. If the model is **confidently wrong**, `p` for the true class underflows, gets clipped, and `−y/p` becomes a meaningless huge number multiplied by a vanishing softmax derivative. The product comes out ≈ 0 instead of the correct −1.

I built that exact situation: logits strongly anti-correlated with the right answer (weights `−50·I`), 60 samples, 3 classes:

| Backward pass | Accuracy after training | Final loss |
|---|---|---|
| **Fused** `(p − y)/N` | **100%** | 0.0015 |
| Unfused (chain rule through softmax) | **0%**, never moves | 27.63 (= −ln 1e-12, the clip cap) |

The unfused model gets zero gradient exactly when it needs one the most. Both outcomes are pinned by tests (`test_softmax_model.py`).

## 4. Sanity check: the initial loss

A freshly initialized K-class model with small weights predicts almost-uniform probabilities, so the loss should be about `ln K`:

| K | ln K | Measured (200 samples) |
|---|---|---|
| 3 | 1.099 | 1.099 |
| 10 | 2.303 | within 0.02 |
| 20 | 2.996 | within 0.02 |

If your initial loss is far from `ln K`, something is wrong (wrong labels, no softmax, bad init). With large weights (std 0.5) the 3-class loss started at 2.50: the model is *confidently wrong*, so the check only applies to small initial weights.

## 5. Shuffling & mini-batches

`fit(x, y, epochs, batch_size=None, shuffle=True, seed=None)`:

- `batch_size=None` → full-batch gradient descent (Day 4 behaviour, bit-for-bit unchanged).
- `batch_size=32` → each epoch is split into batches; **one parameter update per batch**. Shuffling gives each epoch a new random order, so batches differ every time. This is what makes it *stochastic* gradient descent.
- The last batch is smaller when `N` isn't divisible by the batch size.
- `seed` makes the shuffling reproducible.
- The epoch loss in `history` is the **batch-size-weighted average** of the batch losses during that epoch (a test with a near-zero learning rate proves it equals the full-data loss for any uneven batching).

| Batch size | Updates per epoch (N = 64) | Gradient quality |
|---|---|---|
| 64 (full batch) | 1 | Exact, smooth |
| 16 | 4 | Noisy, but more progress per epoch |
| 1 | 64 | Very noisy |

A test confirms mini-batches make more progress per epoch than full-batch at the same learning rate.

`nn/utils.py` provides `one_hot` and `iterate_minibatches` (x and y are permuted together, so pairs stay aligned; every sample appears exactly once per epoch).

## 6. The example

3 Gaussian blobs, 240 train / 60 test, a `2 → 16 → 3` tanh network, softmax output, `SGD(lr=0.1)`, batch size 32, 60 epochs:

| Metric | Value |
|---|---|
| Train accuracy | 98.3% |
| Test accuracy | 98.3% |

The test loss bottoms out around epoch 30 and then creeps up slightly while the train loss keeps falling: the first hint of overfitting, which Day 11's regularization and Day 12's early stopping address. With only 60 test points, one sample is 1.7%, so treat the accuracy as approximate.

## Mistakes along the way

- **The "initial loss ≈ ln K" printout in my first version of the example showed 2.50, not 1.10.** I had used the std-0.5 weights that train well, but the sanity check only holds for small weights. The example now prints both values and explains the difference.
- **Two of my own new tests were wrong, not the library:** one used random labels that no model can learn from the inputs, and one expected a loss above 50 although clipping caps it at 27.6. I fixed the tests, and the second observation became a documented fact in the table above.

## What the tests prove (313 tests total, 97 new)

- Softmax: rows sum to 1, stable for ±1000, shift invariance, Jacobian check, gradient check.
- Cross-entropy: known values, `ln K` for uniform predictions, finite at `p = 0`, gradient check, fused equals unfused for ordinary logits, fused is correct and unfused vanishes when saturated.
- Whole models: gradient checks for Tanh/ReLU × K ∈ {2, 4, 7}, plus Softmax with MSE (non-fused path).
- Mini-batches: correct number of updates, weighted loss, reproducibility, validation, no mutation of data.
- Utilities: `one_hot` and `iterate_minibatches` edge cases.
- Three-class blobs: > 90% test accuracy across 5 seeds.

## Self-check questions

- Why is softmax unchanged by adding a constant to every logit? *(The factor `e^c` cancels between numerator and denominator.)*
- What is the loss of a 10-class model that outputs uniform probabilities? *(`ln 10 ≈ 2.303`.)*
- Why does the fused gradient avoid the underflow problem? *(It never divides by `p`; it subtracts `y` from `p` directly.)*
- Why weight the epoch loss by batch size? *(The last batch may be smaller; an unweighted mean would over-count it.)*
