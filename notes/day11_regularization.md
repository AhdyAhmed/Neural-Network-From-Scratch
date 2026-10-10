# Day 11 — L2 Regularization & Dropout

**Goal:** reduce overfitting with two complementary regularization techniques while keeping the framework NumPy-only and manually differentiated.

## 1. L2 regularization

For the data objective `L_data`, the model minimizes

`L_total = L_data + (λ / 2) * Σ_l ||W_l||²_F`

where the sum covers each `Dense` layer's weight matrix. Biases are not penalized. The factor `1/2` makes the derivative straightforward:

`∂L_total / ∂W_l = ∂L_data / ∂W_l + λ W_l`

Use it via `model.compile(loss=..., optimizer=..., l2=1e-3)`. The default `l2=0` preserves existing behavior. Training and validation loss history include the penalty; `predict()` remains a pure prediction operation. `evaluate()` reports the same regularized objective used for model selection.

This is often called L2 regularization or weight decay. The implementation adds the penalty gradient before the optimizer step; it is not decoupled weight decay such as AdamW.

## 2. Inverted Dropout

`Dropout(rate=p)` independently drops activations with probability `p` during training. A Bernoulli mask `M` is scaled by the keep probability:

`M_i ~ Bernoulli(1-p) / (1-p)`

`y = x ⊙ M`, so `E[y] = x`. During backpropagation, the same mask multiplies the incoming gradient. In evaluation mode, the layer is an identity function; no scaling is needed at inference.

```python
from nn.layers import Dense, Dropout
from nn.activations import ReLU
from nn.model import Sequential

model = Sequential([
    Dense(100, 64, initializer="he"), ReLU(), Dropout(0.2),
    Dense(64, 10),
])
model.compile(loss=..., optimizer=..., l2=1e-4)
```

The example omits a loss/activation pairing for brevity; use the appropriate output activation and loss for the task.

## 3. Reproducible experiment

Run:

```bash
python examples/day11_regularization.py
```

The script compares no regularization, L2 (`λ=1e-3`), and Dropout (`p=0.15`) on a small noisy-sine regression task. It saves a plot to `assets/day11_regularization.png` and raw metrics/hyperparameters to `results/day11_regularization.json`. Results are measured, not assumed: regularization can reduce overfitting, but excessive regularization can also underfit.

## 4. Verification

`tests/test_day11_regularization.py` checks inverted scaling, inference identity, input validation, fixed-mask dropout derivatives, the L2 objective and weight gradients against centered finite differences, and bias exclusion. The full suite should continue to pass.

## Self-check questions

1. Why does the L2 penalty use `λ/2`? So its derivative is exactly `λW`.
2. Why aren't biases regularized here? Weight magnitude is the primary target; bias regularization is optional and often unnecessary.
3. Why scale retained dropout activations by `1/(1-p)`? It preserves their expectation during training and lets inference use the identity function.
4. Why must gradient checking freeze the dropout mask? Otherwise finite differences compare different random functions at `θ+ε` and `θ-ε`.

## Day 12 follow-up: training controls and evaluation

Day 12 adds optional early stopping to `Sequential.fit()`. Set `early_stopping=True`
with `validation_data`, then configure `patience`, `min_delta`, and
`restore_best_weights`. The default remains disabled, so existing calls behave as
before. When enabled, the best model parameters are snapshotted by validation loss
and restored at the end by default.

Classification evaluation helpers are available in `nn.metrics`: `confusion_matrix`
(rows are actual classes, columns are predicted classes) and `per_class_accuracy`
(per-class recall; classes with no true examples are `nan`). Plotting helpers live in
`nn.visualization` and import Matplotlib only when called.
