# Neural Network From Scratch — Design Document

> A minimal deep learning library built with **only NumPy**: no PyTorch, TensorFlow, or autograd frameworks. The goal is to understand and demonstrate every step of how neural networks learn — forward pass, backpropagation, and optimization.

**Status:** Design / In progress
**Language:** Python 3.10+
**Dependencies:** `numpy` (core), `matplotlib` (plots), `pytest` (tests)

---

## 1. Goals & Non-Goals

### Goals
- Implement a fully connected feedforward neural network (MLP) from first principles.
- Derive and implement backpropagation manually for every layer and loss.
- Provide a clean, modular, Keras-like API that is easy to read and extend.
- Verify correctness with **numerical gradient checking**.
- Train on real data (MNIST) and report honest, reproducible results.

### Non-Goals
- Competing with PyTorch/TensorFlow on speed or features.
- GPU support or automatic differentiation.
- Production deployment.

---

## 2. High-Level Architecture

```
Input (X)
   │
   ▼
┌───────────┐   ┌────────────┐   ┌───────────┐   ┌────────────┐
│  Dense 1  │ → │ Activation │ → │  Dense 2  │ → │ Activation │ → … → ŷ
└───────────┘   └────────────┘   └───────────┘   └────────────┘
                                                                  │
                                                                  ▼
                                                        Loss(ŷ, y) ──► L
                                                                  │
   ◄──────────────── Backpropagation (∂L/∂θ) ◄────────────────────┘
                              │
                              ▼
                     Optimizer updates θ
```

Every component implements the same two-method contract:

| Method | Responsibility |
|---|---|
| `forward(x)` | Compute output; cache values needed for the backward pass |
| `backward(grad_output)` | Return `∂L/∂input`; store `∂L/∂params` on the layer |

This makes the network a simple chain: forward runs layers in order, backward runs them in reverse.

---

## 3. Project Structure

```
nn-from-scratch/
├── nn/
│   ├── __init__.py
│   ├── layers.py          # Layer base class, Dense, Dropout
│   ├── activations.py     # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py          # MSE, BinaryCrossEntropy, CategoricalCrossEntropy
│   ├── optimizers.py      # SGD, Momentum, RMSProp, Adam
│   ├── initializers.py    # Xavier/Glorot, He
│   ├── model.py           # Sequential container: fit / predict / evaluate
│   ├── datasets.py        # MNIST download, checksum, safe loading
│   ├── diagnostics.py     # activation / gradient statistics, vanishing-exploding warnings
│   ├── gradcheck.py       # numerical gradient checking
│   ├── metrics.py         # accuracy, confusion matrix
│   └── utils.py           # batching, shuffling, one-hot, data loading
├── tests/
│   ├── test_layers.py
│   ├── test_activations.py
│   ├── test_losses.py
│   ├── test_gradient_check.py
│   └── test_optimizers.py
├── examples/
│   ├── xor.py             # Sanity check: non-linearly separable problem
│   ├── regression.py      # Fit a noisy sine curve
│   └── mnist.py           # Handwritten digit classification
├── notebooks/
│   └── walkthrough.ipynb  # Step-by-step explanation with plots
├── design.md
├── README.md
└── requirements.txt
```

---

## 4. Mathematical Foundations

### 4.1 Forward pass (Dense layer)

For a batch `X ∈ ℝ^(N×D_in)`:

```
Z = X · W + b          W ∈ ℝ^(D_in×D_out),  b ∈ ℝ^(1×D_out)
A = f(Z)               f = activation function
```

### 4.2 Backward pass (Dense layer)

Given upstream gradient `dZ = ∂L/∂Z`:

```
∂L/∂W = Xᵀ · dZ
∂L/∂b = Σ_rows(dZ)
∂L/∂X = dZ · Wᵀ
```

### 4.3 Activation functions

| Activation | f(x) | f′(x) |
|---|---|---|
| ReLU | `max(0, x)` | `1 if x > 0 else 0` |
| Sigmoid | `1 / (1 + e⁻ˣ)` | `σ(x)(1 − σ(x))` |
| Tanh | `tanh(x)` | `1 − tanh²(x)` |
| Softmax | `eˣⁱ / Σ eˣʲ` | Jacobian (fused with cross-entropy, see below) |

### 4.4 Loss functions

| Loss | Formula | Use case |
|---|---|---|
| MSE | `(1/N) Σ (ŷ − y)²` | Regression |
| Binary cross-entropy | `−(1/N) Σ [y log ŷ + (1−y) log(1−ŷ)]` | Binary classification |
| Categorical cross-entropy | `−(1/N) Σ Σ yₖ log ŷₖ` | Multi-class classification |

**Softmax + cross-entropy fusion:** combining them gives the simple, numerically stable gradient
`∂L/∂Z = (ŷ − y) / N`, so the implementation computes this directly instead of multiplying through the softmax Jacobian.

### 4.5 Optimizers

| Optimizer | Update rule |
|---|---|
| SGD | `θ ← θ − η·g` |
| Momentum | `v ← βv + g;  θ ← θ − η·v` |
| RMSProp | `s ← ρs + (1−ρ)g²;  θ ← θ − η·g / (√s + ε)` |
| Adam | `m ← β₁m + (1−β₁)g;  v ← β₂v + (1−β₂)g²;  θ ← θ − η·m̂ / (√v̂ + ε)` |

---

## 5. API Design

The public API is intentionally similar to Keras so the code reads naturally.

```python
from nn.model import Sequential
from nn.layers import Dense, Dropout
from nn.activations import ReLU, Softmax
from nn.losses import CategoricalCrossEntropy
from nn.optimizers import Adam

model = Sequential([
    Dense(784, 128), ReLU(),
    Dropout(0.2),
    Dense(128, 64),  ReLU(),
    Dense(64, 10),   Softmax(),
])

model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(lr=1e-3))
history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                    validation_data=(X_val, y_val))

preds = model.predict(X_test)
```

### Core interfaces

```python
class Layer:
    def forward(self, x, training=True): ...
    def backward(self, grad_output): ...
    def params(self) -> list[Param]: ...   # (value, grad) pairs for optimizer

class Loss:
    def forward(self, y_pred, y_true) -> float: ...
    def backward(self) -> np.ndarray: ...

class Optimizer:
    def step(self, params: list[Param]) -> None: ...
```

---

## 6. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vectorization | Batched matrix ops, no Python loops over samples | Orders of magnitude faster; idiomatic NumPy |
| Backprop style | Manual per-layer gradients (no autograd) | The point of the project is to understand the math |
| Weight init | He for ReLU, Xavier for tanh/sigmoid | Prevents vanishing/exploding activations early in training |
| Numerical stability | Subtract row max in softmax; clip probabilities in log | Avoids `overflow` and `log(0)` |
| Data layout | Rows = samples, columns = features | Matches NumPy/scikit-learn convention |
| Reproducibility | Single `np.random.default_rng(seed)` passed through | Deterministic experiments |
| Regularization | L2 weight decay + Dropout (inverted dropout) | Standard, simple, and demonstrates train/eval mode |

---

## 7. Testing & Validation Strategy

### 7.1 Gradient checking (most important)
Compare analytic gradients to centered finite differences:

```
∂L/∂θᵢ ≈ [L(θᵢ + ε) − L(θᵢ − ε)] / (2ε),   ε = 1e-5
relative_error = ‖g_analytic − g_numeric‖ / (‖g_analytic‖ + ‖g_numeric‖)
```

Pass threshold: `relative_error < 1e-6` (use `float64` for checks).

### 7.2 Unit tests
- Output shapes for every layer.
- Activation values and derivatives at known points.
- Loss values against hand-computed examples.
- Softmax rows sum to 1 and remain finite for large inputs.
- Optimizer steps on a simple quadratic converge to the minimum.

### 7.3 Sanity checks before real training
1. **Overfit a tiny batch** (e.g., 10 samples) — loss should approach ~0.
2. **Initial loss** for a 10-class softmax should be ≈ `ln(10) ≈ 2.303`.
3. **XOR** must be solved — proves non-linearity and backprop both work.

---

## 8. Experiments & Results

> Fill in with your **actual measured** numbers after running the experiments. Include the seed and hyperparameters so results are reproducible.

| Experiment | Architecture | Optimizer | Epochs | Result |
|---|---|---|---|---|
| XOR | 2 → 4 → 1 | SGD | — | *TBD* |
| Sine regression | 1 → 32 → 32 → 1 | Adam | — | *TBD (MSE)* |
| MNIST (Day 8 baseline) | 784 → 128 → 64 → 10 | SGD, lr 0.1, batch 64 | 15 | 97.62% test accuracy |
| MNIST (final) | 784 → 128 → 64 → 10 | Adam | — | *TBD (Day 13)* |

Planned plots:
- Training vs. validation loss curves
- Decision boundary for XOR / spiral dataset
- Confusion matrix and misclassified digits for MNIST
- Optimizer comparison (SGD vs. Momentum vs. Adam) on the same task

---

## 9. Implementation Roadmap

- [x] **Phase 1 – Core:** `Dense`, `ReLU`, `Sigmoid`, `MSE`, `SGD`, `Sequential`
- [x] **Phase 2 – Verification:** gradient checker, unit tests, XOR example
- [x] **Phase 3 – Classification:** `Softmax`, cross-entropy losses, mini-batching, MNIST
- [x] **Phase 4 – Better training:** Momentum, RMSProp, Adam, He/Xavier init
- [ ] **Phase 5 – Regularization:** L2, Dropout, early stopping
- [ ] **Phase 6 – Polish:** README with results, walkthrough notebook, CI (GitHub Actions running `pytest`)

### Stretch goals
- Batch Normalization
- Learning-rate schedulers (step decay, cosine)
- Convolutional layer (`im2col`) and max-pooling
- A tiny reverse-mode autograd engine (micrograd-style) as an alternative backend
- Model save/load (`.npz`)

---

## 10. Known Risks & Pitfalls

| Risk | Mitigation |
|---|---|
| Shape mismatches in backward pass | Assert shapes in tests; keep a shape table per layer |
| Exploding/vanishing gradients | Correct init, gradient-norm logging, lower LR |
| `NaN` loss from `log(0)` or `exp` overflow | Clip probabilities, stable softmax |
| Forgetting to average gradients over batch | Divide by `N` in the loss backward; verify with gradient check |
| Dropout active during evaluation | `training` flag passed through `forward` |

---

## 11. References

- Goodfellow, Bengio, Courville — *Deep Learning* (deeplearningbook.org)
- Michael Nielsen — *Neural Networks and Deep Learning* (neuralnetworksanddeeplearning.com)
- Stanford CS231n — Backpropagation & optimization notes
- Andrej Karpathy — *micrograd* and "Neural Networks: Zero to Hero"
- Kingma & Ba — *Adam: A Method for Stochastic Optimization* (2014)

---

## License

MIT
