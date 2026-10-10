# Roadmap — Neural Network From Scratch

A 14-day plan to build the library described in [`design.md`](design.md). Each day has a clear goal, concrete tasks, and a "done when" check so progress is easy to verify (and easy to show in your commit history).

**Suggested pace:** 2–4 hours per day. Commit at the end of every day with a message like `Day 3: Dense layer + backward pass`.

## Progress Overview

| Day | Focus | Phase | Status |
|---|---|---|---|
| 1 | Setup & math refresher | Setup | [x] |
| 2 | Dense layer (forward) & activations | Phase 1 | [x] |
| 3 | Backpropagation & losses | Phase 1 | [x] |
| 4 | Sequential model, SGD & first training loop | Phase 1 | [x] |
| 5 | Gradient checking & unit tests | Phase 2 | [x] |
| 6 | XOR & regression examples | Phase 2 | [x] |
| 7 | Softmax, cross-entropy & mini-batching | Phase 3 | [x] |
| 8 | MNIST classification | Phase 3 | [x] |
| 9 | Weight initialization & Momentum | Phase 4 | [x] |
| 10 | RMSProp & Adam | Phase 4 | [x] |
| 11 | L2 regularization & Dropout | Phase 5 | [x] |
| 12 | Early stopping, metrics & visualizations | Phase 5 | [x] |
| 13 | Experiments & results | Phase 6 | [ ] |
| 14 | Documentation, CI & release | Phase 6 | [ ] |

---

## Day 1 — Setup & Math Refresher

**Goal:** Have a working repo skeleton and understand the math you'll implement.

- [x] Create the GitHub repo and clone it locally
- [x] Create the folder structure from `design.md` (`nn/`, `tests/`, `examples/`, `notebooks/`)
- [x] Set up a virtual environment and `requirements.txt` (`numpy`, `matplotlib`, `pytest`)
- [x] Add `.gitignore` (Python, `.venv`, `__pycache__`, data files)
- [x] Review: matrix multiplication, chain rule, partial derivatives
- [x] Work through one tiny network (2 inputs → 1 output) **by hand** on paper

**Done when:** `pytest` runs (even with zero tests) and the first commit is pushed.

---

## Day 2 — Dense Layer (Forward) & Activations

**Goal:** Data can flow forward through the network.

- [x] Implement the `Layer` base class in `layers.py`
- [x] Implement `Dense.forward` with weights `W` and bias `b`
- [x] Implement `ReLU`, `Sigmoid`, and `Tanh` (forward only)
- [x] Add a simple random initializer (improved on Day 9)
- [x] Write shape tests: input `(N, D_in)` → output `(N, D_out)`

**Done when:** A two-layer forward pass on random data returns the expected shape with no errors.

---

## Day 3 — Backpropagation & Losses

**Goal:** Gradients flow backward through every component.

- [x] Implement `Dense.backward` (`dW`, `db`, `dX`)
- [x] Implement backward for `ReLU`, `Sigmoid`, `Tanh`
- [x] Implement `MSE` loss (forward + backward)
- [x] Implement `BinaryCrossEntropy` with probability clipping
- [x] Write the derivations as comments or in a short `notes/backprop.md`

**Done when:** You can compute a loss and get non-zero gradients for every parameter on a toy batch.

---

## Day 4 — Sequential Model, SGD & First Training Loop

**Goal:** Train a network end-to-end for the first time.

- [x] Implement `Sequential` with `forward`, `backward`, and `compile`
- [x] Implement `SGD` optimizer
- [x] Implement `fit()` with an epoch loop and a loss history
- [x] Implement `predict()` and `evaluate()`
- [x] Train on a tiny synthetic dataset (e.g., learn `y = 2x + 1`)

**Done when:** Training loss decreases steadily and the model fits the linear function.

---

## Day 5 — Gradient Checking & Unit Tests

**Goal:** Prove the backprop implementation is correct.

- [x] Implement a numerical gradient checker (centered finite differences)
- [x] Gradient-check every layer, activation, and loss
- [x] Add unit tests for shapes, activation values, and loss values
- [x] Add a test that a tiny batch can be overfit to near-zero loss
- [x] Fix any bugs found (there will be some — that's the point)

**Done when:** All gradient checks show relative error below `1e-6` and `pytest` passes.

---

## Day 6 — XOR & Regression Examples

**Goal:** Show that the network learns non-linear functions.

- [x] `examples/xor.py`: solve XOR with a 2 → 4 → 1 network
- [x] `examples/regression.py`: fit a noisy sine curve
- [x] Plot the loss curves and the fitted curve
- [x] Save plots to an `assets/` folder for the README

**Done when:** XOR reaches 100% accuracy and the sine fit visibly matches the data.

---

## Day 7 — Softmax, Cross-Entropy & Mini-Batching

**Goal:** Support multi-class classification and efficient training.

- [x] Implement a numerically stable `Softmax` (subtract the row max)
- [x] Implement `CategoricalCrossEntropy` and the fused softmax + CE gradient `(ŷ − y) / N`
- [x] Add one-hot encoding in `utils.py`
- [x] Implement shuffling and mini-batch iteration in `fit()`
- [x] Gradient-check the new components
- [x] Sanity check: initial loss on 10 classes ≈ 2.30

**Done when:** Softmax rows sum to 1, there are no `NaN`s, and mini-batch training works on a toy 3-class dataset.

---

## Day 8 — MNIST Classification

**Goal:** Train on real data.

- [x] Write a data loader for MNIST (normalize pixels to `[0, 1]`, flatten to 784)
- [x] Create a train/validation/test split
- [x] Add `accuracy` to `metrics.py`
- [x] Train a 784 → 128 → 64 → 10 network with SGD
- [x] Record training and validation loss/accuracy per epoch

**Done when:** The model is clearly learning (well above 10% chance accuracy) and you have saved baseline numbers.

---

## Day 9 — Weight Initialization & Momentum

**Goal:** Train faster and more stably.

- [x] Implement Xavier/Glorot and He initializers
- [x] Use He for ReLU layers and Xavier for tanh/sigmoid
- [x] Implement the `Momentum` optimizer
- [x] Compare random init vs. He init (plot the first few epochs of loss)
- [x] Log activation and gradient statistics to spot vanishing/exploding values

**Done when:** He init + Momentum trains noticeably faster than the Day 8 baseline.

---

## Day 10 — RMSProp & Adam

**Goal:** Add modern adaptive optimizers.

- [x] Implement `RMSProp`
- [x] Implement `Adam` with bias correction
- [x] Test each optimizer on a simple quadratic function
- [x] Run SGD vs. Momentum vs. RMSProp vs. Adam on the same MNIST setup
- [x] Plot the optimizer comparison

**Done when:** All four optimizers converge on the quadratic test and the comparison plot is saved.

---

## Day 11 — L2 Regularization & Dropout

**Goal:** Reduce overfitting.

- [x] Add L2 weight penalty to the objective and its gradient (Dense weights only)
- [x] Implement inverted `Dropout` with a `training` flag
- [x] Make sure `predict()` runs in evaluation mode (dropout off)
- [x] Numerically verify L2 gradients and test dropout with a fixed mask
- [x] Add a reproducible train/validation comparison example

**Done when:** L2 and Dropout are implemented and tested, evaluation disables dropout, and the comparison script reports actual train/validation metrics. Whether the gap shrinks is an empirical result, not a hard-coded guarantee.

---

## Day 12 — Early Stopping, Metrics & Visualizations

**Goal:** Improve the training experience and analysis tools.

- [x] Implement early stopping (patience on validation loss, min_delta, optional best-weight restoration)
- [x] Add a confusion matrix and per-class accuracy
- [x] Add a plot for misclassified MNIST digits (via `examples/day12_analysis.py --mnist`)
- [x] Plot decision boundaries for a 2D three-class dataset
- [x] Add configurable per-epoch progress output (loss, metrics, elapsed time)

**Done when:** Early stopping halts training correctly and all plots are generated from scripts.

---

## Day 13 — Experiments & Results

**Goal:** Produce honest, reproducible results.

- [ ] Fix random seeds and record all hyperparameters
- [ ] Run the final MNIST experiment with your best configuration
- [ ] Fill in the results table in `design.md` (Section 8) with your **real** numbers
- [ ] Run a small ablation: with/without Dropout, with/without He init, different learning rates
- [ ] Write a walkthrough notebook in `notebooks/walkthrough.ipynb`

**Done when:** Anyone can clone the repo, run one command, and reproduce your reported numbers.

---

## Day 14 — Documentation, CI & Release

**Goal:** Make the repo look and feel professional.

- [ ] Write `README.md`: overview, install steps, quick example, results, plots
- [ ] Add docstrings and type hints to the public API
- [ ] Add a GitHub Actions workflow that runs `pytest` on every push
- [ ] Add a CI badge and an MIT `LICENSE`
- [ ] Clean up code (consistent style, remove dead code, run a formatter like `black`)
- [ ] Tag a release (`v0.1.0`) and add repo topics: `neural-network`, `numpy`, `deep-learning`, `machine-learning`, `from-scratch`

**Done when:** The CI badge is green and the README explains the project clearly to a stranger in under two minutes.

---

## Stretch Goals (After Day 14)

| Item | Difficulty |
|---|---|
| Batch Normalization | Medium |
| Learning-rate schedulers (step decay, cosine) | Easy |
| Model save/load (`.npz`) | Easy |
| Convolutional layer with `im2col` + max-pooling | Hard |
| Tiny reverse-mode autograd engine (micrograd-style) | Hard |
| Blog post explaining backprop with your own diagrams | Medium |

---

## Daily Checklist

1. Write the code for the day's goal.
2. Add or update tests.
3. Run `pytest` — everything must pass before you commit.
4. Update the progress table at the top of this file.
5. Commit and push with a descriptive message.

## Tips

- **Don't skip gradient checking.** It catches most backprop bugs and is a strong signal of quality.
- **If you fall behind, cut scope, not quality.** Skip a stretch goal rather than leaving tests out.
- **Keep commits small and daily.** A steady commit history shows real progress.
