# Day 6 — XOR & Regression Examples

**Goal:** show that the network learns *non-linear* functions.

```bash
python examples/xor.py          # saves assets/day6_xor.png
python examples/regression.py   # saves assets/day6_sine_regression.png
```

New this day: `Dense(..., init_scale=...)` sets the standard deviation of the initial weights (default stays `0.01`). It is a stopgap until He/Xavier initializers arrive on Day 9.

---

## 1. XOR: why a hidden layer matters

| x1 | x2 | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A model with no hidden layer (`Dense(2→1)` + sigmoid) can only draw **one straight line**. No line separates `{(0,1), (1,0)}` from `{(0,0), (1,1)}`, so it is stuck at 50% accuracy and its loss stays at `ln 2 ≈ 0.693`, the loss of always answering "0.5".

A `2 → 4 → 1` network (tanh hidden layer, sigmoid output, binary cross-entropy, `SGD(lr=0.5)`, 3000 epochs) bends the decision boundary around the points:

![XOR result](../assets/day6_xor.png)

| Model | Accuracy | Final loss |
|---|---|---|
| No hidden layer | 50% | 0.69315 |
| 2 → 4 → 1 (tanh) | **100%** | 0.0012 |

### How reliable is it?

Solving XOR can fail on a bad initialization, so I measured instead of trusting one lucky seed:

| Hidden activation | Seeds that reach 100% accuracy |
|---|---|
| Tanh | **30 / 30** |
| ReLU | 17 / 30 |

ReLU fails more often because a hidden unit whose input is negative for all four points outputs 0 and receives **zero gradient**, so it never recovers ("dead ReLU"). With only 4 hidden units, losing one or two is enough to fail. This is a good reason the tests use Tanh and why He initialization (Day 9) matters for ReLU.

### Initialization scale matters

With the default `init_scale=0.01`, all hidden units start almost identical and near zero, so learning is extremely slow. `init_scale=0.5` breaks the symmetry. This is the first concrete example of why initialization is a topic of its own.

---

## 2. Regression: a noisy sine curve

Data: 200 points `y = sin(x) + N(0, 0.1²)` on `[-π, π]`, split 160 train / 40 validation.
Model: `1 → 16 → 16 → 1` with tanh (321 parameters), MSE, `SGD(lr=0.01)`, 10,000 full-batch epochs.

![Sine regression](../assets/day6_sine_regression.png)

| Metric | Value |
|---|---|
| Train MSE | 0.0110 |
| Validation MSE | 0.0075 |
| Noise floor (σ² = 0.01) | 0.0100 |
| **MSE vs. the true `sin(x)`** | **0.00037** |

### Reading the numbers honestly

- **The noise floor** (0.01) is the lowest MSE any model can reach on *noisy* targets. Train MSE sits just above it, which is the sign of a model that has learned the signal without memorising noise.
- **Validation MSE (0.0075) is below the noise floor.** That is luck of the small 40-point split (a few quiet samples), not a sign the model beats the limit. Don't read much into one validation set.
- **The best measure is MSE against the noise-free `sin(x)`: 0.00037.** The fitted curve is within about 0.02 of the true one on average, much more precise than the noisy data it learned from.

### The learning rate, found the hard way

The loss curve is the quickest way to diagnose the learning rate. I tried several on this exact problem:

| lr | What happened |
|---|---|
| 0.05 | Loss spikes during training (5 jumps of > 50% between epochs); still converges |
| 0.02 | Smooth-looking train loss, but the validation loss **flip-flops every epoch** (0.0085 ↔ 0.0075): the update overshoots a steep direction each step, the "edge of stability" |
| **0.01** | **Smooth, no spikes, no oscillation, best fit** ← used |
| 0.005 | Same result, just slower |
| 0.1 (with 32 hidden units) | **Diverges to NaN** |

Moral: a loss curve that looks "mostly fine" can hide oscillation. The test `test_sine_training_loss_is_stable` guards against it.

---

## What the tests prove (`tests/test_examples.py`, 26 new → 216 total)

- XOR reaches 100% accuracy for **every one of 10 seeds**, with probabilities within 0.05 of the targets.
- A linear model never exceeds 75% accuracy on XOR and has loss above 0.6, across 5 seeds.
- The sine fit stays near the noise floor on held-out data and has MSE below 0.002 against the true curve.
- Training is stable: no loss spikes, and the validation loss does not oscillate.

## Self-check questions

- Why does XOR's linear baseline end at loss ≈ 0.693? *(It learns to output 0.5 everywhere: `−ln 0.5 = 0.693`.)*
- What does a hidden unit with a non-linear activation add? *(It lets the network bend the decision boundary instead of drawing one line.)*
- Why can validation loss legitimately be lower than train loss here? *(Small, noisy validation set, and training loss is measured before each update.)*
