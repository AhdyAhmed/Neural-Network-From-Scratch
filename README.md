# Neural Network From Scratch

A minimal deep learning library built with **only NumPy** — no PyTorch, no TensorFlow, no autograd.
The goal is to understand (and demonstrate) every step of how a neural network learns: the forward pass, backpropagation, and optimization.

> **Status:** 🚧 In progress — Day 2 of 14 (forward pass). See the [roadmap](roadmap.md).

## Why this project?

Frameworks hide the math. Here every gradient is derived by hand, implemented manually, and verified with numerical gradient checking.

## Planned features

- Fully connected layers (`Dense`) with manual backpropagation
- Activations: ReLU, Sigmoid, Tanh, Softmax
- Losses: MSE, Binary / Categorical Cross-Entropy
- Optimizers: SGD, Momentum, RMSProp, Adam
- Regularization: L2, Dropout
- Numerical gradient checking + unit tests
- Examples: XOR, sine regression, MNIST

## Getting started

```bash
git clone https://github.com/AhdyAhmed/Neural-Network-From-Scratch.git
cd Neural-Network-From-Scratch

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                           # should pass
python examples/day1_hand_example.py
python examples/day2_forward_pass.py
```

## Project structure

```
nn/          Library code (layers, activations, losses, optimizers, model)
tests/       Unit tests and gradient checks
examples/    Runnable demos
notes/       Math derivations and study notes
notebooks/   Walkthrough notebooks
design.md    Architecture and design decisions
roadmap.md   Day-by-day plan
```

## Documentation

- [Design document](design.md)
- [14-day roadmap](roadmap.md)
- [Day 1 — math refresher and hand-worked example](notes/day1_math_refresher.md)
- [Day 2 — forward pass: Dense layer and activations](notes/day2_forward_pass.md)

## Results

Results will be added here once experiments are run (Day 13).

## License

MIT — see [LICENSE](LICENSE).
