"""Day 3: a full forward + backward pass on a toy batch.

Shows that every parameter receives a non-zero gradient, then takes one
manual gradient-descent step and confirms the loss goes down.
(The Sequential model and SGD optimizer arrive on Day 4.)

Run:  python examples/day3_backprop.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow running as a script

from nn.activations import Sigmoid, Tanh
from nn.layers import Dense
from nn.losses import BinaryCrossEntropy


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 3))
    y = (x.sum(axis=1, keepdims=True) > 0).astype(float)  # simple learnable target

    layers = [Dense(3, 4, seed=1), Tanh(), Dense(4, 1, seed=2), Sigmoid()]
    for layer in layers:  # larger than the default init so gradients are easy to read
        if isinstance(layer, Dense):
            layer.W = rng.normal(scale=0.5, size=layer.W.shape)
    loss_fn = BinaryCrossEntropy()

    def forward():
        out = x
        for layer in layers:
            out = layer.forward(out)
        return loss_fn.forward(out, y)

    # ---- forward + backward
    loss = forward()
    grad = loss_fn.backward()
    for layer in reversed(layers):
        grad = layer.backward(grad)

    print(f"loss before step: {loss:.6f}\n")
    print("Gradient norms:")
    for i, layer in enumerate(layers):
        for name, (value, g) in zip(("W", "b"), layer.params()):
            print(f"  layer {i} {layer!r:<16} d{name}: ||g|| = {np.linalg.norm(g):.6f}")
            assert np.linalg.norm(g) > 0, "gradient should be non-zero"

    # ---- one manual SGD step
    lr = 0.5
    for layer in layers:
        for value, g in layer.params():
            value -= lr * g
    print(f"\nloss after one step (lr={lr}): {forward():.6f}")


if __name__ == "__main__":
    main()
