"""Day 2: push data through a small network (forward pass only).

Run:  python examples/day2_forward_pass.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow running as a script

from nn.activations import ReLU, Sigmoid
from nn.layers import Dense


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3))  # 5 samples, 3 features

    network = [
        Dense(3, 4, seed=1), ReLU(),
        Dense(4, 2, seed=2), ReLU(),
        Dense(2, 1, seed=3), Sigmoid(),
    ]

    print(f"input:  {x.shape}")
    out = x
    for layer in network:
        out = layer.forward(out)
        print(f"  {layer!r:<18} -> {out.shape}")

    print("\nPredictions (untrained, so ~0.5 everywhere):")
    print(np.round(out.ravel(), 4))


if __name__ == "__main__":
    main()
