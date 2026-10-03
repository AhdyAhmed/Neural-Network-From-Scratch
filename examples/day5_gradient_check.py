"""Day 5: gradient-check every component of the library and print a report.

Run:  python examples/day5_gradient_check.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow running as a script

from nn.activations import ReLU, Sigmoid, Tanh
from nn.gradcheck import check_layer, check_loss, check_model
from nn.layers import Dense
from nn.losses import MSE, BinaryCrossEntropy
from nn.model import Sequential
from nn.optimizers import SGD

TOL = 1e-6


def scaled_dense(n_in, n_out, seed):
    layer = Dense(n_in, n_out, seed=seed)
    rng = np.random.default_rng(seed + 100)
    layer.W = rng.normal(scale=0.7, size=layer.W.shape)
    layer.b = rng.normal(scale=0.2, size=layer.b.shape)
    layer.dW, layer.db = np.zeros_like(layer.W), np.zeros_like(layer.b)
    return layer


def report(title, result):
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {title:<34} max rel. error = {result.max_error:.2e}")
    return result.passed


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 4))
    x_nz = np.where(np.abs(x) < 0.05, 0.1, x)  # keep ReLU away from its kink
    results = []

    print(f"Gradient check (centered differences, eps=1e-5, tolerance {TOL:g})\n")

    print("Layers")
    results.append(report("Dense(4 -> 3)", check_layer(scaled_dense(4, 3, 1), x, tol=TOL)))
    for act in (ReLU(), Sigmoid(), Tanh()):
        results.append(report(repr(act), check_layer(act, x_nz, tol=TOL)))

    print("\nLosses")
    p = rng.uniform(0.05, 0.95, size=(8, 1))
    y_bin = (rng.random((8, 1)) > 0.5).astype(float)
    results.append(report("MSE", check_loss(MSE(), rng.normal(size=(8, 1)), rng.normal(size=(8, 1)), tol=TOL)))
    results.append(report("BinaryCrossEntropy", check_loss(BinaryCrossEntropy(), p, y_bin, tol=TOL)))

    print("\nWhole models")
    for hidden in (ReLU, Tanh, Sigmoid):
        for loss in (MSE, BinaryCrossEntropy):
            model = Sequential([
                scaled_dense(4, 6, 1), hidden(), scaled_dense(6, 6, 2), hidden(),
                scaled_dense(6, 1, 3), Sigmoid(),
            ])
            model.compile(loss=loss(), optimizer=SGD(lr=0.1))
            title = f"Dense-{hidden.__name__}x2 + {loss.__name__}"
            results.append(report(title, check_model(model, x_nz, y_bin, tol=TOL)))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
