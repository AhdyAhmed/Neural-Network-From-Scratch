"""Day 6: solve XOR, a problem no single linear layer can solve.

    x1 x2 | XOR
     0  0 |  0
     0  1 |  1
     1  0 |  1
     1  1 |  0

A model without a hidden layer draws one straight line, and no straight line
separates {(0,1), (1,0)} from {(0,0), (1,1)}. A 2 -> 4 -> 1 network with a
non-linear hidden layer can.

Run:  python examples/xor.py
Saves assets/day6_xor.png (needs matplotlib).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running as a script

from nn.activations import Sigmoid, Tanh
from nn.layers import Dense
from nn.losses import BinaryCrossEntropy
from nn.model import Sequential
from nn.optimizers import SGD

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)


def train_xor(seed=0, epochs=3000, lr=0.5, init_scale=0.5, verbose=0):
    """Train the 2 -> 4 -> 1 network. Returns (model, history)."""
    model = Sequential([
        Dense(2, 4, seed=seed, init_scale=init_scale), Tanh(),
        Dense(4, 1, seed=seed + 1000, init_scale=init_scale), Sigmoid(),
    ])
    model.compile(loss=BinaryCrossEntropy(), optimizer=SGD(lr=lr))
    history = model.fit(X, Y, epochs=epochs, verbose=verbose)
    return model, history


def train_linear_baseline(seed=0, epochs=3000, lr=0.5):
    """The same task with NO hidden layer (logistic regression). It cannot succeed."""
    model = Sequential([Dense(2, 1, seed=seed, init_scale=0.5), Sigmoid()])
    model.compile(loss=BinaryCrossEntropy(), optimizer=SGD(lr=lr))
    history = model.fit(X, Y, epochs=epochs, verbose=0)
    return model, history


def accuracy(model):
    return float(np.mean((model.predict(X) > 0.5) == Y))


def make_plot(mlp, mlp_hist, lin, lin_hist, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(lin_hist["loss"], label="no hidden layer", color="C1")
    axes[0].plot(mlp_hist["loss"], label="2 -> 4 -> 1", color="C0")
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="binary cross-entropy", yscale="log")
    axes[0].legend()

    g = np.linspace(-0.5, 1.5, 200)
    xx, yy = np.meshgrid(g, g)
    grid = np.c_[xx.ravel(), yy.ravel()]
    for ax, model, title in ((axes[1], lin, "Linear model (cannot solve XOR)"),
                             (axes[2], mlp, "Hidden layer solves XOR")):
        zz = model.predict(grid).reshape(xx.shape)
        cs = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.8, vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap="RdBu_r", edgecolors="k", s=120, vmin=0, vmax=1)
        ax.set(title=f"{title}\naccuracy {accuracy(model):.0%}", xlabel="x1", ylabel="x2")
    fig.colorbar(cs, ax=axes[2], label="P(output = 1)")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)


def main():
    mlp, mlp_hist = train_xor(verbose=1)
    lin, lin_hist = train_linear_baseline()

    print("\n  x1  x2 | target | P(1)   | predicted")
    print("  -----------------------------------")
    for xi, yi, p in zip(X, Y, mlp.predict(X)):
        print(f"  {xi[0]:.0f}   {xi[1]:.0f}  |   {yi[0]:.0f}    | {p[0]:.4f} |    {int(p[0] > 0.5)}")

    print(f"\nWith hidden layer: accuracy {accuracy(mlp):.0%}, final loss {mlp_hist['loss'][-1]:.5f}")
    print(f"Without hidden layer: accuracy {accuracy(lin):.0%}, final loss {lin_hist['loss'][-1]:.5f}")

    try:
        out = ROOT / "assets" / "day6_xor.png"
        make_plot(mlp, mlp_hist, lin, lin_hist, out)
        print(f"Saved plot to {out.relative_to(ROOT)}")
    except ImportError:
        print("matplotlib not installed - skipping plot.")


if __name__ == "__main__":
    main()
