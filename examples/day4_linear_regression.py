"""Day 4 milestone: train a network end-to-end to learn y = 2x + 1.

Run:  python examples/day4_linear_regression.py
Saves a plot to assets/day4_linear_regression.png (needs matplotlib).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running as a script

from nn.layers import Dense
from nn.losses import MSE
from nn.model import Sequential
from nn.optimizers import SGD


def main():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(100, 1))
    y = 2 * x + 1 + rng.normal(scale=0.05, size=x.shape)  # small noise

    model = Sequential([Dense(1, 1, seed=0)])
    model.compile(loss=MSE(), optimizer=SGD(lr=0.3))
    print(model.summary(), "\n")

    history = model.fit(x, y, epochs=200)

    dense = model.layers[0]
    print(f"\nLearned:  y = {dense.W[0, 0]:.3f} * x + {dense.b[0, 0]:.3f}   (true: 2 * x + 1)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["loss"])
    ax1.set(title="Training loss", xlabel="epoch", ylabel="MSE", yscale="log")

    grid = np.linspace(-1, 1, 50).reshape(-1, 1)
    ax2.scatter(x, y, s=12, alpha=0.6, label="data")
    ax2.plot(grid, model.predict(grid), color="C3", label="model")
    ax2.set(title="Fitted line", xlabel="x", ylabel="y")
    ax2.legend()

    out = ROOT / "assets" / "day4_linear_regression.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved plot to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
