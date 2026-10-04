"""Day 6: fit a noisy sine curve with a 1 -> 16 -> 16 -> 1 tanh network.

Shows that the network learns a smooth non-linear function from noisy samples,
using a held-out validation set to check it generalises.

Run:  python examples/regression.py
Saves assets/day6_sine_regression.png (needs matplotlib).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running as a script

from nn.activations import Tanh
from nn.layers import Dense
from nn.losses import MSE
from nn.model import Sequential
from nn.optimizers import SGD

NOISE_STD = 0.1


def make_data(n=200, seed=0, n_val=40):
    """Noisy samples of sin(x) on [-pi, pi], split into train and validation."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, size=(n, 1))
    y = np.sin(x) + rng.normal(scale=NOISE_STD, size=x.shape)
    return x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]


def train_sine(seed=0, epochs=10000, lr=0.01, hidden=16, init_scale=0.5, verbose=0):
    """Returns (model, history, (x_train, y_train, x_val, y_val))."""
    data = make_data(seed=seed)
    x_train, y_train, x_val, y_val = data
    model = Sequential([
        Dense(1, hidden, seed=seed + 1, init_scale=init_scale), Tanh(),
        Dense(hidden, hidden, seed=seed + 2, init_scale=init_scale), Tanh(),
        Dense(hidden, 1, seed=seed + 3, init_scale=init_scale),
    ])
    model.compile(loss=MSE(), optimizer=SGD(lr=lr))
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), verbose=verbose)
    return model, history, data


def error_vs_true_curve(model):
    """MSE against the noise-free sin(x): the honest measure of what was learned."""
    grid = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
    return float(np.mean((model.predict(grid) - np.sin(grid)) ** 2))


def make_plot(model, history, data, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_train, y_train, x_val, y_val = data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["loss"], label="train")
    ax1.plot(history["val_loss"], label="validation")
    ax1.axhline(NOISE_STD**2, color="gray", ls="--", label=f"noise floor ({NOISE_STD**2:.2f})")
    ax1.set(title="Loss", xlabel="epoch", ylabel="MSE", yscale="log")
    ax1.legend()

    grid = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
    ax2.scatter(x_train, y_train, s=12, alpha=0.5, label="train data")
    ax2.scatter(x_val, y_val, s=18, marker="x", color="C2", label="validation data")
    ax2.plot(grid, np.sin(grid), "k--", lw=1.5, label="true sin(x)")
    ax2.plot(grid, model.predict(grid), color="C3", lw=2, label="model")
    ax2.set(title="Fitted curve", xlabel="x", ylabel="y")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)


def main():
    model, history, data = train_sine(verbose=1)
    print(model.summary())
    print(f"\nTrain MSE:            {history['loss'][-1]:.5f}")
    print(f"Validation MSE:       {history['val_loss'][-1]:.5f}")
    print(f"Noise floor (sigma^2): {NOISE_STD**2:.5f}   <- best achievable on noisy data")
    print(f"MSE vs true sin(x):   {error_vs_true_curve(model):.5f}")

    try:
        out = ROOT / "assets" / "day6_sine_regression.png"
        make_plot(model, history, data, out)
        print(f"Saved plot to {out.relative_to(ROOT)}")
    except ImportError:
        print("matplotlib not installed - skipping plot.")


if __name__ == "__main__":
    main()
