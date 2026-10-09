"""Day 11 demo: compare no regularization, L2, and dropout on noisy sine regression.

Run from the repository root:
    python examples/day11_regularization.py

The script saves a learning-curve plot and a JSON summary under results/ and assets/.
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from nn.activations import Tanh
from nn.layers import Dense, Dropout
from nn.losses import MSE
from nn.model import Sequential
from nn.optimizers import Adam

ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"


def make_model(kind: str, seed: int) -> Sequential:
    rng = np.random.default_rng(seed)
    layers = [Dense(1, 64, rng=rng, initializer="xavier"), Tanh()]
    if kind == "dropout":
        layers.append(Dropout(0.15, rng=rng))
    layers += [Dense(64, 64, rng=rng, initializer="xavier"), Tanh()]
    if kind == "dropout":
        layers.append(Dropout(0.15, rng=rng))
    layers.append(Dense(64, 1, rng=rng, initializer="xavier"))
    model = Sequential(layers)
    model.compile(MSE(), Adam(lr=0.003), l2=1e-3 if kind == "l2" else 0.0)
    return model


def main() -> None:
    rng = np.random.default_rng(2026)
    x_train = rng.uniform(-np.pi, np.pi, (48, 1))
    y_train = np.sin(x_train) + rng.normal(0, 0.20, x_train.shape)
    x_val = np.linspace(-np.pi, np.pi, 400).reshape(-1, 1)
    y_val = np.sin(x_val) + rng.normal(0, 0.20, x_val.shape)

    histories = {}
    summary = {"seed": 2026, "epochs": 250, "train_samples": len(x_train), "validation_samples": len(x_val), "models": {}}
    for kind in ("none", "l2", "dropout"):
        model = make_model(kind, seed=10)
        start = time.perf_counter()
        history = model.fit(
            x_train, y_train, epochs=250, batch_size=16, shuffle=True, seed=21,
            validation_data=(x_val, y_val), verbose=0,
        )
        histories[kind] = history
        train_mse = float(np.mean((model.predict(x_train) - y_train) ** 2))
        val_mse = float(np.mean((model.predict(x_val) - y_val) ** 2))
        summary["models"][kind] = {
            "train_mse": float(train_mse), "validation_mse": float(val_mse),
            "validation_minus_train_mse": float(val_mse - train_mse),
            "seconds": round(time.perf_counter() - start, 3),
            "l2": 1e-3 if kind == "l2" else 0.0,
            "dropout_rate": 0.15 if kind == "dropout" else 0.0,
        }
        print(f"{kind:>7}: train MSE={train_mse:.4f}, validation MSE={val_mse:.4f}, gap={val_mse-train_mse:+.4f}")

    ASSETS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for kind, history in histories.items():
        axes[0].plot(history["loss"], label=kind)
        axes[1].plot(history["val_loss"], label=kind)
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation objective", xlabel="Epoch", ylabel="Loss")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Day 11 — L2 regularization and inverted dropout")
    fig.tight_layout()
    fig.savefig(ASSETS / "day11_regularization.png", dpi=160)
    plt.close(fig)
    (RESULTS / "day11_regularization.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {ASSETS / 'day11_regularization.png'}")
    print(f"Saved {RESULTS / 'day11_regularization.json'}")


if __name__ == "__main__":
    main()
