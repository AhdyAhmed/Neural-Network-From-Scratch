"""Day 8: handwritten digit classification on MNIST (baseline).

Network:   784 -> 128 -> 64 -> 10   (ReLU, ReLU, Softmax)
Training:  mini-batch SGD, batch size 64, lr 0.1, 15 epochs, cross-entropy loss
Data:      50,000 train / 10,000 validation / 10,000 test (the standard split)

The test set is used exactly once, at the very end. Every choice (learning
rate, initial weight scale) was made on the validation set.

Run:  python examples/mnist.py
Needs internet the first time (downloads ~17 MB to data/, which is git-ignored).
Writes results/day8_mnist_baseline.json and assets/day8_mnist_baseline.png.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running as a script

from nn.activations import ReLU, Softmax
from nn.datasets import load_mnist
from nn.layers import Dense
from nn.losses import CategoricalCrossEntropy
from nn.metrics import accuracy
from nn.model import Sequential
from nn.optimizers import SGD
from nn.utils import one_hot

CONFIG = {
    "architecture": "784-128-64-10 (ReLU, ReLU, Softmax)",
    "optimizer": "SGD",
    "learning_rate": 0.1,
    "batch_size": 64,
    "epochs": 15,
    "init_scale": 0.1,
    "seed": 0,
}


def build_model(config=CONFIG):
    s, seed = config["init_scale"], config["seed"]
    model = Sequential([
        Dense(784, 128, seed=seed, init_scale=s), ReLU(),
        Dense(128, 64, seed=seed + 1, init_scale=s), ReLU(),
        Dense(64, 10, seed=seed + 2, init_scale=s), Softmax(),
    ])
    model.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=config["learning_rate"]))
    return model


def train(data, config=CONFIG, verbose=1):
    """Train on the train split, monitor the validation split. Returns (model, history)."""
    (x_train, y_train), (x_val, y_val), _ = data
    model = build_model(config)
    history = model.fit(
        x_train, one_hot(y_train, 10),
        epochs=config["epochs"], batch_size=config["batch_size"], seed=config["seed"],
        validation_data=(x_val, one_hot(y_val, 10)),
        metrics={"accuracy": accuracy}, verbose=verbose,
    )
    return model, history


def make_plot(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.axhline(np.log(10), color="gray", ls="--", label="chance (ln 10)")
    ax1.set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss", yscale="log")
    ax1.legend()

    ax2.plot(epochs, history["accuracy"], label="train (running avg.)")
    ax2.plot(epochs, history["val_accuracy"], label="validation")
    ax2.set(title="Accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0.85, 1.0))
    ax2.legend(loc="lower right")

    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)


def main():
    data = load_mnist(data_dir=ROOT / "data")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    print(f"train {x_train.shape}, validation {x_val.shape}, test {x_test.shape}")
    print(f"pixel range [{x_train.min():.0f}, {x_train.max():.0f}], labels {np.unique(y_train)}\n")

    model = build_model()
    print(model.summary(), "\n")
    start = time.perf_counter()
    model, history = train(data)
    seconds = time.perf_counter() - start

    # The one and only look at the test set.
    test_acc = accuracy(model.predict(x_test), y_test)
    test_loss = model.evaluate(x_test, one_hot(y_test, 10))
    print(f"\nFinal validation accuracy: {history['val_accuracy'][-1]:.2%}")
    print(f"Final TEST accuracy:       {test_acc:.2%}   (test loss {test_loss:.4f})")
    print(f"Training time: {seconds:.1f}s")

    results = {
        "config": CONFIG,
        "final": {
            "val_accuracy": history["val_accuracy"][-1],
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "train_seconds": round(seconds, 1),
        },
        "history": {k: [round(float(v), 6) for v in vals] for k, vals in history.items()},
    }
    out_json = ROOT / "results" / "day8_mnist_baseline.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_json.relative_to(ROOT)}")

    try:
        out_png = ROOT / "assets" / "day8_mnist_baseline.png"
        make_plot(history, out_png)
        print(f"Saved {out_png.relative_to(ROOT)}")
    except ImportError:
        print("matplotlib not installed - skipping plot.")


if __name__ == "__main__":
    main()
