"""Day 12: early stopping, classification diagnostics, and visualizations.

Quick synthetic demo (no download): python examples/day12_analysis.py
MNIST error analysis (downloads MNIST on first run): python examples/day12_analysis.py --mnist
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nn.activations import ReLU, Softmax, Tanh
from nn.datasets import load_mnist
from nn.layers import Dense
from nn.losses import CategoricalCrossEntropy
from nn.metrics import accuracy, confusion_matrix, per_class_accuracy
from nn.model import Sequential
from nn.optimizers import Adam, SGD
from nn.utils import one_hot
from nn.visualization import plot_confusion_matrix, plot_decision_boundary, plot_misclassified_digits


def make_blobs(n_per_class: int = 100, seed: int = 12):
    rng = np.random.default_rng(seed)
    centers = np.array([[-1.8, -1.2], [1.8, -1.2], [0.0, 1.8]])
    x = np.concatenate([center + 0.75 * rng.normal(size=(n_per_class, 2)) for center in centers])
    y = np.repeat(np.arange(3), n_per_class)
    order = rng.permutation(len(x))
    return x[order], y[order]


def run_blobs():
    rng = np.random.default_rng(12)
    x, labels = make_blobs()
    x_train, y_train = x[:240], labels[:240]
    x_val, y_val = x[240:], labels[240:]
    model = Sequential([Dense(2, 16, seed=12), Tanh(), Dense(16, 3, seed=13), Softmax()])
    model.compile(CategoricalCrossEntropy(), Adam(lr=0.03))
    started = time.perf_counter()
    history = model.fit(
        x_train, one_hot(y_train, 3), epochs=250, batch_size=32, seed=12,
        validation_data=(x_val, one_hot(y_val, 3)), metrics={"accuracy": accuracy},
        early_stopping=True, patience=20, min_delta=1e-4, restore_best_weights=True,
        verbose=1, log_every=1,
    )
    seconds = time.perf_counter() - started
    predictions = model.predict(x_val)
    predicted_labels = predictions.argmax(axis=1)
    matrix = confusion_matrix(y_val, predictions, num_classes=3)
    per_class = per_class_accuracy(y_val, predictions, num_classes=3)
    plot_decision_boundary(model, x, labels, ROOT / "assets/day12_decision_boundary.png")
    plot_confusion_matrix(matrix, ROOT / "assets/day12_confusion_matrix.png", class_names=["class 0", "class 1", "class 2"])
    save_history_plot(history, ROOT / "assets/day12_training_history.png")
    result = {
        "config": {"dataset": "synthetic 3-class blobs", "seed": 12, "max_epochs": 250,
                   "patience": 20, "min_delta": 1e-4, "optimizer": "Adam", "learning_rate": 0.03},
        "epochs_ran": len(history["loss"]), "training_seconds": round(seconds, 3),
        "best_validation_loss": float(min(history["val_loss"])),
        "validation_accuracy": float(np.mean(predicted_labels == y_val)),
        "confusion_matrix": matrix.tolist(),
        "per_class_accuracy": [None if np.isnan(v) else float(v) for v in per_class],
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    (ROOT / "results/day12_analysis.json").write_text(json.dumps(result, indent=2))
    print(f"Validation accuracy: {result['validation_accuracy']:.2%}; trained {len(history['loss'])} epochs in {seconds:.2f}s")
    print("Saved assets/day12_decision_boundary.png, assets/day12_confusion_matrix.png, assets/day12_training_history.png")


def save_history_plot(history, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, history["accuracy"], label="train (batch average)")
    axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_mnist():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(ROOT / "data")
    # A smaller subset keeps this demonstration practical on a CPU.
    train_n, val_n = 10000, 2000
    model = Sequential([Dense(784, 128, seed=21), ReLU(), Dense(128, 10, seed=22), Softmax()])
    model.compile(CategoricalCrossEntropy(), SGD(lr=0.08))
    history = model.fit(
        x_train[:train_n], one_hot(y_train[:train_n], 10), epochs=20, batch_size=64, seed=21,
        validation_data=(x_val[:val_n], one_hot(y_val[:val_n], 10)), metrics={"accuracy": accuracy},
        early_stopping=True, patience=3, restore_best_weights=True, verbose=1, log_every=1,
    )
    # Test data is evaluated only after training/model selection is complete.
    probabilities = model.predict(x_test)
    matrix = confusion_matrix(y_test, probabilities, num_classes=10)
    per_class = per_class_accuracy(y_test, probabilities, num_classes=10)
    count = plot_misclassified_digits(x_test, y_test, probabilities, ROOT / "assets/day12_mnist_misclassified.png")
    plot_confusion_matrix(matrix, ROOT / "assets/day12_mnist_confusion_matrix.png", class_names=list(range(10)), normalize=True)
    result = {"config": {"dataset": "MNIST", "train_subset": train_n, "validation_subset": val_n,
                         "max_epochs": 20, "patience": 3, "seed": 21},
              "epochs_ran": len(history["loss"]), "test_accuracy": accuracy(probabilities, y_test),
              "test_confusion_matrix": matrix.tolist(),
              "test_per_class_accuracy": [None if np.isnan(v) else float(v) for v in per_class],
              "misclassified_examples_plotted": count,
              "history": {k: [float(v) for v in vals] for k, vals in history.items()}}
    (ROOT / "results/day12_mnist_analysis.json").write_text(json.dumps(result, indent=2))
    print(f"MNIST test accuracy: {result['test_accuracy']:.2%}; plotted {count} misclassified digits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist", action="store_true", help="also train on MNIST and plot misclassified digits (may download data)")
    args = parser.parse_args()
    run_blobs()
    if args.mnist:
        run_mnist()


if __name__ == "__main__":
    main()
