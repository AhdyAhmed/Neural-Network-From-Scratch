"""Day 13: reproducible MNIST experiment and controlled ablations.

Run the full experiment (downloads verified MNIST on first use):
    python examples/day13_experiments.py

Run a quick smoke experiment on a stratified-size subset:
    python examples/day13_experiments.py --quick

Run only the controlled ablations:
    python examples/day13_experiments.py --ablation

All random seeds, model settings, split sizes, timings, and measured metrics are
written to results/day13_experiments.json. The test split is evaluated only by
the final configuration, never to choose ablation settings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn.activations import ReLU, Softmax
from nn.datasets import load_mnist
from nn.initializers import get_initializer
from nn.layers import Dense, Dropout
from nn.losses import CategoricalCrossEntropy
from nn.metrics import accuracy, confusion_matrix
from nn.model import Sequential
from nn.optimizers import Adam
from nn.utils import one_hot

SEED = 2026
FINAL_CONFIG: dict[str, Any] = {
    "name": "final_adam_he_dropout",
    "architecture": [784, 128, 64, 10],
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 15,
    "initializer": "he",
    "dropout_rate": 0.10,
    "l2": 0.0,
    "seed": SEED,
    "shuffle_seed": SEED + 1,
    "early_stopping": True,
    "patience": 3,
    "min_delta": 1e-4,
}


def build_model(config: dict[str, Any]) -> Sequential:
    """Create a model from a JSON-serializable config without hidden randomness."""
    seed = int(config["seed"])
    initializer = config["initializer"]
    dropout_rate = float(config.get("dropout_rate", 0.0))
    layers = [Dense(784, 128, seed=seed, initializer=initializer), ReLU()]
    if dropout_rate:
        layers.append(Dropout(dropout_rate, seed=seed + 10))
    layers.extend([Dense(128, 64, seed=seed + 1, initializer=initializer), ReLU()])
    if dropout_rate:
        layers.append(Dropout(dropout_rate, seed=seed + 11))
    layers.extend([Dense(64, 10, seed=seed + 2, initializer="xavier"), Softmax()])
    model = Sequential(layers)
    model.compile(
        loss=CategoricalCrossEntropy(),
        optimizer=Adam(lr=float(config["learning_rate"])),
        l2=float(config.get("l2", 0.0)),
    )
    return model


def _subset(x: np.ndarray, y: np.ndarray, limit: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    # Shuffle deterministically, then select; this avoids depending on source ordering.
    indices = rng.permutation(len(y))[:limit]
    return x[indices], y[indices]


def run_one(data, config: dict[str, Any], *, verbose: int = 1) -> tuple[dict[str, Any], dict[str, list[float]], Sequential]:
    (x_train, y_train), (x_val, y_val), _ = data
    model = build_model(config)
    start = time.perf_counter()
    history = model.fit(
        x_train, one_hot(y_train, 10), epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]), shuffle=True,
        seed=int(config["shuffle_seed"]),
        validation_data=(x_val, one_hot(y_val, 10)), metrics={"accuracy": accuracy},
        verbose=verbose,
        early_stopping=bool(config.get("early_stopping", False)),
        patience=int(config.get("patience", 5)),
        min_delta=float(config.get("min_delta", 0.0)),
        restore_best_weights=True,
    )
    elapsed = time.perf_counter() - start
    measured = {
        "name": config["name"], "config": config,
        "epochs_completed": len(history["loss"]),
        "best_validation_loss": float(min(history["val_loss"])),
        "final_recorded_validation_accuracy": float(history["val_accuracy"][-1]),
        "training_seconds": round(elapsed, 3),
        "history": {k: [round(float(v), 7) for v in values] for k, values in history.items()},
    }
    return measured, history, model


def make_ablation_configs(epochs: int) -> list[dict[str, Any]]:
    """One-factor-at-a-time ablations around the final configuration."""
    variants = [
        ("baseline_no_dropout", "he", 0.0, 0.001),
        ("dropout_0.10", "he", 0.10, 0.001),
        ("xavier_no_dropout", "xavier", 0.0, 0.001),
        ("learning_rate_0.0003", "he", 0.0, 0.0003),
    ]
    return [{
        "name": name, "architecture": [784, 128, 64, 10], "optimizer": "Adam",
        "learning_rate": lr, "batch_size": 64, "epochs": epochs,
        "initializer": init, "dropout_rate": dropout, "l2": 0.0,
        "seed": SEED, "shuffle_seed": SEED + 1,
        "early_stopping": True, "patience": 3, "min_delta": 1e-4,
    } for name, init, dropout, lr in variants]


def save_history_plot(final: dict[str, Any], ablations: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    final_history = final.get("history", {})
    axes[0].plot(final_history.get("loss", []), label="train")
    axes[0].plot(final_history.get("val_loss", []), label="validation")
    axes[0].set(title="Final model: cross-entropy", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(final_history.get("val_accuracy", []), label="final config")
    for result in ablations:
        vals = result.get("history", {}).get("val_accuracy", [])
        if vals:
            axes[1].plot(vals, label=result["name"])
    axes[1].set(title="Validation accuracy: ablation", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Day 13 — Reproducible MNIST experiments")
    fig.tight_layout()
    out = ROOT / "assets" / "day13_experiments.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use small train/validation/test subsets and 2 epochs")
    parser.add_argument("--ablation", action="store_true", help="run only ablations; do not evaluate test split")
    parser.add_argument("--no-ablation", action="store_true", help="run final model only")
    parser.add_argument("--train-limit", type=int, default=None, help="optional maximum train examples")
    parser.add_argument("--val-limit", type=int, default=None, help="optional maximum validation examples")
    parser.add_argument("--test-limit", type=int, default=None, help="optional maximum test examples")
    args = parser.parse_args()
    if args.quick:
        args.train_limit = args.train_limit or 3000
        args.val_limit = args.val_limit or 1000
        args.test_limit = args.test_limit or 1000
    data = load_mnist(data_dir=ROOT / "data")
    limits = (args.train_limit, args.val_limit, args.test_limit)
    data = tuple(_subset(x, y, limit, SEED + i) for i, ((x, y), limit) in enumerate(zip(data, limits)))
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    print(f"Data sizes: train={len(y_train)}, validation={len(y_val)}, test={len(y_test)}")

    final_config = dict(FINAL_CONFIG)
    if args.quick:
        final_config["epochs"] = 2
        final_config["patience"] = 2
    final_result: dict[str, Any] | None = None
    ablation_results: list[dict[str, Any]] = []

    if not args.ablation:
        final_result, _, model = run_one(data, final_config)
        if not args.quick:
            # The held-out test set is used only for the predeclared final configuration.
            test_pred = model.predict(x_test)
            final_result["test_accuracy"] = accuracy(test_pred, y_test)
            final_result["test_loss"] = float(model.evaluate(x_test, one_hot(y_test, 10)))
            final_result["test_confusion_matrix"] = confusion_matrix(y_test, test_pred, num_classes=10).tolist()
        else:
            final_result["test_evaluation"] = "omitted in --quick mode; use the full run for the held-out test metric"

    if not args.no_ablation:
        ablation_epochs = 2 if args.quick else 8
        for config in make_ablation_configs(ablation_epochs):
            result, _, _ = run_one(data, config, verbose=0)
            # Keep ablation results compact; the final model retains the full epoch history.
            ablation_results.append(result)
            print(f"{config['name']}: val_accuracy={result['final_recorded_validation_accuracy']:.4f}, "
                  f"best_val_loss={result['best_validation_loss']:.4f}")

    payload = {
        "experiment": "Day 13 — reproducible MNIST experiments",
        "seed": SEED,
        "dataset": "MNIST (verified loader; canonical train/validation/test split)",
        "split_sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "quick_mode": bool(args.quick),
        "final_experiment": final_result,
        "ablations": ablation_results,
        "notes": [
            "All model initialization and minibatch shuffle seeds are recorded in each config.",
            "Ablations use validation data only; test data is reserved for the final configuration.",
            "Quick mode uses deterministic subsets and does not report a held-out test metric.",
            "Training times vary by machine and are informational, not deterministic outputs.",
        ],
    }
    results_path = ROOT / "results" / "day13_experiments.json"
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_history_plot(final_result or {}, ablation_results)
    print(f"Saved {results_path.relative_to(ROOT)}")
    print("Saved assets/day13_experiments.png")
    if final_result and "test_accuracy" in final_result:
        print(f"Final held-out test accuracy: {final_result['test_accuracy']:.2%}")


if __name__ == "__main__":
    main()
