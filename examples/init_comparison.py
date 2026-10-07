"""Day 9: why initialization and momentum matter.

Part 1 (instant, no download): push random data through a deep network with three
initializations and print / plot how the activation and gradient sizes change with depth.

Part 2 (MNIST, a few minutes): train the same 784-128-64-10 ReLU network with five
initialization / optimizer combinations and compare the first epochs.

Run:  python examples/init_comparison.py            # both parts
      python examples/init_comparison.py --part 1   # only the instant demo
      python examples/init_comparison.py --sweep    # learning-rate sweep for part 2 (validation set only)
Saves assets/day9_*.png and results/day9_init_comparison.json.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running as a script

from nn.activations import ReLU, Softmax, Tanh
from nn.datasets import load_mnist
from nn.diagnostics import collect_statistics, diagnose, format_statistics
from nn.initializers import he_normal, initializer_for, normal, xavier_normal
from nn.layers import Dense
from nn.losses import CategoricalCrossEntropy
from nn.metrics import accuracy
from nn.model import Sequential
from nn.optimizers import SGD, Momentum
from nn.utils import one_hot


# ------------------------------------------------------------------ builders
def build_mlp(sizes, hidden_cls, init, optimizer, seed=0):
    """``sizes`` e.g. [784, 128, 64, 10]; softmax output; cross-entropy loss.

    ``init``: "normal:0.01" (plain N(0, s^2)), "xavier", or "he". With "he", the layers that feed
    a ReLU use He and the output layer (which feeds softmax) uses Xavier, per ``initializer_for``.
    """
    layers = []
    n_dense = len(sizes) - 1
    for i in range(n_dense):
        feeds = hidden_cls if i < n_dense - 1 else Softmax
        if init.startswith("normal:"):
            fn = normal(float(init.split(":")[1]))
        elif init == "xavier":
            fn = xavier_normal
        elif init == "he":
            fn = initializer_for(feeds)  # He before ReLU, Xavier before softmax
        else:
            raise ValueError(f"unknown init {init!r}")
        layers.append(Dense(sizes[i], sizes[i + 1], seed=seed + i, initializer=fn))
        layers.append(feeds())
    model = Sequential(layers)
    model.compile(loss=CategoricalCrossEntropy(), optimizer=optimizer)
    return model


# --------------------------------------------------------------- part 1: depth
def signal_through_depth(depth=10, width=100, n=500, seed=0):
    """Returns {label: (activation_std_per_dense_layer, input_gradient_rms_per_dense_layer)}."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, width))
    y = one_hot(rng.integers(0, 10, n), 10)
    cases = {
        "ReLU, N(0, 0.01²)": (ReLU, "normal:0.01"),
        "ReLU, N(0, 1²)": (ReLU, "normal:1"),
        "ReLU, Xavier": (ReLU, "xavier"),
        "ReLU, He": (ReLU, "he"),
        "tanh, N(0, 1²)": (Tanh, "normal:1"),
        "tanh, Xavier": (Tanh, "xavier"),
    }
    out = {}
    for label, (act, init) in cases.items():
        model = build_mlp([width] * (depth + 1) + [10], act, init, SGD(0.1), seed=seed)
        stats = collect_statistics(model, x, y)
        dense = [s for s in stats if s.weight_rms is not None][:depth]
        out[label] = (
            [s.out_std for s in dense],
            [s.grad_in_rms for s in dense],
            diagnose(stats),
            format_statistics(stats),
        )
    return out


def plot_signal(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    styles = {"ReLU, N(0, 0.01²)": ("C3", "-"), "ReLU, N(0, 1²)": ("C1", "-"), "ReLU, Xavier": ("C4", "--"),
              "ReLU, He": ("C2", "-"), "tanh, N(0, 1²)": ("C1", ":"), "tanh, Xavier": ("C0", ":")}
    for label, (act_std, grad_rms, *_rest) in results.items():
        color, ls = styles[label]
        layers = np.arange(1, len(act_std) + 1)
        ax1.plot(layers, act_std, color=color, ls=ls, marker="o", ms=4, label=label)
        ax2.plot(layers, grad_rms, color=color, ls=ls, marker="o", ms=4, label=label)
    ax1.set(title="Layer output std (forward signal)", xlabel="Dense layer", ylabel="std of layer output", yscale="log")
    ax2.set(title="Gradient arriving at each layer's input (backward signal)", xlabel="Dense layer",
            ylabel="rms of dL/d(input)", yscale="log")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)


# --------------------------------------------------------------- part 2: MNIST
SIZES = [784, 128, 64, 10]
EPOCHS, BATCH = 10, 64
# Controlled comparison: the first five rows share the same *effective* step size. Momentum with
# beta = 0.9 amplifies steps up to 10x, so lr 0.01 matches SGD at lr 0.1. The last two rows are the
# best learning rates from the validation-only sweep (--sweep); again both have effective step 0.2.
CONFIGS = {
    "N(0, 0.01²) + SGD": dict(init="normal:0.01", opt="sgd", lr=0.1),
    "N(0, 0.1²) + SGD (Day 8)": dict(init="normal:0.1", opt="sgd", lr=0.1),
    "Xavier + SGD": dict(init="xavier", opt="sgd", lr=0.1),
    "He + SGD": dict(init="he", opt="sgd", lr=0.1),
    "He + Momentum": dict(init="he", opt="momentum", lr=0.01),
    "He + SGD (lr 0.2, tuned)": dict(init="he", opt="sgd", lr=0.2),
    "He + Momentum (lr 0.02, tuned)": dict(init="he", opt="momentum", lr=0.02),
}


def make_optimizer(cfg):
    return SGD(cfg["lr"]) if cfg["opt"] == "sgd" else Momentum(cfg["lr"], beta=0.9)


def train_one(cfg, data, seed, epochs=EPOCHS):
    (x_tr, y_tr), (x_va, y_va), _ = data
    model = build_mlp(SIZES, ReLU, cfg["init"], make_optimizer(cfg), seed=seed)
    history = model.fit(x_tr, one_hot(y_tr, 10), epochs=epochs, batch_size=BATCH, seed=seed,
                        validation_data=(x_va, one_hot(y_va, 10)), metrics={"accuracy": accuracy}, verbose=0)
    return model, history


def run_comparison(data, seeds, epochs=EPOCHS, configs=None, cache_dir=None, budget=None):
    """Train every configuration for every seed. Returns {name: {"config", "runs"}}.

    ``cache_dir``: each finished configuration is saved there, and reused on the next call, so an
    interrupted run can simply be started again (``--resume`` behaviour).
    ``budget``: stop starting new configurations after this many seconds (returns None if unfinished).
    """
    configs = configs or CONFIGS
    results, start = {}, time.perf_counter()
    for name, cfg in configs.items():
        cache = None
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache = Path(cache_dir) / (re.sub(r"\W+", "_", name) + f"_s{len(seeds)}_e{epochs}.json")
        if cache is not None and cache.exists():
            results[name] = json.loads(cache.read_text())
            print(f"  {name:<32} (cached)")
        else:
            if budget is not None and time.perf_counter() - start > budget:
                print("  time budget reached; run the same command again to continue.")
                return None
            runs = []
            for seed in seeds:
                _, h = train_one(cfg, data, seed, epochs)
                runs.append({k: [float(v) for v in vals] for k, vals in h.items()})
            results[name] = {"config": cfg, "runs": runs}
            if cache is not None:
                cache.write_text(json.dumps(results[name]))
        final = np.mean([r["val_accuracy"][-1] for r in results[name]["runs"]])
        print(f"  {name:<32} final val accuracy {final:.2%}  (mean of {len(seeds)} seeds)", flush=True)
    return results


def summarize(results):
    """Per-config means across seeds."""
    rows = {}
    for name, res in results.items():
        acc = np.array([r["val_accuracy"] for r in res["runs"]])
        loss = np.array([r["val_loss"] for r in res["runs"]])
        train_loss = np.array([r["loss"] for r in res["runs"]])
        hit = [next((e + 1 for e, a in enumerate(run) if a >= 0.97), None) for run in acc]
        rows[name] = {
            "val_acc_mean": acc.mean(axis=0).tolist(), "val_acc_std": acc.std(axis=0).tolist(),
            "val_loss_mean": loss.mean(axis=0).tolist(), "train_loss_mean": train_loss.mean(axis=0).tolist(),
            "epochs_to_97pct": hit,
        }
    return rows


def plot_comparison(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["C3", "C1", "C4", "C0", "C2", "C5", "C6"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for (name, s), c in zip(summary.items(), colors):
        epochs = np.arange(1, len(s["val_acc_mean"]) + 1)
        m, sd = np.array(s["val_acc_mean"]), np.array(s["val_acc_std"])
        ls = "--" if "tuned" in name else "-"
        ax1.plot(epochs, m, color=c, ls=ls, marker="o", ms=3, label=name)
        ax1.fill_between(epochs, m - sd, m + sd, color=c, alpha=0.15)
        ax2.plot(epochs, s["train_loss_mean"], color=c, ls=ls, marker="o", ms=3, label=name)
    ax1.set(title="Validation accuracy (mean ± std over seeds)", xlabel="epoch", ylabel="accuracy", ylim=(0.8, 0.99))
    ax1.legend(loc="lower right", fontsize=8)
    ax2.set(title="Training loss (mean over seeds)", xlabel="epoch", ylabel="loss", yscale="log")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)


def sweep(data, epochs=EPOCHS, seed=0):
    """Validation-only learning-rate sweep for the He configurations."""
    print("lr sweep (validation set, seed 0; score = mean val accuracy of the last 3 epochs)\n")
    grid = [("sgd", lr) for lr in (0.05, 0.1, 0.2)] + [("momentum", lr) for lr in (0.005, 0.01, 0.02, 0.05)]
    for opt, lr in grid:
        _, h = train_one(dict(init="he", opt=opt, lr=lr), data, seed, epochs)
        va = h["val_accuracy"]
        print(f"  He + {opt:<8} lr={lr:<6} score {np.mean(va[-3:]):.4f}   epoch1 {va[0]:.4f}  epoch3 {va[2]:.4f}  final {va[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, choices=(1, 2), help="run only one part")
    parser.add_argument("--sweep", action="store_true", help="run the learning-rate sweep and exit")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--budget", type=float, help="stop starting new configurations after this many seconds (resumable)")
    args = parser.parse_args()

    if args.sweep:
        sweep(load_mnist(data_dir=ROOT / "data"), args.epochs)
        return

    if args.part in (None, 1):
        print("Part 1: signal size through a 10-layer network (500 random samples)\n")
        results = signal_through_depth()
        for label, (act_std, grad_rms, warnings, _) in results.items():
            print(f"  {label:<20} output std: layer 1 {act_std[0]:.2e} -> layer 10 {act_std[-1]:.2e}   "
                  f"| gradient at layer 1 input / layer 10 input: {grad_rms[0] / grad_rms[-1]:.1e}")
            for w in warnings:
                print(f"      WARNING: {w}")
        plot_signal(results, ROOT / "assets" / "day9_signal_through_depth.png")
        print("\nSaved assets/day9_signal_through_depth.png\n")

    if args.part in (None, 2):
        print(f"Part 2: MNIST, {args.seeds} seeds x {args.epochs} epochs per configuration\n")
        data = load_mnist(data_dir=ROOT / "data")
        raw = run_comparison(data, seeds=list(range(args.seeds)), epochs=args.epochs,
                             cache_dir=ROOT / "data" / "day9_cache", budget=args.budget)
        if raw is None:
            return
        summary = summarize(raw)
        out = ROOT / "results" / "day9_init_comparison.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"epochs": args.epochs, "seeds": list(range(args.seeds)), "batch_size": BATCH,
                                   "architecture": SIZES, "summary": summary}, indent=2))
        plot_comparison(summary, ROOT / "assets" / "day9_mnist_init_comparison.png")
        print("\nSaved results/day9_init_comparison.json and assets/day9_mnist_init_comparison.png")


if __name__ == "__main__":
    main()
