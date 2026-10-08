"""Day 10: SGD vs Momentum vs RMSProp vs Adam.

Part 1 (instant): paths of the four optimizers on an ill-conditioned 2D quadratic.
Part 2 (MNIST, ~5 minutes, resumable): the same 784-128-64-10 He-initialised ReLU network trained
with each optimizer at its best learning rate (picked on the validation set with --sweep).

Run:  python examples/optimizer_comparison.py --part 1
      python examples/optimizer_comparison.py --sweep      # lr sweep for RMSProp and Adam (validation only)
      python examples/optimizer_comparison.py --part 2     # 4 optimizers x 3 seeds x 10 epochs
Saves assets/day10_*.png and results/day10_optimizer_comparison.json.
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

from examples.init_comparison import BATCH, EPOCHS, SIZES, build_mlp
from nn.activations import ReLU
from nn.datasets import load_mnist
from nn.metrics import accuracy
from nn.optimizers import SGD, Adam, Momentum, RMSProp
from nn.utils import one_hot

# ------------------------------------------------------------ part 1: quadratic
CURVATURE = np.array([1.0, 25.0])
MINIMUM = np.array([3.0, -2.0])
START = np.array([-4.0, 4.0])
QUAD_SETTINGS = {
    "SGD (lr 0.03)": lambda: SGD(0.03),
    "Momentum (lr 0.01, β 0.9)": lambda: Momentum(0.01, 0.9),
    "RMSProp (lr 0.05)": lambda: RMSProp(0.05),
    "Adam (lr 0.2)": lambda: Adam(0.2),
}


def quadratic_path(opt, steps=300):
    """Positions visited by ``opt`` on f(w) = 0.5 * sum(a_i (w_i - c_i)^2). Returns an array (steps+1, 2)."""
    w = START.copy()
    path = [w.copy()]
    for _ in range(steps):
        opt.step([(w, CURVATURE * (w - MINIMUM))])
        path.append(w.copy())
    return np.array(path)


def steps_to_reach(path, tol):
    dist = np.linalg.norm(path - MINIMUM, axis=1)
    hit = np.nonzero(dist < tol)[0]
    return int(hit[0]) if len(hit) else None


def plot_quadratic(paths, path_out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    g0, g1 = np.meshgrid(np.linspace(-6, 7, 300), np.linspace(-9, 6, 300))
    f = 0.5 * (CURVATURE[0] * (g0 - MINIMUM[0]) ** 2 + CURVATURE[1] * (g1 - MINIMUM[1]) ** 2)
    ax1.contour(g0, g1, f, levels=np.geomspace(0.5, 400, 14), colors="lightgray", linewidths=0.8)
    for (name, p), c in zip(paths.items(), ["C0", "C1", "C3", "C2"]):
        ax1.plot(p[:, 0], p[:, 1], "-o", ms=2.5, lw=1.2, color=c, label=name)
        dist = np.linalg.norm(p - MINIMUM, axis=1)
        ax2.plot(dist, color=c, label=name)
    ax1.plot(*MINIMUM, "k*", ms=14)
    ax1.plot(*START, "ks", ms=7)
    ax1.set(title="Paths on f = ½(w₀² + 25w₁²) (shifted)", xlabel="w₀", ylabel="w₁", xlim=(-6, 7), ylim=(-9, 6))
    ax1.legend(fontsize=8, loc="upper right")
    ax2.set(title="Distance to the minimum", xlabel="step", ylabel="‖w − w*‖", yscale="log", ylim=(1e-9, 20))
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path_out.parent.mkdir(exist_ok=True)
    fig.savefig(path_out, dpi=120)


# --------------------------------------------------------------- part 2: MNIST
# Learning rates: SGD and Momentum come from the Day 9 validation sweep; RMSProp and Adam from --sweep.
# SGD 0.2 and Momentum 0.02 have the same effective step (0.02 / (1 - 0.9) = 0.2).
TUNED = {
    "SGD": dict(opt="sgd", lr=0.2),
    "Momentum": dict(opt="momentum", lr=0.02),
    "RMSProp": dict(opt="rmsprop", lr=0.001),
    "Adam": dict(opt="adam", lr=0.001),
}


def make_optimizer(cfg):
    return {"sgd": lambda: SGD(cfg["lr"]), "momentum": lambda: Momentum(cfg["lr"], 0.9),
            "rmsprop": lambda: RMSProp(cfg["lr"]), "adam": lambda: Adam(cfg["lr"])}[cfg["opt"]]()


def train_one(cfg, data, seed, epochs=EPOCHS):
    (x_tr, y_tr), (x_va, y_va), _ = data
    model = build_mlp(SIZES, ReLU, "he", make_optimizer(cfg), seed=seed)
    start = time.perf_counter()
    h = model.fit(x_tr, one_hot(y_tr, 10), epochs=epochs, batch_size=BATCH, seed=seed,
                  validation_data=(x_va, one_hot(y_va, 10)), metrics={"accuracy": accuracy}, verbose=0)
    h = {k: [float(v) for v in vals] for k, vals in h.items()}
    h["seconds_per_epoch"] = (time.perf_counter() - start) / epochs
    return h


def sweep(data, epochs=EPOCHS, seed=0):
    print("lr sweep (validation set, seed 0; score = mean val accuracy of the last 3 epochs)\n")
    grid = [("rmsprop", lr) for lr in (1e-4, 3e-4, 1e-3, 3e-3)] + [("adam", lr) for lr in (3e-4, 1e-3, 3e-3, 1e-2)]
    for opt, lr in grid:
        h = train_one(dict(opt=opt, lr=lr), data, seed, epochs)
        va = h["val_accuracy"]
        print(f"  {opt:<8} lr={lr:<7g} score {np.mean(va[-3:]):.4f}   epoch1 {va[0]:.4f}  epoch3 {va[2]:.4f}  final {va[-1]:.4f}")


def run_comparison(data, seeds, epochs, cache_dir, budget=None):
    results, start = {}, time.perf_counter()
    for name, cfg in TUNED.items():
        cache = Path(cache_dir) / (re.sub(r"\W+", "_", name) + f"_lr{cfg['lr']}_s{len(seeds)}_e{epochs}.json")
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            runs, tag = json.loads(cache.read_text()), " (cached)"
        else:
            if budget is not None and time.perf_counter() - start > budget:
                print("  time budget reached; run the same command again to continue.")
                return None
            runs, tag = [train_one(cfg, data, s, epochs) for s in seeds], ""
            cache.write_text(json.dumps(runs))
        results[name] = {"config": cfg, "runs": runs}
        print(f"  {name:<10} lr={cfg['lr']:<6g} final val accuracy {np.mean([r['val_accuracy'][-1] for r in runs]):.2%}{tag}", flush=True)
    return results


def summarize(results):
    out = {}
    for name, res in results.items():
        runs = res["runs"]
        stack = lambda key: np.array([r[key] for r in runs])
        out[name] = {
            "config": res["config"],
            "val_acc_mean": stack("val_accuracy").mean(0).tolist(), "val_acc_std": stack("val_accuracy").std(0).tolist(),
            "val_loss_mean": stack("val_loss").mean(0).tolist(), "train_loss_mean": stack("loss").mean(0).tolist(),
            "epochs_to_97pct": [next((e + 1 for e, a in enumerate(r["val_accuracy"]) if a >= 0.97), None) for r in runs],
            "seconds_per_epoch": float(np.mean([r["seconds_per_epoch"] for r in runs])),
        }
    return out


def plot_comparison(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for (name, s), c in zip(summary.items(), ["C0", "C1", "C3", "C2"]):
        label = f"{name} (lr {s['config']['lr']:g})"
        epochs = np.arange(1, len(s["val_acc_mean"]) + 1)
        m, sd = np.array(s["val_acc_mean"]), np.array(s["val_acc_std"])
        axes[0].plot(epochs, m, "-o", ms=3, color=c, label=label)
        axes[0].fill_between(epochs, m - sd, m + sd, color=c, alpha=0.15)
        axes[1].plot(epochs, s["train_loss_mean"], "-o", ms=3, color=c, label=label)
        axes[2].plot(epochs, s["val_loss_mean"], "-o", ms=3, color=c, label=label)
    axes[0].set(title="Validation accuracy (mean ± std, 3 seeds)", xlabel="epoch", ylim=(0.88, 0.985))
    axes[1].set(title="Training loss", xlabel="epoch", yscale="log")
    axes[2].set(title="Validation loss", xlabel="epoch", yscale="log")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, choices=(1, 2))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--budget", type=float, help="stop starting new configurations after this many seconds (resumable)")
    args = parser.parse_args()

    if args.sweep:
        sweep(load_mnist(data_dir=ROOT / "data"), args.epochs)
        return

    if args.part in (None, 1):
        print("Part 1: ill-conditioned quadratic f = 0.5 * (1 * x^2 + 25 * y^2)\n")
        paths = {name: quadratic_path(make()) for name, make in QUAD_SETTINGS.items()}
        for name, p in paths.items():
            hit = steps_to_reach(p, 1e-3)
            print(f"  {name:<28} distance after 300 steps: {np.linalg.norm(p[-1] - MINIMUM):.2e}   "
                  f"steps to get within 1e-3: {hit if hit is not None else 'not reached'}")
        plot_quadratic(paths, ROOT / "assets" / "day10_quadratic.png")
        print("\nSaved assets/day10_quadratic.png\n")

    if args.part in (None, 2):
        print(f"Part 2: MNIST, {args.seeds} seeds x {args.epochs} epochs per optimizer\n")
        raw = run_comparison(load_mnist(data_dir=ROOT / "data"), list(range(args.seeds)), args.epochs,
                             ROOT / "data" / "day10_cache", args.budget)
        if raw is None:
            return
        summary = summarize(raw)
        out = ROOT / "results" / "day10_optimizer_comparison.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"epochs": args.epochs, "seeds": list(range(args.seeds)), "batch_size": BATCH,
                                   "architecture": SIZES, "init": "he", "summary": summary}, indent=2))
        plot_comparison(summary, ROOT / "assets" / "day10_optimizer_comparison.png")
        print("\nSaved results/day10_optimizer_comparison.json and assets/day10_optimizer_comparison.png")


if __name__ == "__main__":
    main()
