"""Day 7: 3-class classification with Softmax, cross-entropy, and mini-batches.

Three Gaussian blobs in 2D; a 2 -> 16 -> 3 network (tanh hidden layer, softmax
output) learns to tell them apart. Trained with mini-batch SGD.

Run:  python examples/three_class_blobs.py
Saves assets/day7_three_class.png (needs matplotlib).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow running as a script

from nn.activations import Softmax, Tanh
from nn.layers import Dense
from nn.losses import CategoricalCrossEntropy
from nn.model import Sequential
from nn.optimizers import SGD
from nn.utils import one_hot

CENTERS = np.array([[0.0, 2.0], [-2.0, -1.0], [2.0, -1.0]])
NUM_CLASSES = 3


def make_blobs(n_per_class=100, spread=0.8, seed=0):
    """Returns (x, labels) with labels in {0, 1, 2}, shuffled."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([c + spread * rng.normal(size=(n_per_class, 2)) for c in CENTERS])
    labels = np.repeat(np.arange(NUM_CLASSES), n_per_class)
    order = rng.permutation(len(x))
    return x[order], labels[order]


def build_model(seed=0, lr=0.1, hidden=16, init_scale=0.5):
    model = Sequential([
        Dense(2, hidden, seed=seed, init_scale=init_scale), Tanh(),
        Dense(hidden, NUM_CLASSES, seed=seed + 1, init_scale=init_scale), Softmax(),
    ])
    model.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=lr))
    return model


def accuracy(model, x, labels):
    return float(np.mean(model.predict(x).argmax(axis=1) == labels))


def train(seed=0, epochs=60, batch_size=32, lr=0.1, verbose=0):
    """Train on 240 points, hold out 60. Returns (model, history, data)."""
    x, labels = make_blobs(seed=seed)
    x_tr, l_tr, x_te, l_te = x[:240], labels[:240], x[240:], labels[240:]
    y_tr, y_te = one_hot(l_tr, NUM_CLASSES), one_hot(l_te, NUM_CLASSES)

    model = build_model(seed=seed, lr=lr)
    history = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, seed=seed,
                        validation_data=(x_te, y_te), verbose=verbose)
    return model, history, (x_tr, l_tr, x_te, l_te)


def make_plot(model, history, data, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_tr, l_tr, x_te, l_te = data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history["loss"], label="train")
    ax1.plot(history["val_loss"], label="test")
    ax1.axhline(np.log(NUM_CLASSES), color="gray", ls="--", label=f"chance (ln {NUM_CLASSES} = {np.log(NUM_CLASSES):.2f})")
    ax1.set(title="Cross-entropy loss (mini-batch SGD)", xlabel="epoch", ylabel="loss", yscale="log")
    ax1.legend()

    g = np.linspace(-5, 5, 300)
    xx, yy = np.meshgrid(g, g)
    region = model.predict(np.c_[xx.ravel(), yy.ravel()]).argmax(axis=1).reshape(xx.shape)
    ax2.contourf(xx, yy, region, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#cfe2f3", "#f4cccc", "#d9ead3"])
    colors = np.array(["C0", "C3", "C2"])
    ax2.scatter(x_tr[:, 0], x_tr[:, 1], c=colors[l_tr], s=14, alpha=0.7, label="train")
    ax2.scatter(x_te[:, 0], x_te[:, 1], c=colors[l_te], s=40, marker="s", edgecolors="k", label="test")
    ax2.set(title=f"Decision regions (test accuracy {accuracy(model, x_te, l_te):.0%})",
            xlabel="x1", ylabel="x2", xlim=(-5, 5), ylim=(-5, 5))
    ax2.legend(loc="upper right")

    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)


def main():
    # Sanity check from the roadmap: an untrained K-class softmax model has loss ~ ln(K).
    # (Needs near-zero initial weights so the outputs start near-uniform.)
    x, labels = make_blobs()
    y = one_hot(labels, NUM_CLASSES)
    untrained = build_model(init_scale=0.01)
    print(f"Initial loss, tiny weights:   {untrained.evaluate(x, y):.3f}   (ln 3 = {np.log(3):.3f})")
    print(f"Initial loss, std-0.5 weights: {build_model().evaluate(x, y):.3f}   (confidently wrong -> higher)")
    print(untrained.summary(), "\n")

    model, history, data = train(verbose=1)
    x_tr, l_tr, x_te, l_te = data
    print(f"\nTrain accuracy: {accuracy(model, x_tr, l_tr):.1%}")
    print(f"Test accuracy:  {accuracy(model, x_te, l_te):.1%}")
    probs = model.predict(x_te[:3])
    print("\nFirst 3 test predictions (rows sum to 1):")
    print(np.round(probs, 3), "  true:", l_te[:3])

    try:
        out = ROOT / "assets" / "day7_three_class.png"
        make_plot(model, history, data, out)
        print(f"Saved plot to {out.relative_to(ROOT)}")
    except ImportError:
        print("matplotlib not installed - skipping plot.")


if __name__ == "__main__":
    main()
