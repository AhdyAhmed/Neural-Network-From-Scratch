"""Plotting helpers for inspecting classifier behavior (matplotlib is optional)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_confusion_matrix(matrix: np.ndarray, path: str | Path, class_names=None, normalize: bool = False) -> None:
    """Save a labeled confusion-matrix heatmap to ``path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square 2D array.")
    shown = matrix.astype(float)
    if normalize:
        totals = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, totals, out=np.zeros_like(shown), where=totals != 0)
    names = [str(i) for i in range(matrix.shape[0])] if class_names is None else list(class_names)
    if len(names) != matrix.shape[0]:
        raise ValueError("class_names length must match matrix dimensions.")
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.65), max(4, len(names) * 0.55)))
    image = ax.imshow(shown, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
           xticklabels=names, yticklabels=names, xlabel="Predicted label", ylabel="True label",
           title="Normalized confusion matrix" if normalize else "Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = shown.max() / 2 if shown.size else 0
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            label = f"{shown[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}"
            ax.text(j, i, label, ha="center", va="center", color="white" if shown[i, j] > threshold else "black", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_misclassified_digits(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                              path: str | Path, max_images: int = 25) -> int:
    """Plot misclassified flattened 28x28 images; return the number plotted.

    ``y_pred`` can be predicted integer labels or an ``(N, classes)`` score matrix.
    ``images`` can be ``(N, 784)`` or ``(N, 28, 28)``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images, y_true, y_pred = np.asarray(images), np.asarray(y_true), np.asarray(y_pred)
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("images must have shape (N, 784) or (N, 28, 28).")
    if y_true.ndim != 1 or len(y_true) != len(images):
        raise ValueError("y_true must be a 1D label array matching images.")
    if y_pred.ndim == 2:
        if len(y_pred) != len(images):
            raise ValueError("y_pred must match the number of images.")
        predicted = y_pred.argmax(axis=1)
    elif y_pred.ndim == 1 and len(y_pred) == len(images):
        predicted = y_pred.astype(int)
    else:
        raise ValueError("y_pred must be labels or a score matrix matching images.")
    if max_images <= 0:
        raise ValueError("max_images must be positive.")
    indices = np.flatnonzero(predicted != y_true)[:max_images]
    if len(indices) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassified examples", ha="center", va="center")
        ax.axis("off")
    else:
        cols = min(5, len(indices))
        rows = int(np.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, idx in zip(axes.flat, indices):
            ax.imshow(images[idx], cmap="gray")
            ax.set_title(f"true: {y_true[idx]} | pred: {predicted[idx]}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return len(indices)


def plot_decision_boundary(model, x: np.ndarray, y: np.ndarray, path: str | Path,
                           resolution: int = 250, padding: float = 0.5) -> None:
    """Plot a trained 2D classifier's decision regions and labeled samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) != len(y):
        raise ValueError("x must have shape (N, 2) and y must have N labels.")
    if len(x) == 0 or resolution < 2 or padding < 0:
        raise ValueError("x must be non-empty, resolution >= 2, and padding non-negative.")
    x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
    y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    probabilities = model.predict(np.c_[xx.ravel(), yy.ravel()])
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("model.predict must return class scores with at least two columns.")
    regions = probabilities.argmax(axis=1).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.contourf(xx, yy, regions, alpha=0.25, levels=np.arange(probabilities.shape[1] + 1) - 0.5, cmap="viridis")
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", edgecolors="black", linewidths=0.4, s=24)
    ax.set(title="Decision boundaries", xlabel="Feature 1", ylabel="Feature 2")
    fig.colorbar(scatter, ax=ax, label="Class")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
