"""Helpers: batching, shuffling, one-hot encoding. Data loading arrives on Day 8."""

from __future__ import annotations

from typing import Iterator

import numpy as np


def one_hot(labels: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Convert integer class labels of shape (N,) to a one-hot matrix of shape (N, K).

        one_hot([0, 2, 1], 3)  ->  [[1, 0, 0],
                                    [0, 0, 1],
                                    [0, 1, 0]]

    ``num_classes`` defaults to ``max(labels) + 1``. Pass it explicitly when a
    subset of the data might not contain every class.
    """
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:  # tolerate a column vector
        labels = labels.ravel()
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D (N,), got shape {labels.shape}.")
    if labels.size and not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.mod(labels, 1) == 0):
            raise ValueError("labels must be integers.")
        labels = labels.astype(int)
    if labels.size and labels.min() < 0:
        raise ValueError("labels must be non-negative.")

    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size else 0
    if labels.size and labels.max() >= num_classes:
        raise ValueError(f"label {int(labels.max())} is out of range for num_classes={num_classes}.")

    out = np.zeros((labels.shape[0], num_classes))
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(x_batch, y_batch)`` pairs covering every sample exactly once.

    The last batch is smaller if ``batch_size`` does not divide N. With
    ``shuffle=True`` the sample order is a fresh random permutation each call
    (x and y are permuted together so pairs stay aligned).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    n = len(x)
    if len(y) != n:
        raise ValueError(f"x and y must have the same length, got {n} and {len(y)}.")

    if shuffle:
        rng = rng if rng is not None else np.random.default_rng()
        order = rng.permutation(n)
    else:
        order = np.arange(n)

    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]
