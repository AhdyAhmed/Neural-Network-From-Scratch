"""Evaluation metrics. Day 8: accuracy.  Confusion matrix arrives on Day 12."""

from __future__ import annotations

import numpy as np


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of correct predictions.

    Accepted layouts (``y_pred`` is probabilities / scores, ``y_true`` is targets):

    * multi-class: ``y_pred`` (N, K) and ``y_true`` either one-hot (N, K) or integer labels (N,)
    * binary:      ``y_pred`` (N, 1) or (N,) probabilities, thresholded at 0.5, and ``y_true`` 0/1
    """
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        predicted = y_pred.argmax(axis=1)
        actual = y_true.argmax(axis=1) if y_true.ndim == 2 else y_true
    else:
        predicted = (y_pred.reshape(-1) > 0.5).astype(int)
        actual = y_true.reshape(-1)
    if predicted.shape != actual.shape:
        raise ValueError(f"Shape mismatch: {np.shape(y_pred)} predictions vs {np.shape(y_true)} targets.")
    return float(np.mean(predicted == actual))
