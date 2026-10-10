"""Evaluation metrics for binary and multiclass classification."""

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


def _class_labels(values: np.ndarray, *, name: str) -> np.ndarray:
    """Convert one-hot or integer class targets into a 1D integer label array."""
    values = np.asarray(values)
    if values.ndim == 2:
        if values.shape[1] < 2:
            return values.reshape(-1).astype(int)
        return values.argmax(axis=1).astype(int)
    if values.ndim == 1:
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{name} must contain numeric class labels.")
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError(f"{name} must contain finite integer class labels.")
        return values.astype(int)
    raise ValueError(f"{name} must be a 1D label array or 2D one-hot/probability array.")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Return a ``(K, K)`` confusion matrix; rows are true classes, columns predicted.

    Targets can be integer labels or one-hot arrays. Predictions can be integer
    labels or class-score/probability arrays. ``num_classes`` preserves absent classes.
    """
    true_labels = _class_labels(y_true, name="y_true")
    pred_values = np.asarray(y_pred)
    pred_labels = (pred_values.argmax(axis=1) if pred_values.ndim == 2 and pred_values.shape[1] > 1
                   else _class_labels(pred_values, name="y_pred"))
    if true_labels.size == 0 or pred_labels.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if true_labels.shape != pred_labels.shape:
        raise ValueError(f"Shape mismatch: {true_labels.shape} true labels vs {pred_labels.shape} predictions.")
    if np.any(true_labels < 0) or np.any(pred_labels < 0):
        raise ValueError("Class labels must be non-negative.")
    inferred = int(max(true_labels.max(), pred_labels.max())) + 1
    if num_classes is None:
        num_classes = inferred
    if not isinstance(num_classes, (int, np.integer)) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")
    if inferred > num_classes:
        raise ValueError(f"Observed class label requires at least {inferred} classes, got {num_classes}.")
    encoded = true_labels * num_classes + pred_labels
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Return recall/accuracy for each true class; absent classes are ``nan``."""
    matrix = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    counts = matrix.sum(axis=1)
    result = np.full(matrix.shape[0], np.nan, dtype=float)
    present = counts > 0
    result[present] = np.diag(matrix)[present] / counts[present]
    return result
