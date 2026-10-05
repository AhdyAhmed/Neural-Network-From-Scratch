"""Loss functions (MSE, BinaryCrossEntropy, CategoricalCrossEntropy).

Day 3: MSE and BinaryCrossEntropy.  Day 7: CategoricalCrossEntropy.

Contract (see design.md):
    forward(y_pred, y_true) -> float   (also caches what backward needs)
    backward()              -> dL/dy_pred, same shape as y_pred

Both losses average over *all* elements of the batch, so the 1/N factor lives
here and the layers never need to know the batch size.
"""

from __future__ import annotations

import numpy as np


class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred, y_true)


def _prepare(y_pred, y_true) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred and y_true must have the same shape, got {y_pred.shape} and {y_true.shape}."
        )
    return y_pred, y_true


class MSE(Loss):
    """Mean squared error:  L = mean((y_pred - y_true)^2).

    Gradient:  dL/dy_pred = 2 (y_pred - y_true) / size
    """

    def __init__(self) -> None:
        self._y_pred: np.ndarray | None = None
        self._y_true: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred, y_true = _prepare(y_pred, y_true)
        self._y_pred, self._y_true = y_pred, y_true
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        if self._y_pred is None:
            raise RuntimeError("backward() called before forward().")
        return 2.0 * (self._y_pred - self._y_true) / self._y_pred.size


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy for probabilities from a Sigmoid output.

        L = -mean( y log(p) + (1 - y) log(1 - p) )
        dL/dp = (p - y) / (p (1 - p)) / size

    Probabilities are clipped to [eps, 1 - eps] so log(0) and division by zero
    can never produce inf / NaN.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self._p: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred, y_true = _prepare(y_pred, y_true)
        p = np.clip(y_pred, self.eps, 1.0 - self.eps)
        self._p, self._y = p, y_true
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def backward(self) -> np.ndarray:
        if self._p is None:
            raise RuntimeError("backward() called before forward().")
        return (self._p - self._y) / (self._p * (1.0 - self._p)) / self._p.size


class CategoricalCrossEntropy(Loss):
    """Cross-entropy for multi-class classification.

    Expects softmax probabilities ``y_pred`` of shape (N, K) and one-hot (or
    soft) targets ``y_true`` of shape (N, K)  (see ``nn.utils.one_hot``).

        L = -(1/N) sum_n sum_k y[n,k] log p[n,k]

    Note the average is over the N *samples* (not N*K elements like MSE/BCE).

    Two gradients are provided:

    * ``backward()``        dL/dp = -y / (p N)   (gradient w.r.t. the probabilities)
    * ``backward_logits()`` dL/dz = (p - y) / N  (gradient w.r.t. the logits,
      i.e. softmax and cross-entropy fused)

    The fused form is simpler and more stable: if the model is confidently
    wrong, p for the true class underflows, so ``-y/p`` is clipped and the
    chain through softmax returns a gradient near 0, stalling learning. The
    fused gradient stays exactly ``p - y``.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self._p: np.ndarray | None = None  # raw probabilities
        self._y: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        try:
            y_pred, y_true = _prepare(y_pred, y_true)
        except ValueError as exc:
            raise ValueError(f"{exc} For class labels, convert with nn.utils.one_hot first.") from exc
        if y_pred.ndim != 2:
            raise ValueError(f"CategoricalCrossEntropy expects 2D (N, classes) input, got {y_pred.shape}.")
        self._p, self._y = y_pred, y_true
        p = np.clip(y_pred, self.eps, 1.0)
        return float(-np.sum(y_true * np.log(p)) / y_pred.shape[0])

    def backward(self) -> np.ndarray:
        if self._p is None:
            raise RuntimeError("backward() called before forward().")
        p = np.clip(self._p, self.eps, 1.0)
        return -self._y / p / self._p.shape[0]

    def backward_logits(self) -> np.ndarray:
        """Fused softmax + cross-entropy gradient w.r.t. the logits: (p - y) / N."""
        if self._p is None:
            raise RuntimeError("backward_logits() called before forward().")
        return (self._p - self._y) / self._p.shape[0]
