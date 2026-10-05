"""Activation functions (ReLU, Sigmoid, Tanh, Softmax).

Day 2: forward pass for ReLU, Sigmoid and Tanh.
Day 3: backward passes. Day 7: Softmax.
"""

from __future__ import annotations

import numpy as np

from nn.layers import Layer


def _check_cached(cache: np.ndarray | None, name: str) -> None:
    if cache is None:
        raise RuntimeError(f"{name}.backward() called before forward().")


def _check_shape(grad_output: np.ndarray, shape: tuple, name: str) -> np.ndarray:
    grad_output = np.asarray(grad_output, dtype=float)
    if grad_output.shape != shape:
        raise ValueError(
            f"{name}.backward expected grad_output of shape {shape}, "
            f"got {grad_output.shape}."
        )
    return grad_output


class ReLU(Layer):
    """f(x) = max(0, x)"""

    def __init__(self) -> None:
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self._x = x
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """f'(x) = 1 if x > 0 else 0   ->   dL/dx = dL/da * (x > 0)"""
        _check_cached(self._x, "ReLU")
        grad_output = _check_shape(grad_output, self._x.shape, "ReLU")
        return grad_output * (self._x > 0)

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Layer):
    """f(x) = 1 / (1 + exp(-x)), computed in a numerically stable way.

    Naively evaluating exp(-x) overflows for large negative x, so we use
    1 / (1 + exp(-x)) when x >= 0 and exp(x) / (1 + exp(x)) when x < 0.
    """

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        # Cache a private copy: the caller receives `out` and may modify it in place,
        # which would silently corrupt the gradient. backward only needs s * (1 - s).
        self._out = out.copy()
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """f'(x) = s(1 - s)   ->   dL/dx = dL/da * s * (1 - s)"""
        _check_cached(self._out, "Sigmoid")
        grad_output = _check_shape(grad_output, self._out.shape, "Sigmoid")
        return grad_output * self._out * (1.0 - self._out)

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Layer):
    """f(x) = tanh(x)"""

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.tanh(x)
        # Private copy for the same reason as Sigmoid. backward only needs 1 - t^2.
        self._out = out.copy()
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """f'(x) = 1 - t^2   ->   dL/dx = dL/da * (1 - t^2)"""
        _check_cached(self._out, "Tanh")
        grad_output = _check_shape(grad_output, self._out.shape, "Tanh")
        return grad_output * (1.0 - self._out**2)

    def __repr__(self) -> str:
        return "Tanh()"


class Softmax(Layer):
    """Row-wise softmax:  s_k = exp(z_k) / sum_j exp(z_j).

    Numerically stable: subtracting each row's maximum before exponentiating
    leaves the result unchanged but prevents ``exp`` from overflowing.

    ``backward`` is the full Jacobian-vector product. For one row,
    ``ds_k/dz_j = s_k (delta_kj - s_j)``, so given upstream gradient ``g``:

        dz = s * (g - sum(g * s))

    When Softmax is the last layer and the loss is CategoricalCrossEntropy,
    ``Sequential.backward_from_loss`` skips this method and uses the simpler,
    more stable fused gradient ``(s - y) / N`` instead.
    """

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"Softmax expected a 2D input (N, classes), got shape {x.shape}.")
        shifted = x - x.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        out = exp / exp.sum(axis=1, keepdims=True)
        self._out = out.copy()  # private copy, same reasoning as Sigmoid/Tanh
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        _check_cached(self._out, "Softmax")
        grad_output = _check_shape(grad_output, self._out.shape, "Softmax")
        s = self._out
        return s * (grad_output - np.sum(grad_output * s, axis=1, keepdims=True))

    def __repr__(self) -> str:
        return "Softmax()"
