"""Activation functions (ReLU, Sigmoid, Tanh, Softmax).

Day 2: forward pass for ReLU, Sigmoid and Tanh.
Day 3: backward passes. Softmax arrives on Day 7.
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
        self._out = out  # backward only needs the output: s * (1 - s)
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
        self._out = out  # backward only needs the output: 1 - t^2
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """f'(x) = 1 - t^2   ->   dL/dx = dL/da * (1 - t^2)"""
        _check_cached(self._out, "Tanh")
        grad_output = _check_shape(grad_output, self._out.shape, "Tanh")
        return grad_output * (1.0 - self._out**2)

    def __repr__(self) -> str:
        return "Tanh()"
