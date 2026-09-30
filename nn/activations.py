"""Activation functions (ReLU, Sigmoid, Tanh, Softmax).

Day 2: forward pass for ReLU, Sigmoid and Tanh.
Backward passes arrive on Day 3, Softmax on Day 7.
"""

from __future__ import annotations

import numpy as np

from nn.layers import Layer


class ReLU(Layer):
    """f(x) = max(0, x)"""

    def __init__(self) -> None:
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self._x = x
        return np.maximum(0.0, x)

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

    def __repr__(self) -> str:
        return "Tanh()"
