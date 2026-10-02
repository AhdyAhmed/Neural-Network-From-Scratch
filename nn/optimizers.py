"""Optimizers (SGD, Momentum, RMSProp, Adam).

Day 4: SGD.  Momentum on Day 9, RMSProp and Adam on Day 10.

An optimizer receives the list of ``(value, gradient)`` pairs returned by
``Layer.params()`` and updates each ``value`` **in place**.
"""

from __future__ import annotations

import numpy as np

Param = tuple[np.ndarray, np.ndarray]


class Optimizer:
    def step(self, params: list[Param]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Plain stochastic gradient descent:  theta <- theta - lr * grad."""

    def __init__(self, lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.lr = lr

    def step(self, params: list[Param]) -> None:
        for value, grad in params:
            value -= self.lr * grad

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr})"
