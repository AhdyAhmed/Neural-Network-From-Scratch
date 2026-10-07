"""Optimizers (SGD, Momentum, RMSProp, Adam).

Day 4: SGD.  Day 9: Momentum.  RMSProp and Adam arrive on Day 10.

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


class Momentum(Optimizer):
    """SGD with momentum: a running average of past gradients smooths the updates.

        v <- beta * v + grad
        theta <- theta - lr * v

    Along directions where the gradient keeps pointing the same way, ``v`` builds
    up (up to 1 / (1 - beta) times the gradient: 10x for beta = 0.9) and training
    speeds up. Along directions where the gradient flips sign every step (steep,
    narrow valleys), the contributions cancel and the oscillation is damped.

    ``beta = 0`` is exactly plain SGD. One velocity array is kept per parameter,
    in the order the parameters are passed to ``step``.
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1).")
        self.lr = lr
        self.beta = beta
        self._velocity: list[np.ndarray] | None = None

    def step(self, params: list[Param]) -> None:
        if self._velocity is None:
            self._velocity = [np.zeros_like(value) for value, _ in params]
        elif len(self._velocity) != len(params) or any(
            v.shape != value.shape for v, (value, _) in zip(self._velocity, params)
        ):
            raise ValueError("Momentum was used with a different set of parameters; call reset() first.")

        for v, (value, grad) in zip(self._velocity, params):
            v *= self.beta
            v += grad
            value -= self.lr * v

    def reset(self) -> None:
        """Forget all accumulated velocity (e.g. before reusing the optimizer on a new model)."""
        self._velocity = None

    def __repr__(self) -> str:
        return f"Momentum(lr={self.lr}, beta={self.beta})"
