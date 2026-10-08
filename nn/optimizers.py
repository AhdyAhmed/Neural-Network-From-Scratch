"""Optimizers (SGD, Momentum, RMSProp, Adam).

Day 4: SGD.  Day 9: Momentum.  Day 10: RMSProp and Adam.

An optimizer receives the list of ``(value, gradient)`` pairs returned by
``Layer.params()`` and updates each ``value`` **in place**.
"""

from __future__ import annotations

import numpy as np

Param = tuple[np.ndarray, np.ndarray]


class Optimizer:
    def step(self, params: list[Param]) -> None:
        raise NotImplementedError


class _Stateful(Optimizer):
    """Base for optimizers that keep one or more arrays per parameter (velocity, moment estimates...).

    State is stored by position, in the order parameters are passed to ``step``, and created lazily
    on the first call. Using the optimizer with a different parameter list raises an error instead of
    silently mixing up statistics; call ``reset()`` to start over.
    """

    def __init__(self) -> None:
        self._state: list[list[np.ndarray]] | None = None  # _state[slot][param_index]
        self.t = 0                                          # number of steps taken

    def _slots(self, params: list[Param], n_slots: int) -> list[list[np.ndarray]]:
        if self._state is None:
            self._state = [[np.zeros_like(value) for value, _ in params] for _ in range(n_slots)]
        elif len(self._state[0]) != len(params) or any(
            a.shape != value.shape for a, (value, _) in zip(self._state[0], params)
        ):
            raise ValueError("Optimizer was used with a different set of parameters; call reset() first.")
        return self._state

    def reset(self) -> None:
        """Forget all accumulated state (e.g. before reusing the optimizer on a new model)."""
        self._state = None
        self.t = 0


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


class Momentum(_Stateful):
    """SGD with momentum: a running average of past gradients smooths the updates.

        v <- beta * v + grad
        theta <- theta - lr * v

    Along directions where the gradient keeps pointing the same way, ``v`` builds
    up (up to 1 / (1 - beta) times the gradient: 10x for beta = 0.9) and training
    speeds up. Along directions where the gradient flips sign every step (steep,
    narrow valleys), the contributions cancel and the oscillation is damped.

    ``beta = 0`` is exactly plain SGD.
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        super().__init__()
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1).")
        self.lr = lr
        self.beta = beta

    def step(self, params: list[Param]) -> None:
        (velocity,) = self._slots(params, 1)
        self.t += 1
        for v, (value, grad) in zip(velocity, params):
            v *= self.beta
            v += grad
            value -= self.lr * v

    def __repr__(self) -> str:
        return f"Momentum(lr={self.lr}, beta={self.beta})"


class RMSProp(_Stateful):
    """Scale each parameter's step by a running average of its recent squared gradients.

        s <- rho * s + (1 - rho) * grad^2
        theta <- theta - lr * grad / (sqrt(s) + eps)

    Parameters with consistently large gradients get smaller steps and parameters with small
    gradients get larger ones, so differently-scaled directions all move at a similar pace.
    Note the very first step is ``lr / sqrt(1 - rho)`` (about 3.2 * lr for rho = 0.9) in the
    direction of ``-sign(grad)``, because ``s`` starts at 0 and has not warmed up yet.
    """

    def __init__(self, lr: float = 0.001, rho: float = 0.9, eps: float = 1e-8) -> None:
        super().__init__()
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1).")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.lr, self.rho, self.eps = lr, rho, eps

    def step(self, params: list[Param]) -> None:
        (sq,) = self._slots(params, 1)
        self.t += 1
        for s_, (value, grad) in zip(sq, params):
            s_ *= self.rho
            s_ += (1.0 - self.rho) * grad * grad
            value -= self.lr * grad / (np.sqrt(s_) + self.eps)

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.lr}, rho={self.rho})"


class Adam(_Stateful):
    """Adam: momentum (first moment) plus RMSProp-style scaling (second moment), bias-corrected.

        m <- beta1 * m + (1 - beta1) * grad
        v <- beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)          v_hat = v / (1 - beta2^t)
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

    ``m`` and ``v`` start at 0, so early on they are biased toward 0 (after one step ``m`` is only
    10% of the gradient and ``v`` only 0.1% of its square). Dividing by ``1 - beta^t`` undoes that
    bias exactly. Without it the first steps would be wrong by a large factor; with it the very
    first update is ``lr * sign(grad)``, and the correction fades as ``t`` grows.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        super().__init__()
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps

    def step(self, params: list[Param]) -> None:
        m_all, v_all = self._slots(params, 2)
        self.t += 1
        c1 = 1.0 - self.beta1**self.t   # bias corrections
        c2 = 1.0 - self.beta2**self.t
        for m, v, (value, grad) in zip(m_all, v_all, params):
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * grad * grad
            value -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"
