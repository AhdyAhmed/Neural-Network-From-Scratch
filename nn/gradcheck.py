"""Numerical gradient checking.

Backpropagation is easy to get subtly wrong. This module compares every
analytic gradient against a *centered finite difference*:

    dL/dtheta_i  ~=  [ L(theta_i + eps) - L(theta_i - eps) ] / (2 * eps)

and reports the relative error

    ||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric||)

Rule of thumb (float64, eps = 1e-5):  < 1e-6 is a pass, > 1e-4 is a bug.

Gotchas the checker cannot fix for you:
  * Use realistically scaled weights. With std 0.01 gradients are ~1e-5 and
    finite-difference rounding noise dominates the error.
  * Non-differentiable points (ReLU at 0, probability clipping in BCE) make
    the numerical gradient disagree with the analytic one. Keep test inputs
    away from them.
  * Layers with randomness (Dropout, Day 11) need a fixed mask while checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from nn.layers import Layer
from nn.losses import Loss
from nn.model import Sequential

DEFAULT_EPS = 1e-5
DEFAULT_TOL = 1e-6


# --------------------------------------------------------------- primitives
def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Centered finite-difference gradient of the scalar function ``f`` w.r.t. ``x``.

    ``f`` takes no arguments and must read ``x`` itself; ``x`` is perturbed
    in place one element at a time and always restored.
    """
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    for _ in it:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        plus = f()
        x[idx] = original - eps
        minus = f()
        x[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / (||a|| + ||b||). Falls back to the absolute error when both are ~0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")
    diff = np.linalg.norm(a - b)
    scale = np.linalg.norm(a) + np.linalg.norm(b)
    return float(diff if scale < 1e-12 else diff / scale)


# ------------------------------------------------------------------ result
@dataclass
class GradCheckResult:
    """Relative errors for each checked quantity."""

    errors: dict[str, float] = field(default_factory=dict)
    tol: float = DEFAULT_TOL

    @property
    def max_error(self) -> float:
        return max(self.errors.values()) if self.errors else 0.0

    @property
    def passed(self) -> bool:
        return all(e < self.tol for e in self.errors.values())

    def assert_passed(self) -> None:
        if not self.passed:
            raise AssertionError(f"Gradient check failed (tol={self.tol:g}):\n{self}")

    def __str__(self) -> str:
        lines = [f"  {name:<28} {err:.3e}  {'ok' if err < self.tol else 'FAIL'}"
                 for name, err in self.errors.items()]
        return "\n".join(lines)


# ------------------------------------------------------------------ checkers
def check_layer(
    layer: Layer,
    x: np.ndarray,
    seed: int = 0,
    eps: float = DEFAULT_EPS,
    tol: float = DEFAULT_TOL,
) -> GradCheckResult:
    """Check ``dL/dx`` and every parameter gradient of a single layer.

    The scalar objective is ``L = sum(layer(x) * R)`` with a fixed random
    matrix ``R``, so the upstream gradient is exactly ``R``.
    """
    x = np.array(x, dtype=float)  # private copy we may perturb
    rng = np.random.default_rng(seed)
    out = layer.forward(x, training=True)
    upstream = rng.normal(size=out.shape)

    def objective() -> float:
        return float(np.sum(layer.forward(x, training=True) * upstream))

    layer.forward(x, training=True)
    dx = layer.backward(upstream)
    analytic_params = [(v, g.copy()) for v, g in layer.params()]

    result = GradCheckResult(tol=tol)
    result.errors["input"] = relative_error(dx, numerical_gradient(objective, x, eps))
    for i, (value, grad) in enumerate(analytic_params):
        result.errors[f"param{i} {value.shape}"] = relative_error(
            grad, numerical_gradient(objective, value, eps)
        )
    return result


def check_loss(
    loss: Loss,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = DEFAULT_EPS,
    tol: float = DEFAULT_TOL,
) -> GradCheckResult:
    """Check ``dL/dy_pred`` of a loss function."""
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)
    loss.forward(y_pred, y_true)
    analytic = loss.backward().copy()
    numeric = numerical_gradient(lambda: loss.forward(y_pred, y_true), y_pred, eps)
    return GradCheckResult({"y_pred": relative_error(analytic, numeric)}, tol)


def check_model(
    model: Sequential,
    x: np.ndarray,
    y: np.ndarray,
    eps: float = DEFAULT_EPS,
    tol: float = DEFAULT_TOL,
    check_input: bool = True,
) -> GradCheckResult:
    """Check every parameter gradient (and optionally dL/dx) of a compiled model."""
    if model.loss is None:
        raise RuntimeError("Model must be compiled with a loss before gradient checking.")
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    def objective() -> float:
        return model.loss.forward(model.forward(x, training=True), y)

    objective()
    dx = model.backward(model.loss.backward())
    analytic = [
        (f"layer{i} {type(layer).__name__} param{j}", value, grad.copy())
        for i, layer in enumerate(model.layers)
        for j, (value, grad) in enumerate(layer.params())
    ]

    result = GradCheckResult(tol=tol)
    if check_input:
        result.errors["input"] = relative_error(dx, numerical_gradient(objective, x, eps))
    for name, value, grad in analytic:
        result.errors[name] = relative_error(grad, numerical_gradient(objective, value, eps))
    return result
