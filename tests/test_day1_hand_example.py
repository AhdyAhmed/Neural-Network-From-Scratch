"""Verify the Day 1 hand-worked example: analytic gradients match finite differences."""

import numpy as np

from examples.day1_hand_example import B, W, X, Y, forward, gradients, loss


def test_forward_values_match_hand_calculation():
    z, a = forward(X, W, B)
    assert np.isclose(z, -0.4)
    assert np.isclose(a, 0.401312, atol=1e-6)
    assert np.isclose(loss(a, Y), 0.179213, atol=1e-6)


def test_analytic_gradients_match_hand_calculation():
    dw, db = gradients(X, W, B, Y)
    assert np.allclose(dw, [-0.143841, -0.287682], atol=1e-6)
    assert np.isclose(db, -0.143841, atol=1e-6)


def test_analytic_gradients_match_numerical_gradients():
    dw, db = gradients(X, W, B, Y)
    eps = 1e-6

    def f(w, b):
        _, a = forward(X, w, b)
        return loss(a, Y)

    num_dw = np.zeros_like(W)
    for i in range(W.size):
        w_plus, w_minus = W.copy(), W.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        num_dw[i] = (f(w_plus, B) - f(w_minus, B)) / (2 * eps)
    num_db = (f(W, B + eps) - f(W, B - eps)) / (2 * eps)

    assert np.allclose(dw, num_dw, atol=1e-8)
    assert np.isclose(db, num_db, atol=1e-8)
