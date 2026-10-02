"""Day 4 tests: SGD."""

import numpy as np
import pytest

from nn.optimizers import SGD


def test_sgd_update_rule():
    w, g = np.array([1.0, 2.0]), np.array([0.5, -1.0])
    SGD(lr=0.1).step([(w, g)])
    assert np.allclose(w, [0.95, 2.1])


def test_sgd_updates_in_place():
    w, g = np.ones((2, 2)), np.ones((2, 2))
    ref = w
    SGD(lr=0.5).step([(w, g)])
    assert ref is w and np.allclose(ref, 0.5)


def test_sgd_zero_gradient_leaves_params_unchanged():
    w = np.array([3.0])
    SGD(lr=1.0).step([(w, np.zeros(1))])
    assert w[0] == 3.0


@pytest.mark.parametrize("lr", [0, -0.1])
def test_sgd_rejects_non_positive_lr(lr):
    with pytest.raises(ValueError):
        SGD(lr=lr)


def test_sgd_minimises_a_quadratic():
    """f(w) = (w - 3)^2  ->  grad = 2 (w - 3). Should converge to 3."""
    w = np.array([0.0])
    opt = SGD(lr=0.1)
    for _ in range(200):
        opt.step([(w, 2 * (w - 3.0))])
    assert np.isclose(w[0], 3.0, atol=1e-6)
