"""Day 3 tests: MSE and BinaryCrossEntropy values, gradients, and stability."""

import numpy as np
import pytest

from nn.losses import MSE, BinaryCrossEntropy


# ------------------------------------------------------------------ MSE
def test_mse_value():
    loss = MSE().forward(np.array([[1.0], [3.0]]), np.array([[0.0], [1.0]]))
    assert np.isclose(loss, (1.0 + 4.0) / 2)


def test_mse_zero_when_perfect():
    y = np.array([[0.2, 0.8]])
    assert MSE().forward(y, y) == 0.0


def test_mse_gradient():
    mse = MSE()
    mse.forward(np.array([[1.0], [3.0]]), np.array([[0.0], [1.0]]))
    assert np.allclose(mse.backward(), [[1.0], [2.0]])  # 2 * (pred - true) / 2


def test_mse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        MSE().forward(np.zeros((3, 1)), np.zeros((3,)))


def test_mse_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        MSE().backward()


# ------------------------------------------------------------------ BCE
def test_bce_value_known():
    bce = BinaryCrossEntropy()
    p, y = np.array([[0.5]]), np.array([[1.0]])
    assert np.isclose(bce.forward(p, y), np.log(2))


def test_bce_confident_correct_is_small_and_wrong_is_large():
    bce = BinaryCrossEntropy()
    y = np.array([[1.0]])
    good = bce.forward(np.array([[0.99]]), y)
    bad = bce.forward(np.array([[0.01]]), y)
    assert good < 0.02 < 4.0 < bad


def test_bce_gradient_known_value():
    bce = BinaryCrossEntropy()
    bce.forward(np.array([[0.8]]), np.array([[1.0]]))
    assert np.isclose(bce.backward()[0, 0], -1.0 / 0.8)  # (p - y) / (p (1 - p))


def test_bce_is_finite_at_extremes():
    bce = BinaryCrossEntropy()
    p = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([[1.0], [0.0], [0.0], [1.0]])
    with np.errstate(all="raise"):
        loss = bce.forward(p, y)
        grad = bce.backward()
    assert np.isfinite(loss) and np.all(np.isfinite(grad))


def test_bce_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        BinaryCrossEntropy().backward()


def test_sigmoid_plus_bce_gradient_simplifies():
    """dL/dz through sigmoid + BCE should equal (p - y) / size."""
    from nn.activations import Sigmoid

    z = np.random.default_rng(0).normal(size=(8, 1))
    y = (np.random.default_rng(1).random((8, 1)) > 0.5).astype(float)
    sig, bce = Sigmoid(), BinaryCrossEntropy()
    p = sig.forward(z)
    bce.forward(p, y)
    dz = sig.backward(bce.backward())
    assert np.allclose(dz, (p - y) / p.size, atol=1e-10)
