"""Day 5: extra shape / value / edge-case unit tests."""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Tanh
from nn.layers import Dense
from nn.losses import MSE, BinaryCrossEntropy


def test_dense_accepts_integer_input():
    out = Dense(2, 2, seed=0).forward(np.array([[1, 2]]))
    assert out.dtype == np.float64


def test_dense_batch_of_one():
    assert Dense(4, 3, seed=0).forward(np.ones((1, 4))).shape == (1, 3)


def test_dense_forward_is_linear():
    layer = Dense(3, 2, seed=0)
    layer.b[:] = 0.0
    a, b = np.random.default_rng(0).normal(size=(2, 4, 3))
    assert np.allclose(layer.forward(a + b), layer.forward(a) + layer.forward(b))


def test_dense_forward_does_not_modify_input():
    x = np.random.default_rng(0).normal(size=(4, 3))
    before = x.copy()
    Dense(3, 2, seed=0).forward(x)
    assert np.array_equal(x, before)


@pytest.mark.parametrize("act", [ReLU(), Sigmoid(), Tanh()])
def test_activations_do_not_modify_input(act):
    x = np.random.default_rng(0).normal(size=(4, 3))
    before = x.copy()
    act.forward(x)
    assert np.array_equal(x, before)


def test_relu_is_idempotent():
    x = np.random.default_rng(0).normal(size=(5, 5))
    relu = ReLU()
    assert np.array_equal(relu.forward(relu.forward(x)), relu.forward(x))


def test_tanh_is_odd_and_sigmoid_is_monotonic():
    x = np.linspace(-6, 6, 25).reshape(1, -1)
    assert np.allclose(Tanh().forward(x), -Tanh().forward(-x))
    assert np.all(np.diff(Sigmoid().forward(x)) > 0)


def test_sigmoid_matches_tanh_identity():
    """sigmoid(x) = (1 + tanh(x / 2)) / 2"""
    x = np.random.default_rng(0).normal(scale=3, size=(5, 5))
    assert np.allclose(Sigmoid().forward(x), (1 + np.tanh(x / 2)) / 2)


def test_activations_propagate_nan_without_crashing():
    x = np.array([[np.nan, 1.0]])
    for act in (ReLU(), Sigmoid(), Tanh()):
        assert np.isnan(act.forward(x)[0, 0])


def test_mse_is_symmetric_and_non_negative():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 6, 2))
    assert MSE().forward(a, b) == MSE().forward(b, a) >= 0


def test_mse_accepts_integer_arrays():
    assert MSE().forward(np.array([[1], [2]]), np.array([[0], [0]])) == 2.5


def test_bce_is_non_negative():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, size=(20, 1))
    y = (rng.random((20, 1)) > 0.5).astype(float)
    assert BinaryCrossEntropy().forward(p, y) >= 0


def test_bce_minimised_when_prediction_equals_label():
    y = np.array([[1.0], [0.0]])
    at_label = BinaryCrossEntropy().forward(np.array([[0.999], [0.001]]), y)
    off = BinaryCrossEntropy().forward(np.array([[0.7], [0.3]]), y)
    assert at_label < off
