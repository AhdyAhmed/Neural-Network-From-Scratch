"""Day 3 tests: backward passes of Dense and the activations (hand-checked values)."""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Tanh
from nn.layers import Dense


# ---------------------------------------------------------------- Dense
def test_dense_backward_shapes():
    layer = Dense(5, 3, seed=0)
    x = np.random.default_rng(0).normal(size=(7, 5))
    layer.forward(x)
    dx = layer.backward(np.ones((7, 3)))
    assert dx.shape == (7, 5)
    assert layer.dW.shape == (5, 3)
    assert layer.db.shape == (1, 3)


def test_dense_backward_matches_manual_computation():
    layer = Dense(2, 2, seed=0)
    layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.dW, layer.db = np.zeros((2, 2)), np.zeros((1, 2))
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    g = np.array([[1.0, 0.0], [0.0, 1.0]])
    layer.forward(x)
    dx = layer.backward(g)
    assert np.allclose(layer.dW, x.T @ g)                  # [[1, 3], [2, 4]]
    assert np.allclose(layer.dW, [[1.0, 3.0], [2.0, 4.0]])
    assert np.allclose(layer.db, [[1.0, 1.0]])
    assert np.allclose(dx, [[1.0, 3.0], [2.0, 4.0]])       # g @ W^T


def test_dense_params_stay_linked_to_gradients():
    layer = Dense(3, 2, seed=0)
    (_, dw_ref), (_, db_ref) = layer.params()
    layer.forward(np.ones((4, 3)))
    layer.backward(np.ones((4, 2)))
    assert np.allclose(dw_ref, 4.0) and np.allclose(db_ref, 4.0)  # updated in place


def test_dense_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        Dense(2, 2, seed=0).backward(np.ones((1, 2)))


def test_dense_backward_wrong_shape_raises():
    layer = Dense(3, 2, seed=0)
    layer.forward(np.ones((4, 3)))
    with pytest.raises(ValueError):
        layer.backward(np.ones((4, 3)))


# ---------------------------------------------------------- activations
def test_relu_backward_masks_non_positive_inputs():
    relu = ReLU()
    relu.forward(np.array([[-1.0, 0.0, 2.0]]))
    assert np.array_equal(relu.backward(np.array([[5.0, 5.0, 5.0]])), [[0.0, 0.0, 5.0]])


def test_sigmoid_backward_known_value():
    sig = Sigmoid()
    sig.forward(np.array([[0.0]]))
    assert np.isclose(sig.backward(np.array([[1.0]]))[0, 0], 0.25)  # s(1 - s) at 0


def test_sigmoid_backward_saturates():
    sig = Sigmoid()
    sig.forward(np.array([[-50.0, 50.0]]))
    assert np.allclose(sig.backward(np.ones((1, 2))), 0.0, atol=1e-12)


def test_tanh_backward_known_values():
    t = Tanh()
    t.forward(np.array([[0.0, 1.0]]))
    expected = [[1.0, 1.0 - np.tanh(1.0) ** 2]]
    assert np.allclose(t.backward(np.ones((1, 2))), expected)


@pytest.mark.parametrize("act", [ReLU(), Sigmoid(), Tanh()])
def test_activation_backward_before_forward_raises(act):
    with pytest.raises(RuntimeError):
        act.backward(np.ones((1, 1)))


@pytest.mark.parametrize("act", [ReLU(), Sigmoid(), Tanh()])
def test_activation_backward_wrong_shape_raises(act):
    act.forward(np.ones((2, 3)))
    with pytest.raises(ValueError):
        act.backward(np.ones((2, 2)))


@pytest.mark.parametrize("act", [ReLU(), Sigmoid(), Tanh()])
def test_activation_backward_preserves_shape(act):
    x = np.random.default_rng(0).normal(size=(6, 4))
    act.forward(x)
    assert act.backward(np.ones_like(x)).shape == x.shape


# ------------------------------------------------- cross-check with Day 1
def test_matches_day1_hand_example():
    """Dense(2->1) + Sigmoid + MSE reproduces the Day 1 gradients.

    Day 1 used L = 0.5 (a - y)^2, while MSE here is (a - y)^2, so every
    gradient is exactly twice the hand-calculated value.
    """
    from nn.losses import MSE

    dense = Dense(2, 1, seed=0)
    dense.W = np.array([[0.5], [-0.5]])
    dense.b = np.array([[0.1]])
    sig, mse = Sigmoid(), MSE()

    a = sig.forward(dense.forward(np.array([[1.0, 2.0]])))
    assert np.isclose(mse.forward(a, np.array([[1.0]])), 2 * 0.179213, atol=1e-6)
    dense.backward(sig.backward(mse.backward()))

    assert np.allclose(dense.dW.ravel(), 2 * np.array([-0.143841, -0.287682]), atol=1e-5)
    assert np.isclose(dense.db[0, 0], 2 * -0.143841, atol=1e-5)
