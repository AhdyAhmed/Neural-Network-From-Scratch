"""Day 2 tests: Dense layer shapes, values, and validation."""

import numpy as np
import pytest

from nn.layers import Dense, Layer


def test_base_layer_methods_not_implemented():
    with pytest.raises(NotImplementedError):
        Layer().forward(np.zeros((1, 1)))
    with pytest.raises(NotImplementedError):
        Layer().backward(np.zeros((1, 1)))


def test_dense_parameter_shapes():
    layer = Dense(5, 3, seed=0)
    assert layer.W.shape == (5, 3)
    assert layer.b.shape == (1, 3)
    assert np.all(layer.b == 0)


@pytest.mark.parametrize("n", [1, 4, 32])
def test_dense_output_shape(n):
    layer = Dense(5, 3, seed=0)
    out = layer.forward(np.random.default_rng(1).normal(size=(n, 5)))
    assert out.shape == (n, 3)


def test_dense_matches_manual_computation():
    layer = Dense(2, 2, seed=0)
    layer.W = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.b = np.array([[0.5, -0.5]])
    x = np.array([[1.0, 1.0], [2.0, 0.0]])
    expected = np.array([[4.5, 5.5], [2.5, 3.5]])
    assert np.allclose(layer.forward(x), expected)


def test_dense_bias_is_broadcast_over_batch():
    layer = Dense(3, 2, seed=0)
    layer.W = np.zeros((3, 2))
    layer.b = np.array([[1.0, 2.0]])
    out = layer.forward(np.ones((4, 3)))
    assert np.allclose(out, np.tile([1.0, 2.0], (4, 1)))


def test_dense_is_callable():
    layer = Dense(3, 2, seed=0)
    x = np.ones((2, 3))
    assert np.allclose(layer(x), layer.forward(x))


def test_dense_seed_is_reproducible():
    a, b = Dense(4, 4, seed=42), Dense(4, 4, seed=42)
    c = Dense(4, 4, seed=43)
    assert np.array_equal(a.W, b.W)
    assert not np.array_equal(a.W, c.W)


def test_dense_rejects_wrong_feature_count():
    with pytest.raises(ValueError):
        Dense(5, 3, seed=0).forward(np.zeros((2, 4)))


def test_dense_rejects_non_2d_input():
    with pytest.raises(ValueError):
        Dense(5, 3, seed=0).forward(np.zeros(5))


def test_dense_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        Dense(0, 3)


def test_dense_params_pairs_match_shapes():
    layer = Dense(5, 3, seed=0)
    (w, dw), (b, db) = layer.params()
    assert w.shape == dw.shape == (5, 3)
    assert b.shape == db.shape == (1, 3)


