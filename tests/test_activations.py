"""Day 2 tests: ReLU, Sigmoid, Tanh forward values and stability."""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Tanh


def test_relu_values():
    x = np.array([[-2.0, -0.5, 0.0, 0.5, 3.0]])
    assert np.array_equal(ReLU().forward(x), [[0.0, 0.0, 0.0, 0.5, 3.0]])


def test_sigmoid_known_values():
    out = Sigmoid().forward(np.array([[0.0, -0.4]]))
    assert np.allclose(out, [[0.5, 0.401312]], atol=1e-6)  # matches Day 1 example


def test_sigmoid_is_stable_for_extreme_inputs():
    x = np.array([[-1000.0, -50.0, 0.0, 50.0, 1000.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):  # underflow to 0 is harmless
        out = Sigmoid().forward(x)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, [[0.0, 0.0, 0.5, 1.0, 1.0]], atol=1e-12)


def test_sigmoid_symmetry():
    x = np.linspace(-5, 5, 11).reshape(1, -1)
    s = Sigmoid()
    assert np.allclose(s.forward(x) + s.forward(-x), 1.0)


def test_tanh_values():
    x = np.array([[-1.0, 0.0, 1.0]])
    assert np.allclose(Tanh().forward(x), np.tanh(x))
    assert np.isclose(Tanh().forward(np.array([[0.0]]))[0, 0], 0.0)


@pytest.mark.parametrize("act", [ReLU(), Sigmoid(), Tanh()])
def test_activations_preserve_shape(act):
    x = np.random.default_rng(0).normal(size=(6, 4))
    assert act.forward(x).shape == x.shape


@pytest.mark.parametrize("act", [Sigmoid(), Tanh()])
def test_activation_output_ranges(act):
    x = np.random.default_rng(0).normal(scale=10, size=(50, 5))
    out = act.forward(x)
    low = 0.0 if isinstance(act, Sigmoid) else -1.0
    assert np.all(out >= low) and np.all(out <= 1.0)
