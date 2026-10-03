"""Day 5 bug fix: a caller modifying a returned array must not corrupt backprop.

Found while probing edge cases: Sigmoid and Tanh used to cache the very array
they returned, so ``out += 5`` after ``forward`` silently changed the gradient.
"""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Tanh
from nn.layers import Dense


def make(kind):
    if kind == "dense":
        layer = Dense(2, 2, seed=0)
        layer.W = np.array([[0.5, -1.0], [1.5, 0.3]])
        return layer
    return {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}[kind]()


@pytest.mark.parametrize("kind", ["dense", "relu", "sigmoid", "tanh"])
def test_mutating_the_returned_output_does_not_change_gradients(kind):
    x = np.array([[0.3, -1.2], [1.1, 0.4]])
    g = np.array([[1.0, -2.0], [0.5, 3.0]])

    layer = make(kind)
    layer.forward(x.copy())
    expected = layer.backward(g).copy()

    layer = make(kind)
    out = layer.forward(x.copy())
    out += 5.0  # caller modifies the result in place
    out *= -3.0
    assert np.allclose(layer.backward(g), expected)


@pytest.mark.parametrize("act", [Sigmoid(), Tanh()])
def test_forward_returns_array_independent_of_cache(act):
    out = act.forward(np.array([[0.1, 0.2]]))
    assert out is not act._out
