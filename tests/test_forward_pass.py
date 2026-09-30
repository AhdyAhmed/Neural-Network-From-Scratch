"""Day 2 integration test: chain layers manually (Sequential arrives on Day 4)."""

import numpy as np

from nn.activations import ReLU, Sigmoid
from nn.layers import Dense


def run(layers, x):
    for layer in layers:
        x = layer.forward(x)
    return x


def test_two_layer_forward_shape():
    net = [Dense(4, 8, seed=0), ReLU(), Dense(8, 1, seed=1), Sigmoid()]
    x = np.random.default_rng(0).normal(size=(16, 4))
    out = run(net, x)
    assert out.shape == (16, 1)
    assert np.all((out > 0) & (out < 1))


def test_forward_is_deterministic():
    make = lambda: [Dense(3, 5, seed=7), ReLU(), Dense(5, 2, seed=8)]
    x = np.random.default_rng(3).normal(size=(10, 3))
    assert np.array_equal(run(make(), x), run(make(), x))


def test_day1_neuron_reproduced_with_layers():
    """Dense(2->1) + Sigmoid must reproduce the Day 1 hand calculation."""
    dense = Dense(2, 1, seed=0)
    dense.W = np.array([[0.5], [-0.5]])
    dense.b = np.array([[0.1]])
    out = run([dense, Sigmoid()], np.array([[1.0, 2.0]]))
    assert np.isclose(out[0, 0], 0.401312, atol=1e-6)
