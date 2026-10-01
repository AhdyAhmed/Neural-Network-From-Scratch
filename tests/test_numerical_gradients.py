"""Day 3 sanity check: analytic gradients vs. centered finite differences.

This is a small local helper. On Day 5 it is promoted to a reusable
gradient checker with a proper pass threshold across the whole library.
"""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Tanh
from nn.layers import Dense
from nn.losses import MSE, BinaryCrossEntropy


def numerical_grad(f, arr, eps=1e-6):
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps
        plus = f()
        arr[i] = old - eps
        minus = f()
        arr[i] = old
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def rel_error(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)


def build(hidden):
    return [Dense(4, 6, seed=0), hidden, Dense(6, 1, seed=1), Sigmoid()]


@pytest.mark.parametrize("hidden", [ReLU, Tanh, Sigmoid])
@pytest.mark.parametrize("loss_cls", [MSE, BinaryCrossEntropy])
def test_full_network_gradients_match_numerical(hidden, loss_cls):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(10, 4))
    y = (rng.random((10, 1)) > 0.5).astype(float)

    layers = build(hidden())
    for layer in layers:  # larger weights than the default so gradients are not tiny
        if isinstance(layer, Dense):
            layer.W = rng.normal(scale=0.5, size=layer.W.shape)
            layer.b = rng.normal(scale=0.1, size=layer.b.shape)
    loss = loss_cls()

    def forward_loss():
        out = x
        for layer in layers:
            out = layer.forward(out)
        return loss.forward(out, y)

    # analytic
    forward_loss()
    grad = loss.backward()
    for layer in reversed(layers):
        grad = layer.backward(grad)

    for layer in layers:
        if isinstance(layer, Dense):
            assert rel_error(layer.dW, numerical_grad(forward_loss, layer.W)) < 1e-6
            assert rel_error(layer.db, numerical_grad(forward_loss, layer.b)) < 1e-6


def test_input_gradient_matches_numerical():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 4))
    y = rng.normal(size=(5, 1))
    layers = [Dense(4, 3, seed=0), Tanh(), Dense(3, 1, seed=1)]
    for layer in layers:  # default weights (std 0.01) give gradients ~1e-5, below float noise
        if isinstance(layer, Dense):
            layer.W = rng.normal(scale=0.5, size=layer.W.shape)
    loss = MSE()

    def forward_loss():
        out = x
        for layer in layers:
            out = layer.forward(out)
        return loss.forward(out, y)

    forward_loss()
    grad = loss.backward()
    for layer in reversed(layers):
        grad = layer.backward(grad)

    assert rel_error(grad, numerical_grad(forward_loss, x)) < 1e-6
