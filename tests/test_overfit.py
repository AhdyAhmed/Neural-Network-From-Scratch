"""Day 5 sanity check: a network must be able to memorise a tiny batch.

If backprop or the optimizer is subtly wrong, loss will stall well above zero.
Weights use a moderate scale because the default init (std 0.01) trains slowly
(He/Xavier arrive on Day 9).
"""

import numpy as np

from nn.activations import ReLU, Sigmoid, Tanh
from nn.layers import Dense
from nn.losses import MSE, BinaryCrossEntropy
from nn.model import Sequential
from nn.optimizers import SGD


def build(layers, loss, lr, scale=0.8):
    rng = np.random.default_rng(0)
    for layer in layers:
        if isinstance(layer, Dense):
            layer.W = rng.normal(scale=scale, size=layer.W.shape)
    model = Sequential(layers)
    model.compile(loss=loss, optimizer=SGD(lr=lr))
    return model


def tiny_batch(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(10, 3)), rng


def test_tanh_network_memorises_random_regression_targets():
    x, rng = tiny_batch()
    y = rng.normal(size=(10, 1))  # pure noise: only memorisation can fit it
    model = build([Dense(3, 32, seed=1), Tanh(), Dense(32, 1, seed=2)], MSE(), lr=0.1)
    history = model.fit(x, y, epochs=2000, verbose=0)
    assert history["loss"][-1] < 1e-6
    assert history["loss"][-1] < history["loss"][0] * 1e-6


def test_relu_network_memorises_random_regression_targets():
    x, rng = tiny_batch()
    y = rng.normal(size=(10, 1))
    model = build([Dense(3, 32, seed=1), ReLU(), Dense(32, 1, seed=2)], MSE(), lr=0.05)
    model.fit(x, y, epochs=2000, verbose=0)
    assert model.evaluate(x, y) < 1e-4


def test_network_memorises_random_binary_labels():
    x, rng = tiny_batch()
    y = (rng.random((10, 1)) > 0.5).astype(float)
    model = build(
        [Dense(3, 32, seed=1), Tanh(), Dense(32, 1, seed=2), Sigmoid()],
        BinaryCrossEntropy(),
        lr=0.5,
    )
    model.fit(x, y, epochs=2000, verbose=0)
    assert np.mean((model.predict(x) > 0.5) == y) == 1.0
    assert model.evaluate(x, y) < 0.05
