"""Day 7: whole models with Softmax + CategoricalCrossEntropy (fused backward path)."""

import numpy as np
import pytest

from nn.activations import ReLU, Softmax, Tanh
from nn.gradcheck import check_model
from nn.layers import Dense
from nn.losses import MSE, CategoricalCrossEntropy
from nn.model import Sequential
from nn.optimizers import SGD
from nn.utils import one_hot


def scaled_dense(n_in, n_out, seed):
    layer = Dense(n_in, n_out, seed=seed)
    rng = np.random.default_rng(seed + 100)
    layer.W = rng.normal(scale=0.7, size=layer.W.shape)
    layer.b = rng.normal(scale=0.2, size=layer.b.shape)
    layer.dW, layer.db = np.zeros_like(layer.W), np.zeros_like(layer.b)
    return layer


def classifier(hidden, k=4, loss=None, seed=0):
    layers = [scaled_dense(3, 6, seed), hidden(), scaled_dense(6, k, seed + 1), Softmax()]
    m = Sequential(layers)
    m.compile(loss=loss or CategoricalCrossEntropy(), optimizer=SGD(lr=0.1))
    return m


def batch(n=9, k=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), one_hot(rng.integers(0, k, size=n), k)


# --------------------------------------------------- initial loss sanity check
@pytest.mark.parametrize("k", [3, 10, 20])
def test_initial_loss_is_ln_k_for_small_random_weights(k):
    """Roadmap sanity check: an untrained K-class softmax model predicts ~uniform, loss ~ ln(K)."""
    x = np.random.default_rng(0).normal(size=(200, 20))
    y = one_hot(np.random.default_rng(1).integers(0, k, size=200), k)
    m = Sequential([Dense(20, 32, seed=0), Tanh(), Dense(32, k, seed=1), Softmax()])
    m.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=0.1))
    assert abs(m.evaluate(x, y) - np.log(k)) < 0.02


def test_large_random_weights_start_well_above_ln_k():
    x, y = batch(200, 4)
    big = Sequential([Dense(3, 4, seed=0, init_scale=3.0), Softmax()])
    big.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=0.1))
    assert big.evaluate(x, y) > np.log(4) + 0.5  # confidently wrong on average


# ------------------------------------------------------------- gradient checks
@pytest.mark.parametrize("hidden", [Tanh, ReLU])
@pytest.mark.parametrize("k", [2, 4, 7])
def test_fused_path_gradients_match_numerical(hidden, k):
    x, y = batch(9, k, seed=k)
    assert classifier(hidden, k)._is_fused()
    check_model(classifier(hidden, k, seed=k), x, y, tol=1e-6).assert_passed()


def test_single_sample_and_single_layer_softmax_models():
    x, y = batch(1, 3)
    m = Sequential([scaled_dense(3, 3, 5), Softmax()])
    m.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=0.1))
    check_model(m, x, y, tol=1e-6).assert_passed()


def test_softmax_with_a_different_loss_uses_the_normal_chain_rule():
    """Softmax.backward (full Jacobian) inside a model, trained against MSE."""
    x, y = batch(8, 4)
    m = classifier(Tanh, loss=MSE())
    assert not m._is_fused()
    check_model(m, x, y, tol=1e-6).assert_passed()


@pytest.mark.parametrize("seed", range(5))
def test_fused_and_unfused_parameter_gradients_agree(seed):
    x, y = batch(10, 4, seed)
    fused = classifier(Tanh, seed=seed)
    fused.loss.forward(fused.forward(x), y)
    fused.backward_from_loss()
    g_fused = [g.copy() for _, g in fused.params()]

    unfused = classifier(Tanh, seed=seed)
    unfused.loss.forward(unfused.forward(x), y)
    unfused.backward(unfused.loss.backward())  # plain chain rule through Softmax.backward
    for a, (_, b) in zip(g_fused, unfused.params()):
        assert np.allclose(a, b, atol=1e-9)


# --------------------------------------------------------------------- training
def test_training_with_fused_softmax_cross_entropy_learns_a_rule():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 3))
    labels = (x @ rng.normal(size=(3, 4))).argmax(axis=1)  # a learnable (linear) class rule
    y = one_hot(labels, 4)
    m = classifier(Tanh, seed=3)
    h = m.fit(x, y, epochs=300, verbose=0)
    assert h["loss"][-1] < h["loss"][0] * 0.5
    assert np.mean(m.predict(x).argmax(axis=1) == labels) > 0.85


def test_model_that_starts_confidently_wrong_still_learns():
    """The practical payoff of fusion: saturated softmax outputs still produce a useful gradient."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=60)
    x = np.eye(3)[labels] * 2.0 + rng.normal(scale=0.1, size=(60, 3))
    y = one_hot(labels, 3)
    m = Sequential([Dense(3, 3, seed=0)])
    m.layers[0].W = -50.0 * np.eye(3)  # strongly anti-correlated: confidently wrong
    m.layers.append(Softmax())
    m.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=0.2))
    start = m.evaluate(x, y)
    m.fit(x, y, epochs=2000, verbose=0)
    assert start > 25  # the loss is capped at -ln(eps) ~ 27.6 by probability clipping
    assert np.mean(m.predict(x).argmax(1) == labels) > 0.95


def test_without_fusion_the_same_confidently_wrong_model_never_recovers():
    """Contrast for the test above: the plain chain rule through softmax gives ~0 gradient here."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=60)
    x = np.eye(3)[labels] * 2.0 + rng.normal(scale=0.1, size=(60, 3))
    y = one_hot(labels, 3)
    m = Sequential([Dense(3, 3, seed=0), Softmax()])
    m.layers[0].W = -50.0 * np.eye(3)
    m.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=0.2))
    for _ in range(500):
        m.loss.forward(m.forward(x), y)
        m.backward(m.loss.backward())  # unfused
        m.optimizer.step(m.params())
    assert np.mean(m.predict(x).argmax(axis=1) == labels) == 0.0  # still 0% accurate
