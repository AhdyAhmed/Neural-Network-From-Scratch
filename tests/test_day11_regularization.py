"""Day 11: L2 regularization and inverted dropout."""
import numpy as np
import pytest

from nn.activations import ReLU
from nn.layers import Dense, Dropout
from nn.losses import MSE
from nn.model import Sequential
from nn.optimizers import SGD


def test_dropout_training_uses_inverted_scaling_and_backward_mask():
    layer = Dropout(rate=0.5, seed=2)
    x = np.ones((100, 20))
    out = layer.forward(x, training=True)
    assert set(np.unique(out)).issubset({0.0, 2.0})
    assert np.isclose(out.mean(), 1.0, atol=0.1)
    grad = layer.backward(np.ones_like(x))
    assert np.array_equal(grad, out)


def test_dropout_is_identity_during_inference_and_rate_zero():
    x = np.arange(12.0).reshape(3, 4)
    layer = Dropout(rate=0.75, seed=0)
    assert np.array_equal(layer.forward(x, training=False), x)
    assert np.array_equal(layer.backward(np.ones_like(x)), np.ones_like(x))
    zero = Dropout(rate=0.0, seed=0)
    assert np.array_equal(zero.forward(x, training=True), x)


@pytest.mark.parametrize("rate", [-0.1, 1.0, 2.0])
def test_dropout_rejects_invalid_rate(rate):
    with pytest.raises(ValueError):
        Dropout(rate=rate)


def test_dropout_requires_forward_and_checks_gradient_shape():
    layer = Dropout(0.2)
    with pytest.raises(RuntimeError):
        layer.backward(np.ones((2, 3)))
    layer.forward(np.ones((2, 3)))
    with pytest.raises(ValueError):
        layer.backward(np.ones((3, 2)))


def test_dropout_fixed_mask_gradient_matches_finite_difference():
    # Reset the RNG for each function evaluation so finite differences see the same mask.
    x = np.array([[0.4, -0.8, 1.2], [0.7, 0.3, -0.2]])
    upstream = np.array([[0.3, -0.4, 0.2], [0.1, 0.5, -0.6]])
    layer = Dropout(rate=0.5, seed=17)
    layer.forward(x, training=True)
    fixed_mask = layer._mask.copy()
    analytic = upstream * fixed_mask

    def objective():
        layer._mask = fixed_mask
        return float(np.sum((x * layer._mask) * upstream))

    numeric = np.zeros_like(x)
    eps = 1e-6
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        plus = objective()
        x[idx] = old - eps
        minus = objective()
        x[idx] = old
        numeric[idx] = (plus - minus) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-8)


def test_l2_objective_and_weight_gradient_match_finite_difference():
    model = Sequential([Dense(2, 3, seed=0, init_scale=0.3), ReLU(), Dense(3, 1, seed=1, init_scale=0.3)])
    model.compile(MSE(), SGD(lr=0.01), l2=0.2)
    x = np.array([[0.2, -0.4], [0.8, 0.1], [-0.5, 0.7]])
    y = np.array([[0.3], [-0.2], [0.5]])
    pred = model.forward(x, training=False)
    data_loss = model.loss.forward(pred, y)
    model.backward_from_loss()
    model._add_regularization_gradients()
    analytic = [layer.dW.copy() for layer in model.layers if isinstance(layer, Dense)]
    assert np.isclose(model.evaluate(x, y), data_loss + model._regularization_penalty())

    for layer, grad in zip((l for l in model.layers if isinstance(l, Dense)), analytic):
        numeric = np.zeros_like(layer.W)
        eps = 1e-6
        for idx in np.ndindex(layer.W.shape):
            old = layer.W[idx]
            layer.W[idx] = old + eps
            plus = model.loss.forward(model.forward(x, training=False), y) + model._regularization_penalty()
            layer.W[idx] = old - eps
            minus = model.loss.forward(model.forward(x, training=False), y) + model._regularization_penalty()
            layer.W[idx] = old
            numeric[idx] = (plus - minus) / (2 * eps)
        assert np.allclose(grad, numeric, atol=1e-6, rtol=1e-5)


def test_l2_does_not_regularize_biases_and_zero_l2_preserves_behavior():
    layer = Dense(2, 1, seed=0, init_scale=0.5)
    model = Sequential([layer])
    model.compile(MSE(), SGD(lr=0.01), l2=0.3)
    x, y = np.array([[1.0, 2.0]]), np.array([[0.5]])
    model.forward(x)
    model.loss.forward(model.forward(x), y)
    model.backward_from_loss()
    bias_grad_before = layer.db.copy()
    model._add_regularization_gradients()
    assert np.array_equal(layer.db, bias_grad_before)
    assert np.allclose(layer.dW, layer.dW)  # gradients remain finite

    with pytest.raises(ValueError):
        model.compile(MSE(), SGD(), l2=-1)
    with pytest.raises(ValueError):
        model.compile(MSE(), SGD(), l2=float("inf"))
