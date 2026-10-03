"""Day 5: gradient-check every layer, activation, loss, and model in the library."""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Tanh
from nn.gradcheck import (
    GradCheckResult,
    check_layer,
    check_loss,
    check_model,
    numerical_gradient,
    relative_error,
)
from nn.layers import Dense
from nn.losses import MSE, BinaryCrossEntropy
from nn.model import Sequential
from nn.optimizers import SGD

TOL = 1e-6


def away_from_zero(x, margin=0.05):
    """Push values away from 0 so ReLU's kink never sits inside the finite-difference window."""
    return np.where(np.abs(x) < margin, margin * 2, x)


def scaled_dense(n_in, n_out, seed):
    layer = Dense(n_in, n_out, seed=seed)
    rng = np.random.default_rng(seed + 100)
    layer.W = rng.normal(scale=0.7, size=layer.W.shape)
    layer.b = rng.normal(scale=0.2, size=layer.b.shape)
    layer.dW, layer.db = np.zeros_like(layer.W), np.zeros_like(layer.b)
    return layer


# ------------------------------------------------------- checker internals
def test_numerical_gradient_of_known_function():
    x = np.array([[1.0, -2.0], [3.0, 0.5]])
    g = numerical_gradient(lambda: float(np.sum(x**3)), x)
    assert np.allclose(g, 3 * x**2, atol=1e-6)


def test_numerical_gradient_restores_input():
    x = np.random.default_rng(0).normal(size=(3, 3))
    before = x.copy()
    numerical_gradient(lambda: float(np.sum(x**2)), x)
    assert np.array_equal(x, before)


def test_relative_error_properties():
    a = np.array([1.0, 2.0, 3.0])
    assert relative_error(a, a) == 0.0
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0
    assert relative_error(a, 1.1 * a) > 0.04
    with pytest.raises(ValueError):
        relative_error(np.zeros(2), np.zeros(3))


def test_result_reports_pass_and_fail():
    ok = GradCheckResult({"a": 1e-9}, tol=1e-6)
    bad = GradCheckResult({"a": 1e-9, "b": 1e-2}, tol=1e-6)
    assert ok.passed and not bad.passed
    assert bad.max_error == 1e-2
    ok.assert_passed()
    with pytest.raises(AssertionError):
        bad.assert_passed()


# --------------------------------------------- the checker catches real bugs
class DenseWrongWeightGrad(Dense):
    """Deliberately buggy: forgets the transpose scale (gradient 2x too large)."""

    def backward(self, grad_output):
        dx = super().backward(grad_output)
        self.dW *= 2.0
        return dx


class DenseWrongBiasGrad(Dense):
    """Deliberately buggy: averages the bias gradient instead of summing."""

    def backward(self, grad_output):
        dx = super().backward(grad_output)
        self.db /= grad_output.shape[0]
        return dx


class DenseWrongInputGrad(Dense):
    """Deliberately buggy: uses W instead of W^T when it happens to be square."""

    def backward(self, grad_output):
        super().backward(grad_output)
        return grad_output @ self.W


@pytest.mark.parametrize("buggy", [DenseWrongWeightGrad, DenseWrongBiasGrad, DenseWrongInputGrad])
def test_checker_detects_broken_backward(buggy):
    layer = buggy(4, 4, seed=0)
    layer.W = np.random.default_rng(1).normal(scale=0.7, size=(4, 4))
    x = np.random.default_rng(2).normal(size=(6, 4))
    assert not check_layer(layer, x, tol=TOL).passed


def test_model_checker_requires_compiled_model():
    with pytest.raises(RuntimeError):
        check_model(Sequential([Dense(2, 1, seed=0)]), np.zeros((2, 2)), np.zeros((2, 1)))


# ------------------------------------------------------------------ layers
@pytest.mark.parametrize(
    "n, n_in, n_out",
    [(1, 1, 1), (1, 4, 3), (5, 1, 4), (8, 5, 3), (16, 10, 10), (3, 2, 7)],
)
def test_dense_gradients(n, n_in, n_out):
    x = np.random.default_rng(n + n_in).normal(size=(n, n_in))
    check_layer(scaled_dense(n_in, n_out, seed=n_out), x, tol=TOL).assert_passed()


@pytest.mark.parametrize("act_cls", [ReLU, Sigmoid, Tanh])
@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (6, 1), (7, 4), (10, 10)])
def test_activation_gradients(act_cls, shape):
    x = away_from_zero(np.random.default_rng(sum(shape)).normal(scale=1.5, size=shape))
    check_layer(act_cls(), x, tol=TOL).assert_passed()


def test_sigmoid_gradient_in_saturated_region():
    x = np.array([[-30.0, -10.0, 10.0, 30.0]])
    check_layer(Sigmoid(), x, tol=TOL).assert_passed()


def test_tanh_gradient_in_saturated_region():
    x = np.array([[-15.0, -5.0, 5.0, 15.0]])
    check_layer(Tanh(), x, tol=TOL).assert_passed()


# ------------------------------------------------------------------ losses
@pytest.mark.parametrize("shape", [(1, 1), (5, 1), (4, 3), (10, 2)])
def test_mse_gradient(shape):
    rng = np.random.default_rng(sum(shape))
    check_loss(MSE(), rng.normal(size=shape), rng.normal(size=shape), tol=TOL).assert_passed()


@pytest.mark.parametrize("shape", [(1, 1), (6, 1), (5, 3)])
def test_bce_gradient_with_hard_labels(shape):
    rng = np.random.default_rng(sum(shape))
    p = rng.uniform(0.05, 0.95, size=shape)
    y = (rng.random(shape) > 0.5).astype(float)
    check_loss(BinaryCrossEntropy(), p, y, tol=TOL).assert_passed()


def test_bce_gradient_with_soft_labels():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, size=(8, 2))
    y = rng.uniform(0, 1, size=(8, 2))
    check_loss(BinaryCrossEntropy(), p, y, tol=TOL).assert_passed()


def test_bce_gradient_is_not_checkable_in_the_clipped_region():
    """Documented limitation: clipping makes the loss flat, so finite differences read ~0."""
    p = np.array([[1e-15]])
    y = np.array([[1.0]])
    assert not check_loss(BinaryCrossEntropy(), p, y, tol=TOL).passed


# ------------------------------------------------------------ whole models
def build(hidden_cls, loss_cls, depth, seed):
    layers, width = [], 5
    layers.append(scaled_dense(3, width, seed))
    for d in range(depth):
        layers += [hidden_cls(), scaled_dense(width, width, seed + d + 1)]
    layers += [hidden_cls(), scaled_dense(width, 1, seed + 50), Sigmoid()]
    model = Sequential(layers)
    model.compile(loss=loss_cls(), optimizer=SGD(lr=0.1))
    return model


@pytest.mark.parametrize("hidden_cls", [ReLU, Tanh, Sigmoid])
@pytest.mark.parametrize("loss_cls", [MSE, BinaryCrossEntropy])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_whole_model_gradients(hidden_cls, loss_cls, depth):
    rng = np.random.default_rng(depth + 11)
    x = rng.normal(size=(9, 3))
    y = (rng.random((9, 1)) > 0.5).astype(float)
    result = check_model(build(hidden_cls, loss_cls, depth, seed=depth), x, y, tol=TOL)
    result.assert_passed()


def test_single_sample_model_gradients():
    rng = np.random.default_rng(5)
    model = build(Tanh, MSE, 1, seed=2)
    check_model(model, rng.normal(size=(1, 3)), rng.normal(size=(1, 1)), tol=TOL).assert_passed()


def test_multi_output_regression_gradients():
    rng = np.random.default_rng(6)
    model = Sequential([scaled_dense(3, 6, 1), Tanh(), scaled_dense(6, 4, 2)])
    model.compile(loss=MSE(), optimizer=SGD(lr=0.1))
    check_model(model, rng.normal(size=(7, 3)), rng.normal(size=(7, 4)), tol=TOL).assert_passed()


@pytest.mark.parametrize("seed", range(10))
def test_random_models_over_many_seeds(seed):
    rng = np.random.default_rng(seed)
    model = build(Tanh, BinaryCrossEntropy, depth=2, seed=seed)
    x = rng.normal(size=(6, 3))
    y = (rng.random((6, 1)) > 0.5).astype(float)
    check_model(model, x, y, tol=TOL).assert_passed()
