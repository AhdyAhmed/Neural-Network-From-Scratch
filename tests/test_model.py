"""Day 4 tests: Sequential (forward / backward / fit / predict / evaluate)."""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Tanh
from nn.layers import Dense
from nn.losses import MSE, BinaryCrossEntropy
from nn.model import Sequential
from nn.optimizers import SGD


def line_data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 1))
    return x, 2 * x + 1


def make_linear():
    m = Sequential([Dense(1, 1, seed=0)])
    m.compile(loss=MSE(), optimizer=SGD(lr=0.3))
    return m


# --------------------------------------------------------------- structure
def test_empty_model_rejected():
    with pytest.raises(ValueError):
        Sequential([])


def test_forward_shape_and_predict_matches_forward():
    m = Sequential([Dense(3, 5, seed=0), ReLU(), Dense(5, 2, seed=1)])
    x = np.random.default_rng(0).normal(size=(7, 3))
    assert m.forward(x).shape == (7, 2)
    assert np.allclose(m.predict(x), m.forward(x, training=False))


def test_params_collects_every_layer():
    m = Sequential([Dense(3, 4, seed=0), Tanh(), Dense(4, 1, seed=1)])
    assert len(m.params()) == 4  # W, b, W, b


def test_summary_counts_parameters():
    m = Sequential([Dense(3, 4, seed=0), Tanh(), Dense(4, 1, seed=1)])
    assert "Total" in m.summary() and str(3 * 4 + 4 + 4 * 1 + 1) in m.summary()


def test_backward_returns_input_gradient_shape():
    m = Sequential([Dense(3, 4, seed=0), Tanh(), Dense(4, 1, seed=1)])
    x = np.random.default_rng(0).normal(size=(6, 3))
    m.forward(x)
    assert m.backward(np.ones((6, 1))).shape == (6, 3)


# --------------------------------------------------------------- validation
def test_fit_requires_compile():
    with pytest.raises(RuntimeError):
        Sequential([Dense(1, 1, seed=0)]).fit(*line_data(), epochs=1)


def test_evaluate_requires_compile():
    with pytest.raises(RuntimeError):
        Sequential([Dense(1, 1, seed=0)]).evaluate(*line_data())


def test_fit_rejects_mismatched_samples():
    with pytest.raises(ValueError):
        make_linear().fit(np.zeros((5, 1)), np.zeros((4, 1)), epochs=1)


def test_fit_rejects_non_2d_x():
    with pytest.raises(ValueError):
        make_linear().fit(np.zeros(5), np.zeros((5, 1)), epochs=1)


def test_fit_rejects_non_positive_epochs():
    with pytest.raises(ValueError):
        make_linear().fit(*line_data(), epochs=0)


def test_flat_target_vector_is_accepted():
    x, y = line_data()
    make_linear().fit(x, y.ravel(), epochs=2, verbose=0)


# ----------------------------------------------------------------- training
def test_history_has_one_entry_per_epoch():
    h = make_linear().fit(*line_data(), epochs=25, verbose=0)
    assert list(h) == ["loss"] and len(h["loss"]) == 25


def test_history_includes_validation_loss():
    x, y = line_data()
    h = make_linear().fit(x, y, epochs=10, validation_data=line_data(seed=1), verbose=0)
    assert len(h["val_loss"]) == 10


def test_loss_decreases_during_training():
    h = make_linear().fit(*line_data(), epochs=100, verbose=0)
    assert h["loss"][-1] < h["loss"][0] * 0.01
    assert all(b <= a + 1e-12 for a, b in zip(h["loss"], h["loss"][1:]))  # monotone for small lr


def test_learns_y_equals_2x_plus_1():
    """The Day 4 milestone: recover slope 2 and intercept 1."""
    m = make_linear()
    m.fit(*line_data(), epochs=300, verbose=0)
    dense = m.layers[0]
    assert np.isclose(dense.W[0, 0], 2.0, atol=1e-2)
    assert np.isclose(dense.b[0, 0], 1.0, atol=1e-2)


def test_hidden_layer_network_also_fits_the_line():
    m = Sequential([Dense(1, 8, seed=0), Tanh(), Dense(8, 1, seed=1)])
    m.compile(loss=MSE(), optimizer=SGD(lr=0.2))
    x, y = line_data()
    for layer in m.layers:  # default init is tiny; use moderate weights so learning is quick
        if isinstance(layer, Dense):
            layer.W = np.random.default_rng(3).normal(scale=0.5, size=layer.W.shape)
    m.fit(x, y, epochs=800, verbose=0)
    assert m.evaluate(x, y) < 1e-3


def test_binary_classification_learns_separable_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = (x[:, :1] + x[:, 1:] > 0).astype(float)
    m = Sequential([Dense(2, 1, seed=0), Sigmoid()])
    m.compile(loss=BinaryCrossEntropy(), optimizer=SGD(lr=1.0))
    m.fit(x, y, epochs=300, verbose=0)
    acc = np.mean((m.predict(x) > 0.5) == y)
    assert acc > 0.95


def test_predict_does_not_change_parameters():
    m = make_linear()
    x, _ = line_data()
    before = [v.copy() for v, _ in m.params()]
    m.predict(x)
    assert all(np.array_equal(b, v) for b, (v, _) in zip(before, m.params()))


def test_evaluate_matches_loss_of_predictions_and_is_pure():
    m = make_linear()
    x, y = line_data()
    before = [v.copy() for v, _ in m.params()]
    assert np.isclose(m.evaluate(x, y), np.mean((m.predict(x) - y) ** 2))
    assert all(np.array_equal(b, v) for b, (v, _) in zip(before, m.params()))


def test_training_is_deterministic_with_seeds():
    h1 = make_linear().fit(*line_data(), epochs=20, verbose=0)
    h2 = make_linear().fit(*line_data(), epochs=20, verbose=0)
    assert h1["loss"] == h2["loss"]


def test_verbose_prints_progress(capsys):
    make_linear().fit(*line_data(), epochs=10, verbose=1)
    out = capsys.readouterr().out
    assert "epoch" in out and "loss" in out


def test_silent_when_verbose_zero(capsys):
    make_linear().fit(*line_data(), epochs=10, verbose=0)
    assert capsys.readouterr().out == ""
