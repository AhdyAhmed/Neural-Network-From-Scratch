"""Day 6: the example problems are solved (these tests double as regression tests).

XOR needs a hidden layer; the sine fit must reach the noise floor on held-out data.
"""

import numpy as np
import pytest

from examples.regression import NOISE_STD, error_vs_true_curve, make_data, train_sine
from examples.xor import X, Y, accuracy, train_linear_baseline, train_xor


# ------------------------------------------------------------------ XOR
@pytest.mark.parametrize("seed", range(10))
def test_xor_is_solved_for_every_seed(seed):
    model, _ = train_xor(seed=seed)
    assert accuracy(model) == 1.0


def test_xor_probabilities_are_confident():
    model, _ = train_xor(seed=0)
    p = model.predict(X)
    assert np.all(np.abs(p - Y) < 0.05)


def test_xor_loss_decreases_by_orders_of_magnitude():
    _, history = train_xor(seed=0)
    assert history["loss"][-1] < history["loss"][0] * 0.01
    assert history["loss"][-1] < 0.01


@pytest.mark.parametrize("seed", range(5))
def test_linear_model_cannot_solve_xor(seed):
    """No straight line separates XOR, so accuracy can never reach 100%."""
    model, history = train_linear_baseline(seed=seed)
    assert accuracy(model) <= 0.75
    assert history["loss"][-1] > 0.6  # stuck near ln(2) = 0.693


def test_xor_dataset_is_correct():
    assert np.array_equal(Y.ravel(), np.logical_xor(X[:, 0], X[:, 1]).astype(float))


# ----------------------------------------------------------- sine regression
def test_sine_data_split_sizes_and_noise():
    x_tr, y_tr, x_val, y_val = make_data()
    assert len(x_tr) == 160 and len(x_val) == 40
    resid = np.concatenate([y_tr - np.sin(x_tr), y_val - np.sin(x_val)])
    assert np.isclose(resid.std(), NOISE_STD, rtol=0.2)


def test_sine_fit_reaches_noise_floor_on_validation_data():
    model, history, _ = train_sine()
    assert history["val_loss"][-1] < 2 * NOISE_STD**2  # ~noise floor, not memorising
    assert history["loss"][-1] < 2 * NOISE_STD**2


def test_sine_fit_matches_the_true_curve():
    model, _, _ = train_sine()
    assert error_vs_true_curve(model) < 0.002  # far below the noise variance (0.01)


def test_sine_training_loss_is_stable():
    """No oscillation at the chosen learning rate (a too-large lr gave spikes)."""
    _, history, _ = train_sine()
    loss = np.array(history["loss"])
    assert not np.any(loss[1:] > 1.5 * loss[:-1])
    assert np.abs(np.diff(history["val_loss"][-500:])).max() < 1e-6
