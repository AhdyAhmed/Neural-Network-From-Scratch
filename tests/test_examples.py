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


# ------------------------------------------------- Day 7: three-class blobs
from examples.three_class_blobs import NUM_CLASSES, accuracy as blob_accuracy, build_model, make_blobs, train as train_blobs
from nn.utils import one_hot as _one_hot


@pytest.mark.parametrize("seed", range(5))
def test_three_class_blobs_reach_high_accuracy(seed):
    model, history, (x_tr, l_tr, x_te, l_te) = train_blobs(seed=seed)
    assert blob_accuracy(model, x_tr, l_tr) > 0.93
    assert blob_accuracy(model, x_te, l_te) > 0.90
    assert history["loss"][-1] < 0.3 * history["loss"][0]


def test_three_class_predictions_are_probability_distributions():
    model, _, (_, _, x_te, _) = train_blobs(seed=0)
    p = model.predict(x_te)
    assert np.allclose(p.sum(axis=1), 1.0) and np.all(p >= 0)


def test_three_class_untrained_loss_is_ln_3():
    x, labels = make_blobs()
    loss = build_model(init_scale=0.01).evaluate(x, _one_hot(labels, NUM_CLASSES))
    assert abs(loss - np.log(3)) < 0.01


# ----------------------------------------------- Day 9: initialization demo
from examples.init_comparison import CONFIGS as INIT_CONFIGS, build_mlp, make_optimizer, signal_through_depth
from nn.activations import ReLU as _ReLU, Tanh as _Tanh
from nn.optimizers import SGD as _SGD, Momentum as _Momentum


def test_build_mlp_applies_he_before_relu_and_xavier_before_softmax():
    m = build_mlp([300, 300, 300, 10], _ReLU, "he", _SGD(0.1), seed=0)
    dense = [layer for layer in m.layers if hasattr(layer, "W")]
    assert np.isclose(dense[0].W.std(), np.sqrt(2 / 300), rtol=0.05)  # He (feeds ReLU)
    assert np.isclose(dense[1].W.std(), np.sqrt(2 / 300), rtol=0.05)
    assert np.isclose(dense[2].W.std(), np.sqrt(2 / 310), rtol=0.1)   # Xavier (feeds softmax)


def test_build_mlp_rejects_unknown_init():
    with pytest.raises(ValueError):
        build_mlp([4, 4, 2], _ReLU, "kaiming", _SGD(0.1))


def test_momentum_configs_use_the_same_effective_step_as_their_sgd_partner():
    sgd, mom = INIT_CONFIGS["He + SGD"], INIT_CONFIGS["He + Momentum"]
    assert np.isclose(mom["lr"] / (1 - 0.9), sgd["lr"])
    tuned_sgd, tuned_mom = INIT_CONFIGS["He + SGD (lr 0.2, tuned)"], INIT_CONFIGS["He + Momentum (lr 0.02, tuned)"]
    assert np.isclose(tuned_mom["lr"] / (1 - 0.9), tuned_sgd["lr"])
    assert isinstance(make_optimizer(mom), _Momentum) and isinstance(make_optimizer(sgd), _SGD)


def test_signal_through_depth_shows_the_expected_pattern():
    r = signal_through_depth()
    ratio = lambda label: r[label][0][-1] / r[label][0][0]  # output std at layer 10 / layer 1
    assert ratio("ReLU, N(0, 0.01²)") < 1e-8     # vanishes
    assert ratio("ReLU, N(0, 1²)") > 1e6         # explodes
    assert 0.1 < ratio("ReLU, He") < 10          # preserved
    assert ratio("ReLU, Xavier") < 0.1           # shrinks (ReLU halves the variance each layer)
    assert r["ReLU, He"][2] == [] and r["tanh, Xavier"][2] == []          # healthy: no warnings
    assert any("vanishing" in w for w in r["ReLU, N(0, 0.01²)"][2])
    assert any("exploding" in w for w in r["ReLU, N(0, 1²)"][2])
    assert any("saturated" in w for w in r["tanh, N(0, 1²)"][2])


def test_flat_activation_scale_does_not_mean_healthy_tanh_with_large_weights():
    """tanh with N(0,1) weights keeps a constant output std only because every unit is pinned at +/-1."""
    r = signal_through_depth()
    act_std, grad_rms, warnings, _ = r["tanh, N(0, 1²)"]
    assert max(act_std) / min(act_std) < 1.5          # looks stable...
    assert grad_rms[0] / grad_rms[-1] > 100            # ...but gradients explode toward the input
    assert any("saturated" in w for w in warnings)
