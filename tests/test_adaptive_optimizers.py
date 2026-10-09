"""Day 10: RMSProp and Adam, plus a convergence check shared by all four optimizers."""

import numpy as np
import pytest

from nn.optimizers import SGD, Adam, Momentum, RMSProp

# f(w) = 0.5 * sum(a_i * (w_i - c_i)^2): an ill-conditioned quadratic (curvatures 1 and 25).
A = np.array([1.0, 25.0])
C = np.array([3.0, -2.0])


def distance_after(opt, steps, a=A, c=C, w0=(-4.0, 4.0)):
    w = np.array(w0, dtype=float)
    for _ in range(steps):
        opt.step([(w, a * (w - c))])
    return np.linalg.norm(w - c)


# ------------------------------------------- all four converge on the quadratic
@pytest.mark.parametrize(
    "make",
    [lambda: SGD(0.03), lambda: Momentum(0.01, 0.9), lambda: RMSProp(0.01), lambda: Adam(0.1)],
    ids=["SGD", "Momentum", "RMSProp", "Adam"],
)
def test_every_optimizer_converges_on_the_quadratic(make):
    assert distance_after(make(), 1000) < 1e-6


# ---------------------------------------------------------------- RMSProp
def test_rmsprop_first_step_is_lr_over_sqrt_one_minus_rho_times_sign():
    for g in (0.5, -0.5, 100.0, 0.001):
        w = np.array([1.0])
        RMSProp(lr=0.1, rho=0.9).step([(w, np.array([g]))])
        assert np.isclose(1.0 - w[0], np.sign(g) * 0.1 / np.sqrt(0.1), rtol=1e-4)


def test_rmsprop_two_steps_match_hand_calculation():
    lr, rho, eps, g = 0.1, 0.9, 1e-8, 0.5
    w = np.array([1.0])
    opt = RMSProp(lr, rho, eps)
    opt.step([(w, np.array([g]))])
    s1 = (1 - rho) * g**2
    expected = 1.0 - lr * g / (np.sqrt(s1) + eps)
    assert np.isclose(w[0], expected)
    opt.step([(w, np.array([g]))])
    s2 = rho * s1 + (1 - rho) * g**2
    expected -= lr * g / (np.sqrt(s2) + eps)
    assert np.isclose(w[0], expected)


def test_rmsprop_with_a_steady_gradient_settles_at_lr_per_step():
    w, g = np.zeros(1), np.array([4.0])
    opt = RMSProp(lr=0.01, rho=0.9)
    for _ in range(200):
        before = w[0]
        opt.step([(w, g)])
    assert np.isclose(before - w[0], 0.01, rtol=1e-3)  # sqrt(s) -> |g|, so the step -> lr


# -------------------------------------------------------------------- Adam
def test_adam_first_step_is_exactly_lr_times_sign_thanks_to_bias_correction():
    for g in (0.5, -0.5, 123.0, 1e-3):
        w = np.array([1.0])
        Adam(lr=0.1).step([(w, np.array([g]))])
        assert np.isclose(1.0 - w[0], np.sign(g) * 0.1, rtol=1e-5)


def test_adam_two_steps_match_hand_calculation():
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    g1, g2 = 0.5, -0.2
    w = np.array([1.0])
    opt = Adam(lr, b1, b2, eps)

    opt.step([(w, np.array([g1]))])
    m = (1 - b1) * g1
    v = (1 - b2) * g1**2
    expected = 1.0 - lr * (m / (1 - b1)) / (np.sqrt(v / (1 - b2)) + eps)
    assert np.isclose(w[0], expected)

    opt.step([(w, np.array([g2]))])
    m = b1 * m + (1 - b1) * g2
    v = b2 * v + (1 - b2) * g2**2
    expected -= lr * (m / (1 - b1**2)) / (np.sqrt(v / (1 - b2**2)) + eps)
    assert np.isclose(w[0], expected)


def test_without_bias_correction_the_first_step_would_be_about_3x_too_big():
    """Documents what the correction fixes: raw m = 0.1 g and raw v = 0.001 g^2 give m/sqrt(v) = 3.16."""
    g = 0.5
    uncorrected = (0.1 * g) / np.sqrt(0.001 * g**2)
    assert np.isclose(uncorrected, 3.162, atol=1e-3)
    w = np.array([0.0])
    Adam(lr=1.0).step([(w, np.array([g]))])
    assert np.isclose(-w[0], 1.0, rtol=1e-5)  # corrected step is 1.0 * lr, not 3.16 * lr


def test_adam_step_size_is_bounded_by_roughly_lr():
    rng = np.random.default_rng(0)
    w, opt = np.zeros(50), Adam(lr=0.01)
    for _ in range(100):
        before = w.copy()
        opt.step([(w, rng.normal(scale=10 ** rng.uniform(-3, 3), size=50))])
        assert np.abs(w - before).max() < 0.01 * 3.2  # a few times lr at the very most


def test_adam_and_rmsprop_ignore_the_overall_gradient_scale_but_sgd_does_not():
    rng = np.random.default_rng(1)
    grads = [rng.normal(size=3) for _ in range(40)]
    for make, invariant in [(lambda: Adam(0.01), True), (lambda: RMSProp(0.01), True), (lambda: SGD(0.01), False)]:
        w1, w2, o1, o2 = np.ones(3), np.ones(3), make(), make()
        for g in grads:
            o1.step([(w1, g)])
            o2.step([(w2, 1000.0 * g)])
        assert (np.abs(w1 - w2).max() < 1e-6) == invariant


def test_adaptive_methods_equalise_coordinates_with_very_different_gradient_scales():
    """First step: SGD moves the big-gradient coordinate 10,000x further; Adam / RMSProp move both equally."""
    g = np.array([1e-3, 10.0])
    for opt in (Adam(0.1), RMSProp(0.1)):
        w = np.zeros(2)
        opt.step([(w, g)])
        assert abs(w[0] / w[1] - 1.0) < 1e-3
    w = np.zeros(2)
    SGD(0.1).step([(w, g)])
    assert abs(w[1] / w[0]) > 1e3


def test_adaptive_methods_beat_sgd_on_a_badly_scaled_quadratic():
    """Curvatures 1 and 10,000: SGD's step is capped by the steep direction (lr < 2/10,000),
    which leaves the shallow direction crawling."""
    a, c = np.array([1.0, 1e4]), np.array([3.0, -2.0])
    sgd = distance_after(SGD(1e-4), 500, a, c)
    adam = distance_after(Adam(0.1), 500, a, c)
    rms = distance_after(RMSProp(0.05), 500, a, c)
    assert adam < 0.01 * sgd and rms < 0.01 * sgd


# --------------------------------------------------------- shared behaviour
@pytest.mark.parametrize("make", [lambda: RMSProp(0.01), lambda: Adam(0.01), lambda: Momentum(0.01)])
def test_update_in_place_zero_gradient_and_step_counter(make):
    opt = make()
    w = np.array([1.0, 2.0])
    ref = w
    opt.step([(w, np.zeros(2))])
    assert ref is w and np.allclose(w, [1.0, 2.0]) and np.all(np.isfinite(w))
    opt.step([(w, np.ones(2))])
    assert opt.t == 2


@pytest.mark.parametrize("make", [lambda: RMSProp(0.1), lambda: Adam(0.1)])
def test_reset_restarts_the_optimizer(make):
    opt = make()
    first = np.array([1.0])
    opt.step([(first, np.array([0.5]))])
    one_step_result = first.copy()
    for _ in range(5):
        opt.step([(first, np.array([0.5]))])
    opt.reset()
    again = np.array([1.0])
    opt.step([(again, np.array([0.5]))])
    assert np.isclose(again[0], one_step_result[0]) and opt.t == 1


@pytest.mark.parametrize("make", [lambda: RMSProp(0.1), lambda: Adam(0.1)])
def test_changing_parameter_set_without_reset_is_an_error(make):
    opt = make()
    opt.step([(np.zeros(2), np.ones(2))])
    with pytest.raises(ValueError, match="reset"):
        opt.step([(np.zeros(3), np.ones(3))])
    with pytest.raises(ValueError, match="reset"):
        opt.step([(np.zeros(2), np.ones(2)), (np.zeros(2), np.ones(2))])


def test_multiple_parameters_get_independent_state():
    w1, w2 = np.zeros(2), np.zeros((2, 3))
    opt = Adam(0.1)
    opt.step([(w1, np.array([1.0, -1.0])), (w2, np.full((2, 3), 5.0))])
    assert np.allclose(w1, [-0.1, 0.1], atol=1e-6) and np.allclose(w2, -0.1, atol=1e-6)


@pytest.mark.parametrize("cls", [RMSProp, Adam])
@pytest.mark.parametrize("kwargs", [{"lr": 0}, {"lr": -1}, {"eps": 0}, {"eps": -1e-8}])
def test_rejects_invalid_lr_or_eps(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


@pytest.mark.parametrize("kwargs", [{"rho": -0.1}, {"rho": 1.0}])
def test_rmsprop_rejects_invalid_rho(kwargs):
    with pytest.raises(ValueError):
        RMSProp(**kwargs)


@pytest.mark.parametrize("kwargs", [{"beta1": 1.0}, {"beta1": -0.1}, {"beta2": 1.0}, {"beta2": -0.5}])
def test_adam_rejects_invalid_betas(kwargs):
    with pytest.raises(ValueError):
        Adam(**kwargs)


def test_repr():
    assert repr(RMSProp(0.01, 0.9)) == "RMSProp(lr=0.01, rho=0.9)"
    assert repr(Adam(0.001)) == "Adam(lr=0.001, beta1=0.9, beta2=0.999)"


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("make", [lambda: RMSProp(0.02), lambda: Adam(0.05)], ids=["RMSProp", "Adam"])
def test_train_a_linear_model(make):
    from nn.layers import Dense
    from nn.losses import MSE
    from nn.model import Sequential

    x = np.random.default_rng(0).uniform(-1, 1, size=(64, 1))
    m = Sequential([Dense(1, 1, seed=0)])
    m.compile(loss=MSE(), optimizer=make())
    m.fit(x, 2 * x + 1, epochs=300, verbose=0)
    assert np.isclose(m.layers[0].W[0, 0], 2.0, atol=0.05) and np.isclose(m.layers[0].b[0, 0], 1.0, atol=0.05)


def test_adam_solves_xor():
    from examples.xor import X, Y, accuracy
    from nn.activations import Sigmoid, Tanh
    from nn.layers import Dense
    from nn.losses import BinaryCrossEntropy
    from nn.model import Sequential

    for seed in range(5):
        m = Sequential([Dense(2, 4, seed=seed, initializer="xavier"), Tanh(),
                        Dense(4, 1, seed=seed + 100, initializer="xavier"), Sigmoid()])
        m.compile(loss=BinaryCrossEntropy(), optimizer=Adam(0.05))
        m.fit(X, Y, epochs=500, verbose=0)
        assert accuracy(m) == 1.0


@pytest.mark.parametrize("make", [lambda: RMSProp(0.1), lambda: Adam(0.1), lambda: Momentum(0.1)])
def test_changing_to_different_same_shape_parameter_is_an_error(make):
    opt = make()
    first = np.zeros(2)
    opt.step([(first, np.ones(2))])
    # Matching shapes are not enough: optimizer state belongs to the original arrays.
    with pytest.raises(ValueError, match="reset"):
        opt.step([(np.zeros(2), np.ones(2))])
