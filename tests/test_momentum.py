"""Day 9: Momentum optimizer."""

import numpy as np
import pytest

from nn.optimizers import SGD, Momentum


def test_first_two_steps_match_hand_calculation():
    w, g = np.array([1.0]), np.array([0.5])
    opt = Momentum(lr=0.1, beta=0.9)
    opt.step([(w, g)])               # v = 0.5          w = 1 - 0.05  = 0.95
    assert np.isclose(w[0], 0.95)
    opt.step([(w, g)])               # v = 0.9*0.5+0.5 = 0.95   w = 0.95 - 0.095 = 0.855
    assert np.isclose(w[0], 0.855)


def test_beta_zero_is_exactly_sgd():
    rng = np.random.default_rng(0)
    w1, w2 = rng.normal(size=(3, 2)), None
    w2 = w1.copy()
    a, b = Momentum(lr=0.05, beta=0.0), SGD(lr=0.05)
    for _ in range(5):
        g = rng.normal(size=(3, 2))
        a.step([(w1, g)])
        b.step([(w2, g)])
    assert np.allclose(w1, w2)


def test_updates_in_place_and_keeps_one_velocity_per_parameter():
    w1, w2 = np.ones(2), np.ones((2, 2))
    ref = w1
    opt = Momentum(lr=0.1)
    opt.step([(w1, np.ones(2)), (w2, np.ones((2, 2)))])
    assert ref is w1 and np.allclose(w1, 0.9) and np.allclose(w2, 0.9)


def test_velocity_builds_up_toward_1_over_1_minus_beta_times_the_gradient():
    w, g = np.zeros(1), np.ones(1)
    opt = Momentum(lr=1.0, beta=0.9)
    prev = 0.0
    for _ in range(200):
        opt.step([(w, g)])
        step, prev = prev - w[0], w[0]
    assert np.isclose(step, 10.0, rtol=1e-3)  # steady-state step is lr * g / (1 - beta)


def test_converges_on_a_quadratic():
    w = np.array([5.0])
    opt = Momentum(lr=0.05, beta=0.9)
    for _ in range(400):
        opt.step([(w, 2 * (w - 3.0))])
    assert np.isclose(w[0], 3.0, atol=1e-6)


def test_faster_than_sgd_in_a_narrow_valley():
    """f = 0.5 (w0^2 + 100 w1^2): a long shallow direction and a steep one. SGD's stable step size
    is capped by the steep direction, which leaves it crawling along the shallow one."""
    def run(opt, steps=150):
        w = np.array([10.0, 1.0])
        for _ in range(steps):
            opt.step([(w, np.array([w[0], 100.0 * w[1]]))])
        return abs(w[0]) + abs(w[1])

    assert run(Momentum(lr=0.015, beta=0.9)) < 0.05 * run(SGD(lr=0.015))


def test_reset_clears_velocity():
    w, g = np.zeros(1), np.ones(1)
    opt = Momentum(lr=0.1, beta=0.9)
    opt.step([(w, g)])
    opt.reset()
    w2 = np.zeros(1)
    opt.step([(w2, g)])
    assert np.isclose(w2[0], -0.1)  # behaves like a first step again


def test_changing_parameter_set_without_reset_is_an_error():
    opt = Momentum(lr=0.1)
    opt.step([(np.zeros(2), np.ones(2))])
    with pytest.raises(ValueError, match="reset"):
        opt.step([(np.zeros(3), np.ones(3))])
    with pytest.raises(ValueError, match="reset"):
        opt.step([(np.zeros(2), np.ones(2)), (np.zeros(2), np.ones(2))])


@pytest.mark.parametrize("lr", [0, -0.1])
def test_rejects_non_positive_lr(lr):
    with pytest.raises(ValueError):
        Momentum(lr=lr)


@pytest.mark.parametrize("beta", [-0.1, 1.0, 1.5])
def test_rejects_beta_outside_zero_one(beta):
    with pytest.raises(ValueError):
        Momentum(beta=beta)


def test_repr():
    assert repr(Momentum(lr=0.01, beta=0.9)) == "Momentum(lr=0.01, beta=0.9)"


def test_trains_a_model_end_to_end():
    from nn.layers import Dense
    from nn.losses import MSE
    from nn.model import Sequential

    x = np.random.default_rng(0).uniform(-1, 1, size=(64, 1))
    m = Sequential([Dense(1, 1, seed=0)])
    m.compile(loss=MSE(), optimizer=Momentum(lr=0.05, beta=0.9))
    m.fit(x, 2 * x + 1, epochs=200, verbose=0)
    assert np.isclose(m.layers[0].W[0, 0], 2.0, atol=0.01) and np.isclose(m.layers[0].b[0, 0], 1.0, atol=0.01)


def test_changing_to_different_same_shape_parameter_is_an_error():
    opt = Momentum(lr=0.1)
    first = np.zeros(2)
    opt.step([(first, np.ones(2))])
    with pytest.raises(ValueError, match="reset"):
        opt.step([(np.zeros(2), np.ones(2))])
