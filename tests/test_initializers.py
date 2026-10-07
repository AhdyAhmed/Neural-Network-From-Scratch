"""Day 9: He / Xavier initializers and their integration with Dense."""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Softmax, Tanh
from nn.initializers import (
    get_initializer,
    he_normal,
    he_uniform,
    initializer_for,
    normal,
    xavier_normal,
    xavier_uniform,
)
from nn.layers import Dense

FAN_IN, FAN_OUT = 400, 300  # 120k weights: enough for tight statistical checks


def rng(seed=0):
    return np.random.default_rng(seed)


# ------------------------------------------------------------- statistics
def test_he_normal_variance():
    w = he_normal(FAN_IN, FAN_OUT, rng())
    assert w.shape == (FAN_IN, FAN_OUT)
    assert np.isclose(w.std(), np.sqrt(2 / FAN_IN), rtol=0.02) and abs(w.mean()) < 5e-4


def test_xavier_normal_variance():
    w = xavier_normal(FAN_IN, FAN_OUT, rng())
    assert np.isclose(w.std(), np.sqrt(2 / (FAN_IN + FAN_OUT)), rtol=0.02) and abs(w.mean()) < 5e-4


def test_he_uniform_bounds_and_variance():
    w = he_uniform(FAN_IN, FAN_OUT, rng())
    limit = np.sqrt(6 / FAN_IN)
    assert w.min() >= -limit and w.max() <= limit
    assert np.isclose(w.std(), np.sqrt(2 / FAN_IN), rtol=0.02)  # same variance as the normal version


def test_xavier_uniform_bounds_and_variance():
    w = xavier_uniform(FAN_IN, FAN_OUT, rng())
    limit = np.sqrt(6 / (FAN_IN + FAN_OUT))
    assert w.min() >= -limit and w.max() <= limit
    assert np.isclose(w.std(), np.sqrt(2 / (FAN_IN + FAN_OUT)), rtol=0.02)


def test_he_depends_only_on_fan_in_while_xavier_uses_both():
    assert np.isclose(he_normal(100, 10, rng()).std(), he_normal(100, 1000, rng()).std(), rtol=0.1)
    assert xavier_normal(100, 10, rng()).std() > xavier_normal(100, 1000, rng()).std() * 2


def test_normal_scale():
    assert np.isclose(normal(0.3)(FAN_IN, FAN_OUT, rng()).std(), 0.3, rtol=0.02)


@pytest.mark.parametrize("fn", [he_normal, he_uniform, xavier_normal, xavier_uniform])
def test_reproducible_and_seed_dependent(fn):
    assert np.array_equal(fn(5, 4, rng(1)), fn(5, 4, rng(1)))
    assert not np.array_equal(fn(5, 4, rng(1)), fn(5, 4, rng(2)))


@pytest.mark.parametrize("fn", [he_normal, he_uniform, xavier_normal, xavier_uniform])
@pytest.mark.parametrize("bad", [(0, 3), (3, 0), (-1, 3)])
def test_invalid_fans_rejected(fn, bad):
    with pytest.raises(ValueError):
        fn(*bad, rng())


@pytest.mark.parametrize("scale", [0, -0.1])
def test_normal_rejects_non_positive_scale(scale):
    with pytest.raises(ValueError):
        normal(scale)


# --------------------------------------------------------------- registry
@pytest.mark.parametrize(
    "name, fn",
    [("he", he_normal), ("He_Normal", he_normal), ("he_uniform", he_uniform),
     ("xavier", xavier_normal), ("glorot", xavier_normal), ("glorot_uniform", xavier_uniform),
     ("xavier_uniform", xavier_uniform)],
)
def test_get_initializer_by_name(name, fn):
    assert get_initializer(name) is fn


def test_get_initializer_passes_callables_through():
    assert get_initializer(he_uniform) is he_uniform


def test_unknown_initializer_lists_the_options():
    with pytest.raises(ValueError, match="xavier_normal"):
        get_initializer("kaiming")
    with pytest.raises(ValueError):
        get_initializer(3.5)


@pytest.mark.parametrize("act", [ReLU(), ReLU, "relu", "ReLU"])
def test_recommended_initializer_for_relu_is_he(act):
    assert initializer_for(act) is he_normal


@pytest.mark.parametrize("act", [Tanh(), Sigmoid, Softmax(), "tanh", None])
def test_recommended_initializer_otherwise_is_xavier(act):
    assert initializer_for(act) is xavier_normal


# ------------------------------------------------------------------- Dense
def test_dense_with_named_initializer_has_he_variance():
    layer = Dense(FAN_IN, FAN_OUT, seed=0, initializer="he")
    assert np.isclose(layer.W.std(), np.sqrt(2 / FAN_IN), rtol=0.02)
    assert np.all(layer.b == 0)


def test_dense_with_callable_initializer():
    layer = Dense(3, 2, initializer=lambda fi, fo, r: np.full((fi, fo), 7.0))
    assert np.all(layer.W == 7.0)


def test_dense_rejects_wrong_shaped_initializer_output():
    with pytest.raises(ValueError, match="shape"):
        Dense(3, 2, initializer=lambda fi, fo, r: np.zeros((fo, fi)))


def test_dense_rejects_both_initializer_and_init_scale():
    with pytest.raises(ValueError, match="either"):
        Dense(3, 2, initializer="he", init_scale=0.1)


def test_dense_default_and_init_scale_behaviour_unchanged():
    assert np.isclose(Dense(FAN_IN, FAN_OUT, seed=0).W.std(), 0.01, rtol=0.02)
    assert np.isclose(Dense(FAN_IN, FAN_OUT, seed=0, init_scale=0.5).W.std(), 0.5, rtol=0.02)
    # Same seed, same weights as the pre-Day-9 code path (N(0, scale) drawn from default_rng(seed)):
    expected = np.random.default_rng(3).normal(0.0, 0.5, size=(4, 5))
    assert np.array_equal(Dense(4, 5, seed=3, init_scale=0.5).W, expected)


def test_dense_initializer_is_reproducible_with_seed():
    a, b = Dense(6, 6, seed=9, initializer="xavier"), Dense(6, 6, seed=9, initializer="xavier")
    assert np.array_equal(a.W, b.W)


# ---------------------- the point of it all: signal size through a deep net
def second_moment_ratio(act_cls, depth=10, width=200, **init):
    x = np.random.default_rng(0).normal(size=(500, width))
    base, a = np.mean(x**2), x
    for i in range(depth):
        a = act_cls().forward(Dense(width, width, seed=i, **init).forward(a))
    return np.mean(a**2) / base


def test_he_keeps_relu_activations_at_the_same_scale_through_ten_layers():
    assert 0.5 < second_moment_ratio(ReLU, initializer="he") < 2.0


def test_xavier_shrinks_relu_activations_because_relu_halves_the_variance():
    assert second_moment_ratio(ReLU, initializer="xavier") < 0.01  # about 2^-10


def test_tiny_init_vanishes_and_large_init_explodes():
    assert second_moment_ratio(ReLU, init_scale=0.01) < 1e-10
    assert second_moment_ratio(ReLU, init_scale=1.0) > 1e6


def test_xavier_keeps_tanh_activations_in_a_healthy_range():
    assert 0.01 < second_moment_ratio(Tanh, initializer="xavier") < 0.5
