"""Day 7: shuffling and mini-batch training in Sequential.fit()."""

import numpy as np
import pytest

from nn.layers import Dense
from nn.losses import MSE
from nn.model import Sequential
from nn.optimizers import SGD


class CountingSGD(SGD):
    """SGD that counts how many parameter updates happen."""

    def __init__(self, lr=0.1):
        super().__init__(lr)
        self.steps = 0

    def step(self, params):
        self.steps += 1
        super().step(params)


def line_data(n=64, seed=0):
    x = np.random.default_rng(seed).uniform(-1, 1, size=(n, 1))
    return x, 2 * x + 1


def make(lr=0.1, optimizer=None):
    m = Sequential([Dense(1, 1, seed=0)])
    m.compile(loss=MSE(), optimizer=optimizer or SGD(lr=lr))
    return m


@pytest.mark.parametrize("n, batch, expected", [(64, 16, 4), (10, 3, 4), (10, 10, 1), (10, 100, 1), (7, 1, 7)])
def test_updates_per_epoch(n, batch, expected):
    opt = CountingSGD()
    make(optimizer=opt).fit(*line_data(n), epochs=1, batch_size=batch, verbose=0)
    assert opt.steps == expected


def test_default_is_full_batch_with_one_update_per_epoch():
    opt = CountingSGD()
    make(optimizer=opt).fit(*line_data(), epochs=5, verbose=0)
    assert opt.steps == 5


def test_minibatch_training_learns_the_line():
    m = make(lr=0.1)
    m.fit(*line_data(), epochs=100, batch_size=8, seed=0, verbose=0)
    w, b = m.layers[0].W[0, 0], m.layers[0].b[0, 0]
    assert np.isclose(w, 2.0, atol=0.02) and np.isclose(b, 1.0, atol=0.02)


def test_minibatches_make_more_progress_per_epoch_than_full_batch():
    x, y = line_data()
    full = make(lr=0.05).fit(x, y, epochs=5, verbose=0)
    mini = make(lr=0.05).fit(x, y, epochs=5, batch_size=8, seed=0, verbose=0)
    assert mini["loss"][-1] < full["loss"][-1]


def test_same_seed_is_reproducible_and_different_seeds_differ():
    x, y = line_data()
    run = lambda s: make().fit(x, y, epochs=5, batch_size=8, seed=s, verbose=0)["loss"]
    assert run(0) == run(0)
    assert run(0) != run(1)


def test_shuffle_false_ignores_the_seed():
    x, y = line_data()
    run = lambda s: make().fit(x, y, epochs=5, batch_size=8, shuffle=False, seed=s, verbose=0)["loss"]
    assert run(0) == run(99)


def test_full_batch_result_does_not_depend_on_shuffle_or_seed():
    x, y = line_data()
    a = make().fit(x, y, epochs=10, verbose=0)["loss"]
    b = make().fit(x, y, epochs=10, shuffle=True, seed=5, verbose=0)["loss"]
    assert a == b


def test_epoch_loss_is_weighted_by_batch_size():
    """With a ~zero learning rate the weights never change, so the epoch loss must equal
    the full-data loss no matter how the (uneven) batches are cut."""
    x, y = line_data(10)
    m = make(lr=1e-15)
    expected = m.evaluate(x, y)
    for batch in (1, 3, 4, 7):
        got = make(lr=1e-15).fit(x, y, epochs=1, batch_size=batch, seed=0, verbose=0)["loss"][0]
        assert np.isclose(got, expected, rtol=1e-9)


def test_validation_loss_recorded_with_minibatches():
    x, y = line_data()
    h = make().fit(x, y, epochs=4, batch_size=16, validation_data=line_data(seed=1), verbose=0)
    assert len(h["loss"]) == len(h["val_loss"]) == 4


@pytest.mark.parametrize("bad", [0, -4, 2.5, "8"])
def test_invalid_batch_size_rejected(bad):
    with pytest.raises(ValueError):
        make().fit(*line_data(), epochs=1, batch_size=bad, verbose=0)


def test_does_not_modify_the_training_arrays():
    x, y = line_data()
    xc, yc = x.copy(), y.copy()
    make().fit(x, y, epochs=3, batch_size=8, seed=0, verbose=0)
    assert np.array_equal(x, xc) and np.array_equal(y, yc)
