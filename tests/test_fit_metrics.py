"""Day 8: fit(metrics=...) records per-epoch metrics for train and validation data."""

import numpy as np
import pytest

from nn.activations import Softmax
from nn.layers import Dense
from nn.losses import CategoricalCrossEntropy
from nn.metrics import accuracy
from nn.model import Sequential
from nn.optimizers import SGD
from nn.utils import one_hot


def data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    labels = (x @ rng.normal(size=(3, 3))).argmax(axis=1)
    return x, one_hot(labels, 3), labels


def make(lr=0.5):
    m = Sequential([Dense(3, 3, seed=0, init_scale=0.5), Softmax()])
    m.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=lr))
    return m


def test_no_metrics_keeps_history_unchanged():
    x, y, _ = data()
    assert list(make().fit(x, y, epochs=2, verbose=0)) == ["loss"]


def test_metric_keys_and_lengths():
    x, y, _ = data()
    h = make().fit(x, y, epochs=4, batch_size=16, validation_data=data(seed=1)[:2],
                   metrics={"accuracy": accuracy}, verbose=0)
    assert set(h) == {"loss", "accuracy", "val_loss", "val_accuracy"}
    assert all(len(v) == 4 for v in h.values())


def test_metric_without_validation_data():
    x, y, _ = data()
    h = make().fit(x, y, epochs=3, metrics={"accuracy": accuracy}, verbose=0)
    assert set(h) == {"loss", "accuracy"}


def test_accuracy_improves_with_training():
    x, y, _ = data(200)
    h = make().fit(x, y, epochs=60, batch_size=32, seed=0, metrics={"accuracy": accuracy}, verbose=0)
    assert h["accuracy"][-1] > h["accuracy"][0] and h["accuracy"][-1] > 0.9


@pytest.mark.parametrize("batch", [None, 7, 16, 50])
def test_running_metric_is_batch_size_weighted(batch):
    """With ~zero learning rate the weights never change, so the epoch's running accuracy
    must equal the full-data accuracy however the (uneven) batches are cut."""
    x, y, labels = data(50)
    expected = accuracy(make(lr=1e-15).predict(x), labels)
    h = make(lr=1e-15).fit(x, y, epochs=1, batch_size=batch, seed=0, metrics={"accuracy": accuracy}, verbose=0)
    assert np.isclose(h["accuracy"][0], expected)


def test_validation_metric_matches_a_manual_computation():
    x, y, _ = data()
    x_val, y_val, labels_val = data(seed=1)
    m = make(lr=1e-15)
    h = m.fit(x, y, epochs=1, validation_data=(x_val, y_val), metrics={"accuracy": accuracy}, verbose=0)
    assert np.isclose(h["val_accuracy"][0], accuracy(m.predict(x_val), labels_val))
    assert np.isclose(h["val_loss"][0], m.evaluate(x_val, y_val))


def test_multiple_custom_metrics():
    x, y, _ = data()
    h = make().fit(x, y, epochs=2, metrics={"acc": accuracy, "const": lambda p, t: 0.5}, verbose=0)
    assert h["const"] == [0.5, 0.5] and "acc" in h


def test_verbose_output_shows_metrics(capsys):
    x, y, _ = data()
    make().fit(x, y, epochs=2, validation_data=data(seed=1)[:2], metrics={"accuracy": accuracy}, verbose=1)
    out = capsys.readouterr().out
    assert "accuracy:" in out and "val_accuracy:" in out
