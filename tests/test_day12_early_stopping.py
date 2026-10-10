import numpy as np
import pytest

from nn.activations import Softmax
from nn.layers import Dense
from nn.losses import CategoricalCrossEntropy
from nn.model import Sequential
from nn.optimizers import SGD
from nn.utils import one_hot


def build():
    model = Sequential([Dense(2, 3, seed=0), Softmax()])
    model.compile(CategoricalCrossEntropy(), SGD(lr=1e-30))
    return model


def dataset():
    x = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
    y = one_hot(np.array([0, 1, 0, 1]), 3)
    return x, y


def test_early_stopping_stops_after_patience_and_restores_best_weights():
    x, y = dataset()
    model = build()
    reference = build()
    reference.fit(x, y, epochs=1, validation_data=(x, y), verbose=0)
    best_params = [value.copy() for value, _ in reference.params()]
    history = model.fit(x, y, epochs=20, validation_data=(x, y), early_stopping=True,
                        patience=2, verbose=0)
    assert len(history["loss"]) == 3
    for (value, _), expected in zip(model.params(), best_params):
        np.testing.assert_array_equal(value, expected)


def test_early_stopping_requires_validation_data():
    x, y = dataset()
    with pytest.raises(ValueError, match="requires validation_data"):
        build().fit(x, y, epochs=2, early_stopping=True, verbose=0)


@pytest.mark.parametrize("kwargs", [{"patience": 0}, {"min_delta": -0.1}, {"min_delta": float("nan")}])
def test_early_stopping_arguments_are_validated(kwargs):
    x, y = dataset()
    with pytest.raises(ValueError):
        build().fit(x, y, epochs=2, validation_data=(x, y), early_stopping=True, verbose=0, **kwargs)


def test_default_training_still_runs_requested_epochs():
    x, y = dataset()
    history = build().fit(x, y, epochs=4, validation_data=(x, y), verbose=0)
    assert len(history["loss"]) == 4
