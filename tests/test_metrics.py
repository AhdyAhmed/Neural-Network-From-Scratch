"""Day 8: accuracy metric."""

import numpy as np
import pytest

from nn.metrics import accuracy
from nn.utils import one_hot


def test_multiclass_with_integer_labels():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.5, 0.4, 0.1]])
    assert accuracy(p, np.array([0, 1, 2, 1])) == 0.75


def test_multiclass_with_one_hot_labels_matches_integer_labels():
    p = np.random.default_rng(0).random((50, 5))
    labels = np.random.default_rng(1).integers(0, 5, size=50)
    assert accuracy(p, one_hot(labels, 5)) == accuracy(p, labels)


def test_perfect_and_worst():
    labels = np.array([0, 1, 2, 1])
    assert accuracy(one_hot(labels, 3), labels) == 1.0
    assert accuracy(one_hot((labels + 1) % 3, 3), labels) == 0.0


def test_binary_column_vector_thresholds_at_half():
    p = np.array([[0.9], [0.4], [0.51], [0.1]])
    assert accuracy(p, np.array([[1], [0], [0], [0]])) == 0.75


def test_binary_flat_vector():
    assert accuracy(np.array([0.9, 0.2]), np.array([1, 0])) == 1.0


def test_threshold_is_strictly_greater_than_half():
    assert accuracy(np.array([[0.5]]), np.array([[0]])) == 1.0  # 0.5 counts as class 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(np.zeros((4, 3)), np.zeros(5))


def test_returns_python_float_between_0_and_1():
    value = accuracy(np.random.default_rng(0).random((20, 4)), np.random.default_rng(1).integers(0, 4, 20))
    assert isinstance(value, float) and 0.0 <= value <= 1.0
