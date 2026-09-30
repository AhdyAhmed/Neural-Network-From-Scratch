"""Day 1 smoke tests: the package imports and the environment works."""

import numpy as np

import nn


def test_package_imports():
    assert nn.__version__ == "0.0.1"


def test_numpy_matmul_shapes():
    x = np.random.default_rng(0).normal(size=(4, 3))
    w = np.random.default_rng(1).normal(size=(3, 2))
    assert (x @ w).shape == (4, 2)
