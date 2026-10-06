"""Day 8: checks on the real MNIST data and the committed baseline results.

The data tests are skipped when MNIST is not cached in data/ (e.g. fresh clone
without network); run `python examples/mnist.py` once to download it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from nn.datasets import load_mnist

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mnist():
    try:
        return load_mnist(data_dir=ROOT / "data", download=False)
    except FileNotFoundError:
        pytest.skip("MNIST not cached; run `python examples/mnist.py` to download it")


def test_split_sizes_and_shapes(mnist):
    (xtr, ytr), (xva, yva), (xte, yte) = mnist
    assert (xtr.shape, xva.shape, xte.shape) == ((50000, 784), (10000, 784), (10000, 784))
    assert ytr.shape == (50000,) and yva.shape == (10000,) and yte.shape == (10000,)


def test_pixels_are_scaled_to_unit_range(mnist):
    x = mnist[0][0]
    assert x.min() == 0.0 and 0.99 <= x.max() <= 1.0


def test_all_ten_classes_present_and_roughly_balanced(mnist):
    for _, y in mnist:
        counts = np.bincount(y, minlength=10)
        assert counts.min() > 0.07 * len(y) and counts.max() < 0.13 * len(y)


def test_no_duplicate_images_across_train_and_test(mnist):
    train_hashes = {hash(row.tobytes()) for row in mnist[0][0][:5000]}
    assert not any(hash(row.tobytes()) in train_hashes for row in mnist[2][0][:1000])


def test_one_short_epoch_is_far_above_chance(mnist):
    from examples.mnist import CONFIG, train

    small = ((mnist[0][0][:10000], mnist[0][1][:10000]), (mnist[1][0][:2000], mnist[1][1][:2000]), mnist[2])
    _, history = train(small, {**CONFIG, "epochs": 2}, verbose=0)
    assert history["val_accuracy"][-1] > 0.7  # chance is 10%; two short epochs on 10k samples


# ---------------------------------------------------- committed baseline numbers
def test_committed_baseline_results_are_consistent():
    path = ROOT / "results" / "day8_mnist_baseline.json"
    if not path.exists():
        pytest.skip("run `python examples/mnist.py` to generate the baseline results")
    r = json.loads(path.read_text())
    assert len(r["history"]["val_accuracy"]) == r["config"]["epochs"] == 15
    assert r["final"]["val_accuracy"] == r["history"]["val_accuracy"][-1]
    assert r["final"]["test_accuracy"] > 0.97
    assert r["history"]["loss"][-1] < r["history"]["loss"][0] * 0.1
