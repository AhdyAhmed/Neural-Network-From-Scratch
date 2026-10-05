"""Day 7: one_hot and iterate_minibatches."""

import numpy as np
import pytest

from nn.utils import iterate_minibatches, one_hot


# ----------------------------------------------------------------- one_hot
def test_one_hot_values_and_shape():
    out = one_hot([0, 2, 1], 3)
    assert out.shape == (3, 3)
    assert np.array_equal(out, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])


def test_one_hot_infers_num_classes():
    assert one_hot([0, 3, 1]).shape == (3, 4)


def test_one_hot_rows_sum_to_one():
    out = one_hot(np.random.default_rng(0).integers(0, 10, size=50), 10)
    assert np.all(out.sum(axis=1) == 1.0)


def test_one_hot_explicit_classes_handles_missing_class():
    assert one_hot([0, 0, 1], 5).shape == (3, 5)


def test_one_hot_accepts_column_vector_and_integral_floats():
    assert np.array_equal(one_hot(np.array([[1], [0]]), 2), [[0, 1], [1, 0]])
    assert np.array_equal(one_hot(np.array([1.0, 0.0]), 2), [[0, 1], [1, 0]])


def test_one_hot_empty_input():
    assert one_hot(np.array([], dtype=int), 3).shape == (0, 3)


@pytest.mark.parametrize("bad, k", [([0, 3], 3), ([-1, 0], 3), ([0.5, 1.0], 3)])
def test_one_hot_rejects_invalid_labels(bad, k):
    with pytest.raises(ValueError):
        one_hot(bad, k)


def test_one_hot_rejects_2d_input():
    with pytest.raises(ValueError):
        one_hot(np.zeros((3, 2), dtype=int), 3)


# ------------------------------------------------------ iterate_minibatches
def data(n=10):
    x = np.arange(n * 2, dtype=float).reshape(n, 2)
    y = np.arange(n, dtype=float).reshape(n, 1)
    return x, y


def test_batches_cover_every_sample_exactly_once():
    x, y = data(10)
    seen = np.concatenate([yb.ravel() for _, yb in iterate_minibatches(x, y, 3, rng=np.random.default_rng(0))])
    assert sorted(seen) == list(range(10))


def test_batch_sizes_with_remainder():
    x, y = data(10)
    sizes = [len(xb) for xb, _ in iterate_minibatches(x, y, 4, shuffle=False)]
    assert sizes == [4, 4, 2]


def test_batch_larger_than_dataset_gives_one_batch():
    x, y = data(5)
    assert [len(xb) for xb, _ in iterate_minibatches(x, y, 100)] == [5]


def test_no_shuffle_preserves_order():
    x, y = data(6)
    out = np.concatenate([yb.ravel() for _, yb in iterate_minibatches(x, y, 2, shuffle=False)])
    assert np.array_equal(out, np.arange(6))


def test_x_and_y_stay_aligned_after_shuffling():
    x, y = data(20)
    for xb, yb in iterate_minibatches(x, y, 6, rng=np.random.default_rng(1)):
        assert np.array_equal(xb[:, 0] / 2, yb.ravel())  # row i of x is [2i, 2i+1]


def test_shuffle_is_reproducible_with_a_seed_and_actually_shuffles():
    x, y = data(30)
    run = lambda s: np.concatenate([yb.ravel() for _, yb in iterate_minibatches(x, y, 7, rng=np.random.default_rng(s))])
    assert np.array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))
    assert not np.array_equal(run(0), np.arange(30))


def test_minibatch_validation():
    x, y = data(4)
    with pytest.raises(ValueError):
        list(iterate_minibatches(x, y, 0))
    with pytest.raises(ValueError):
        list(iterate_minibatches(x, y[:3], 2))


def test_input_arrays_are_not_modified():
    x, y = data(8)
    xc, yc = x.copy(), y.copy()
    list(iterate_minibatches(x, y, 3, rng=np.random.default_rng(0)))
    assert np.array_equal(x, xc) and np.array_equal(y, yc)
