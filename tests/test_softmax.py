"""Day 7: Softmax layer."""

import numpy as np
import pytest

from nn.activations import Softmax
from nn.gradcheck import check_layer


def test_rows_sum_to_one_and_are_positive():
    x = np.random.default_rng(0).normal(scale=3, size=(8, 5))
    out = Softmax().forward(x)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.all(out > 0)


def test_known_values():
    out = Softmax().forward(np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]]))
    assert np.allclose(out[0], 0.25)
    assert np.allclose(out[1], [0.0320586, 0.0871443, 0.2368828, 0.6439143], atol=1e-6)


def test_stable_for_huge_inputs():
    x = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = Softmax().forward(x)
    assert np.all(np.isfinite(out)) and np.allclose(out.sum(axis=1), 1.0)
    assert out[0].argmax() == 1 and out[1].argmax() == 2


def test_shift_invariance():
    x = np.random.default_rng(1).normal(size=(4, 6))
    s = Softmax()
    assert np.allclose(s.forward(x), s.forward(x + 123.456))


def test_preserves_argmax_and_shape():
    x = np.random.default_rng(2).normal(size=(7, 4))
    out = Softmax().forward(x)
    assert out.shape == x.shape and np.array_equal(out.argmax(1), x.argmax(1))


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        Softmax().forward(np.zeros(5))


def test_does_not_modify_input():
    x = np.random.default_rng(0).normal(size=(3, 4))
    before = x.copy()
    Softmax().forward(x)
    assert np.array_equal(x, before)


def test_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        Softmax().backward(np.ones((1, 3)))


def test_backward_wrong_shape_raises():
    s = Softmax()
    s.forward(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        s.backward(np.ones((2, 2)))


def test_backward_rows_sum_to_zero():
    """Softmax outputs always sum to 1, so no change in logits can change that sum."""
    s = Softmax()
    s.forward(np.random.default_rng(0).normal(size=(5, 4)))
    dz = s.backward(np.random.default_rng(1).normal(size=(5, 4)))
    assert np.allclose(dz.sum(axis=1), 0.0, atol=1e-12)


def test_backward_matches_explicit_jacobian():
    x = np.random.default_rng(3).normal(size=(1, 4))
    g = np.random.default_rng(4).normal(size=(1, 4))
    s = Softmax()
    p = s.forward(x)[0]
    jac = np.diag(p) - np.outer(p, p)  # ds_k/dz_j = s_k (delta_kj - s_j)
    assert np.allclose(s.backward(g)[0], jac @ g[0])


@pytest.mark.parametrize("shape", [(1, 2), (1, 5), (6, 3), (10, 10)])
def test_gradient_check(shape):
    x = np.random.default_rng(sum(shape)).normal(scale=1.5, size=shape)
    check_layer(Softmax(), x, tol=1e-6).assert_passed()


def test_output_is_independent_of_cache():
    """Same aliasing protection as Sigmoid/Tanh (Day 5 fix)."""
    s = Softmax()
    x = np.random.default_rng(0).normal(size=(2, 3))
    g = np.random.default_rng(1).normal(size=(2, 3))
    s.forward(x)
    expected = s.backward(g).copy()
    out = s.forward(x)
    out += 10.0
    assert np.allclose(s.backward(g), expected)
