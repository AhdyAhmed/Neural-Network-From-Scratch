"""Day 7: CategoricalCrossEntropy and the fused softmax + CE gradient."""

import numpy as np
import pytest

from nn.activations import Softmax
from nn.gradcheck import check_loss
from nn.losses import CategoricalCrossEntropy
from nn.utils import one_hot


def probs(n, k, seed):
    z = np.random.default_rng(seed).normal(size=(n, k))
    return Softmax().forward(z)


def test_uniform_prediction_gives_ln_k():
    for k in (2, 3, 10):
        p = np.full((4, k), 1.0 / k)
        y = one_hot(np.arange(4) % k, k)
        assert np.isclose(CategoricalCrossEntropy().forward(p, y), np.log(k))


def test_perfect_prediction_gives_near_zero():
    y = one_hot([0, 2, 1], 3)
    assert CategoricalCrossEntropy().forward(y.copy(), y) < 1e-9


def test_known_value():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y = one_hot([0, 1], 3)
    expected = -(np.log(0.7) + np.log(0.8)) / 2
    assert np.isclose(CategoricalCrossEntropy().forward(p, y), expected)


def test_average_is_over_samples_not_elements():
    p = np.array([[0.5, 0.5]])
    y = one_hot([0], 2)
    one = CategoricalCrossEntropy().forward(p, y)
    doubled = CategoricalCrossEntropy().forward(np.vstack([p, p]), np.vstack([y, y]))
    assert np.isclose(one, doubled) and np.isclose(one, np.log(2))


def test_confident_wrong_is_much_worse_than_confident_right():
    y = one_hot([0], 3)
    ce = CategoricalCrossEntropy()
    assert ce.forward(np.array([[0.98, 0.01, 0.01]]), y) < 0.05
    assert ce.forward(np.array([[0.01, 0.98, 0.01]]), y) > 4.0


def test_loss_is_finite_when_probability_is_zero():
    p = np.array([[1.0, 0.0, 0.0]])
    y = one_hot([1], 3)
    ce = CategoricalCrossEntropy()
    with np.errstate(all="raise"):
        assert np.isfinite(ce.forward(p, y))
        assert np.all(np.isfinite(ce.backward()))


def test_shape_mismatch_message_points_to_one_hot():
    with pytest.raises(ValueError, match="one_hot"):
        CategoricalCrossEntropy().forward(np.full((4, 3), 1 / 3), np.array([0, 1, 2, 0]))


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        CategoricalCrossEntropy().forward(np.array([0.5, 0.5]), np.array([1.0, 0.0]))


def test_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        CategoricalCrossEntropy().backward()
    with pytest.raises(RuntimeError):
        CategoricalCrossEntropy().backward_logits()


# -------------------------------------------------------- gradients
@pytest.mark.parametrize("n, k", [(1, 2), (1, 5), (6, 3), (10, 10)])
def test_gradient_wrt_probabilities_matches_numerical(n, k):
    p = np.random.default_rng(n + k).uniform(0.05, 0.95, size=(n, k))
    y = one_hot(np.random.default_rng(k).integers(0, k, size=n), k)
    check_loss(CategoricalCrossEntropy(), p, y, tol=1e-6).assert_passed()


def test_gradient_with_soft_labels():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=(5, 4))
    y = Softmax().forward(rng.normal(size=(5, 4)))
    check_loss(CategoricalCrossEntropy(), p, y, tol=1e-6).assert_passed()


def test_fused_gradient_is_p_minus_y_over_n():
    p, y = probs(6, 4, 0), one_hot([0, 1, 2, 3, 0, 1], 4)
    ce = CategoricalCrossEntropy()
    ce.forward(p, y)
    assert np.allclose(ce.backward_logits(), (p - y) / 6)


def test_fused_gradient_equals_chain_through_softmax_for_normal_logits():
    z = np.random.default_rng(0).normal(size=(8, 5))
    y = one_hot(np.random.default_rng(1).integers(0, 5, size=8), 5)
    sm, ce = Softmax(), CategoricalCrossEntropy()
    ce.forward(sm.forward(z), y)
    assert np.allclose(sm.backward(ce.backward()), ce.backward_logits(), atol=1e-9)


def test_fused_gradient_rows_sum_to_zero():
    p, y = probs(5, 4, 3), one_hot([0, 1, 2, 3, 1], 4)
    ce = CategoricalCrossEntropy()
    ce.forward(p, y)
    assert np.allclose(ce.backward_logits().sum(axis=1), 0.0, atol=1e-12)


def test_fusion_matters_when_the_model_is_confidently_wrong():
    """Why fusing is not just a speed trick.

    Logits [100, 0, 0] with true class 1: p[1] ~ 4e-44 underflows below the
    clipping threshold, so the unfused chain (-y/p, then the softmax Jacobian)
    returns a gradient of ~0 and learning stalls. The fused gradient is exact.
    """
    z, y = np.array([[100.0, 0.0, 0.0]]), one_hot([1], 3)
    sm, ce = Softmax(), CategoricalCrossEntropy()
    ce.forward(sm.forward(z), y)

    unfused = sm.backward(ce.backward())
    fused = ce.backward_logits()

    assert np.allclose(fused, [[1.0, -1.0, 0.0]])
    assert np.abs(unfused).max() < 1e-6  # silently vanished
