"""Day 9: activation / gradient statistics and the vanishing / exploding diagnosis."""

import numpy as np
import pytest

from nn.activations import ReLU, Sigmoid, Softmax, Tanh
from nn.diagnostics import LayerStats, collect_statistics, diagnose, format_statistics
from nn.layers import Dense
from nn.losses import MSE, CategoricalCrossEntropy
from nn.model import Sequential
from nn.optimizers import SGD
from nn.utils import one_hot


def data(n=256, d=64, k=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), one_hot(rng.integers(0, k, n), k)


def deep(act=ReLU, depth=8, width=64, bias=0.0, **init):
    layers, fan_in = [], 64
    for i in range(depth):
        d = Dense(fan_in, width, seed=i, **init)
        d.b[:] = bias
        layers += [d, act()]
        fan_in = width
    layers += [Dense(fan_in, 10, seed=99, **init), Softmax()]
    m = Sequential(layers)
    m.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(lr=0.1))
    return m


def stats_for(model, seed=0):
    x, y = data(seed=seed)
    return collect_statistics(model, x, y)


# ----------------------------------------------------------- Sequential hooks
def test_forward_record_collects_every_layer_output():
    m = deep(depth=2, initializer="he")
    rec = []
    out = m.forward(data()[0], record=rec)
    assert len(rec) == len(m.layers) and rec[-1] is out


def test_backward_record_collects_input_gradients_in_reverse_order():
    m = deep(depth=2, initializer="he")
    x, y = data()
    m.loss.forward(m.forward(x), y)
    rec = []
    m.backward_from_loss(record=rec)
    assert len(rec) == len(m.layers) - 1            # fused path skips the Softmax layer
    assert rec[-1].shape == x.shape                 # last recorded = gradient w.r.t. the data
    assert rec[0].shape == (len(x), 64)             # first recorded = gradient entering the last Dense layer's input


def test_recording_does_not_change_the_computation():
    x, y = data()
    a, b = deep(depth=2, initializer="he"), deep(depth=2, initializer="he")
    for m, rec in ((a, None), (b, [])):
        m.loss.forward(m.forward(x, record=rec), y)
        m.backward_from_loss(record=rec)
    assert all(np.array_equal(ga, gb) for (_, ga), (_, gb) in zip(a.params(), b.params()))


# ------------------------------------------------------------ collect/format
def test_one_stats_entry_per_layer_with_expected_fields():
    m = deep(depth=2, initializer="he")
    stats = stats_for(m)
    assert [s.index for s in stats] == list(range(len(m.layers)))
    dense = [s for s in stats if isinstance(m.layers[s.index], Dense)]
    assert all(s.weight_rms is not None and s.update_ratio is not None for s in dense)
    assert all(s.weight_rms is None for s in stats if s not in dense)


def test_statistics_match_manual_numpy_calculations():
    m = deep(depth=1, initializer="he")
    x, y = data()
    stats = collect_statistics(m, x, y)
    out0 = m.layers[0].forward(x)
    assert np.isclose(stats[0].out_std, out0.std()) and np.isclose(stats[0].out_mean, out0.mean())
    assert np.isclose(stats[0].weight_rms, np.sqrt(np.mean(m.layers[0].W ** 2)))
    relu_out = np.maximum(out0, 0)
    assert np.isclose(stats[1].frac_zero, np.mean(relu_out == 0))


def test_gradient_rms_of_last_layer_input_matches_fused_gradient():
    m = deep(depth=1, initializer="he")
    x, y = data()
    stats = collect_statistics(m, x, y)
    expected = np.sqrt(np.mean(((m.layers[-1]._out - y) / len(x)) ** 2))
    assert np.isclose(stats[-1].grad_in_rms, expected)


def test_collect_statistics_never_changes_parameters():
    m = deep(depth=2, initializer="he")
    before = [v.copy() for v, _ in m.params()]
    stats_for(m)
    assert all(np.array_equal(b, v) for b, (v, _) in zip(before, m.params()))


def test_works_for_non_fused_models_and_flat_targets():
    m = Sequential([Dense(3, 4, seed=0, initializer="he"), Tanh(), Dense(4, 1, seed=1)])
    m.compile(loss=MSE(), optimizer=SGD(lr=0.1))
    x = np.random.default_rng(0).normal(size=(20, 3))
    stats = collect_statistics(m, x, x[:, 0])
    assert len(stats) == 3 and all(np.isfinite(s.grad_in_rms) for s in stats)


def test_requires_a_compiled_model():
    with pytest.raises(RuntimeError):
        collect_statistics(Sequential([Dense(2, 2, seed=0)]), np.zeros((2, 2)), np.zeros((2, 2)))


def test_format_statistics_is_a_table_with_one_row_per_layer():
    m = deep(depth=2, initializer="he")
    text = format_statistics(stats_for(m))
    assert len(text.splitlines()) == 2 + len(m.layers) and "Dense(64 -> 64)" in text and "ReLU()" in text


def test_update_ratio_is_none_without_weights():
    s = LayerStats(0, "ReLU()", 0, 1, 1, 0.5, 0, 0, 1.0)
    assert s.update_ratio is None


# ---------------------------------------------------------------- diagnose
def test_healthy_networks_produce_no_warnings():
    assert diagnose(stats_for(deep(ReLU, initializer="he"))) == []
    assert diagnose(stats_for(deep(Tanh, initializer="xavier"))) == []


def test_tiny_init_is_diagnosed_as_vanishing():
    warnings = " | ".join(diagnose(stats_for(deep(ReLU, init_scale=0.01))))
    assert "vanishing signal" in warnings and "vanishing gradients" in warnings
    assert "negligible updates" in warnings


def test_large_init_is_diagnosed_as_exploding():
    warnings = " | ".join(diagnose(stats_for(deep(ReLU, init_scale=1.0))))
    assert "exploding signal" in warnings and "exploding gradients" in warnings
    assert "huge updates" in warnings


def test_deep_sigmoid_network_shows_vanishing_gradients_even_with_xavier():
    warnings = " | ".join(diagnose(stats_for(deep(Sigmoid, initializer="xavier"))))
    assert "vanishing gradients" in warnings


def test_saturated_tanh_is_reported():
    warnings = " | ".join(diagnose(stats_for(deep(Tanh, depth=3, init_scale=3.0))))
    assert "saturated units" in warnings


def test_dead_relu_units_are_reported():
    warnings = " | ".join(diagnose(stats_for(deep(ReLU, depth=3, bias=-50.0, initializer="he"))))
    assert "dead ReLU units" in warnings


def test_nan_inputs_are_reported():
    m = deep(depth=2, initializer="he")
    x, y = data()
    x[0, 0] = np.nan
    with np.errstate(invalid="ignore"):
        warnings = diagnose(collect_statistics(m, x, y))
    assert any("non-finite" in w for w in warnings)


def test_good_init_beats_bad_init_on_every_measure():
    he, tiny = stats_for(deep(ReLU, initializer="he")), stats_for(deep(ReLU, init_scale=0.01))
    dense = lambda st: [s for s in st if s.weight_rms is not None]
    assert dense(he)[-1].out_std > 100 * dense(tiny)[-1].out_std
    assert dense(he)[0].grad_in_rms > 1e6 * dense(tiny)[0].grad_in_rms
