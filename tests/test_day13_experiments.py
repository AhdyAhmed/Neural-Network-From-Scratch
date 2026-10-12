"""Tests for the Day 13 experiment configuration and reproducibility."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "day13_experiments", ROOT / "examples" / "day13_experiments.py"
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_final_config_records_seeds_and_training_parameters():
    config = module.FINAL_CONFIG
    assert config["seed"] == 2026
    assert config["shuffle_seed"] == 2027
    assert config["architecture"] == [784, 128, 64, 10]
    assert config["learning_rate"] > 0
    assert config["batch_size"] == 64
    assert config["early_stopping"] is True


def test_build_model_initialization_is_reproducible():
    first = module.build_model(dict(module.FINAL_CONFIG))
    second = module.build_model(dict(module.FINAL_CONFIG))
    first_weights = [layer.W for layer in first.layers if hasattr(layer, "W")]
    second_weights = [layer.W for layer in second.layers if hasattr(layer, "W")]
    assert len(first_weights) == len(second_weights) == 3
    for a, b in zip(first_weights, second_weights):
        np.testing.assert_array_equal(a, b)


def test_ablation_configs_cover_requested_dimensions():
    configs = module.make_ablation_configs(epochs=3)
    by_name = {config["name"]: config for config in configs}
    assert set(by_name) == {
        "baseline_no_dropout",
        "dropout_0.10",
        "xavier_no_dropout",
        "learning_rate_0.0003",
    }
    assert by_name["baseline_no_dropout"]["initializer"] == "he"
    assert by_name["dropout_0.10"]["dropout_rate"] == 0.10
    assert by_name["xavier_no_dropout"]["initializer"] == "xavier"
    assert by_name["learning_rate_0.0003"]["learning_rate"] == 0.0003
    assert all(c["epochs"] == 3 for c in configs)


def test_run_one_returns_measured_history_on_small_synthetic_data():
    rng = np.random.default_rng(7)
    x_train = rng.random((24, 784))
    y_train = rng.integers(0, 10, size=24)
    x_val = rng.random((8, 784))
    y_val = rng.integers(0, 10, size=8)
    x_test = rng.random((4, 784))
    y_test = rng.integers(0, 10, size=4)
    config = dict(module.FINAL_CONFIG)
    config.update({"epochs": 1, "early_stopping": False, "dropout_rate": 0.0})
    result, history, model = module.run_one(
        ((x_train, y_train), (x_val, y_val), (x_test, y_test)), config, verbose=0
    )
    assert result["epochs_completed"] == 1
    assert len(history["loss"]) == len(history["val_loss"]) == 1
    assert 0.0 <= result["final_recorded_validation_accuracy"] <= 1.0
    assert np.isfinite(model.predict(x_test)).all()
