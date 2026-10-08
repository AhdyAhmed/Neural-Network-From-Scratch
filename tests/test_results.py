"""Committed experiment results must stay consistent with the claims made in the notes / README."""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    path = ROOT / "results" / name
    if not path.exists():
        pytest.skip(f"{name} not generated; see the matching example script")
    return json.loads(path.read_text())


def test_day9_comparison_structure():
    r = load("day9_init_comparison.json")
    assert r["epochs"] == 10 and r["seeds"] == [0, 1, 2] and r["architecture"] == [784, 128, 64, 10]
    assert len(r["summary"]) == 7
    for s in r["summary"].values():
        assert len(s["val_acc_mean"]) == 10 and len(s["epochs_to_97pct"]) == 3


def test_day9_claims_hold_in_the_saved_numbers():
    s = load("day9_init_comparison.json")["summary"]
    tiny, day8 = s["N(0, 0.01²) + SGD"], s["N(0, 0.1²) + SGD (Day 8)"]
    he, he_mom = s["He + SGD"], s["He + Momentum"]
    # initialization: a much better first epoch than the library's old default
    assert he["val_acc_mean"][0] > tiny["val_acc_mean"][0] + 0.08
    # He + Momentum beats the Day 8 setup: higher final accuracy and 97% reached sooner
    assert he_mom["val_acc_mean"][-1] > day8["val_acc_mean"][-1] + 0.005
    assert max(he_mom["epochs_to_97pct"]) < min(day8["epochs_to_97pct"])
    # momentum is steadier than SGD at the same effective step (epoch 3)
    assert he_mom["val_acc_std"][2] < he["val_acc_std"][2] / 5
    assert he_mom["val_acc_mean"][2] > he["val_acc_mean"][2] + 0.02
    # the unreached-97% marker (None) never appears for the He rows
    assert None not in he_mom["epochs_to_97pct"] + he["epochs_to_97pct"]


def test_day10_comparison_structure():
    r = load("day10_optimizer_comparison.json")
    assert r["epochs"] == 10 and r["seeds"] == [0, 1, 2] and r["init"] == "he"
    assert list(r["summary"]) == ["SGD", "Momentum", "RMSProp", "Adam"]
    for s in r["summary"].values():
        assert len(s["val_acc_mean"]) == 10 and len(s["epochs_to_97pct"]) == 3


def test_day10_claims_hold_in_the_saved_numbers():
    s = load("day10_optimizer_comparison.json")["summary"]
    finals = {k: v["val_acc_mean"][-1] for k, v in s.items()}
    assert max(finals.values()) - min(finals.values()) < 0.005       # all four end within half a point
    assert all(f > 0.975 for f in finals.values())
    # Adam is the quickest and steadiest early on; plain SGD at its tuned lr wobbles
    assert s["Adam"]["val_acc_mean"][2] > 0.97 and s["SGD"]["val_acc_mean"][2] < 0.95
    assert s["SGD"]["val_acc_std"][2] > 10 * s["Adam"]["val_acc_std"][2]
    assert max(s["Adam"]["epochs_to_97pct"]) <= 4 and max(s["SGD"]["epochs_to_97pct"]) >= 5
    # per-step cost: the adaptive methods are slower than plain SGD
    assert s["Adam"]["seconds_per_epoch"] > s["SGD"]["seconds_per_epoch"]
