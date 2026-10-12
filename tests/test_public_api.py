"""Release-level checks for the documented public package surface."""

from __future__ import annotations

import nn


def test_release_version() -> None:
    assert nn.__version__ == "0.1.0"


def test_every_public_export_is_available() -> None:
    assert len(nn.__all__) == len(set(nn.__all__))
    for name in nn.__all__:
        assert hasattr(nn, name), name


def test_quick_start_api_is_exported() -> None:
    model = nn.Sequential([nn.Dense(1, 2, seed=0), nn.ReLU(), nn.Dense(2, 1, seed=1)])
    model.compile(loss=nn.MSE(), optimizer=nn.SGD(lr=0.01))
    assert model.summary().endswith("     7")
