"""Day 8: MNIST loader (tested on small synthetic files; no network needed)."""

import gzip
import io
import os
import pickle
import urllib.request

import numpy as np
import pytest

from nn import datasets
from nn.datasets import MNIST_FILENAME, load_mnist, parse_mnist_pickle, sha256_of


def fake_mnist(n=(20, 6, 6), seed=0):
    rng = np.random.default_rng(seed)
    return tuple(
        (rng.random((k, 784)).astype(np.float32), rng.integers(0, 10, size=k).astype(np.int64)) for k in n
    )


def write(path, obj, protocol=4):  # protocol 4 stores arrays without _codecs.encode (not allowed by the loader)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)


@pytest.fixture
def fake_file(tmp_path):
    path = tmp_path / MNIST_FILENAME
    write(path, fake_mnist())
    return path


# -------------------------------------------------------------- parsing
def test_parse_returns_three_splits_with_expected_dtypes(fake_file):
    train, val, test = parse_mnist_pickle(fake_file)
    assert train[0].shape == (20, 784) and val[0].shape == (6, 784) and test[0].shape == (6, 784)
    assert train[1].dtype == np.int64 and train[0].dtype == np.float32


def test_parse_round_trips_values(fake_file):
    original = fake_mnist()
    parsed = parse_mnist_pickle(fake_file)
    for (x, y), (xo, yo) in zip(parsed, original):
        assert np.array_equal(x, xo) and np.array_equal(y, yo)


def test_parse_rejects_wrong_layout(tmp_path):
    path = tmp_path / "bad.pkl.gz"
    write(path, fake_mnist()[:2])
    with pytest.raises(ValueError, match="layout"):
        parse_mnist_pickle(path)


def test_parse_rejects_mismatched_shapes(tmp_path):
    bad = list(fake_mnist())
    bad[0] = (bad[0][0], bad[0][1][:-1])
    path = tmp_path / "bad.pkl.gz"
    write(path, tuple(bad))
    with pytest.raises(ValueError, match="shapes"):
        parse_mnist_pickle(path)


# ------------------------------------------------------------- security
class Evil:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def test_malicious_pickle_is_refused_and_never_executed(tmp_path):
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.pkl.gz"
    write(path, Evil(marker))
    with pytest.raises(pickle.UnpicklingError, match="Refusing"):
        parse_mnist_pickle(path)
    assert not marker.exists()


def test_arbitrary_python_objects_are_refused(tmp_path):
    path = tmp_path / "obj.pkl.gz"
    write(path, (io.StringIO("x"),))  # any non-NumPy class
    with pytest.raises(Exception):
        parse_mnist_pickle(path)


# ------------------------------------------------------------- load_mnist
def test_load_uses_cached_file_without_downloading(tmp_path, fake_file):
    train, _, _ = load_mnist(data_dir=tmp_path, download=False, verify=False)
    assert train[0].shape[1] == 784


def test_missing_file_with_downloads_disabled_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mnist(data_dir=tmp_path, download=False)


def test_checksum_mismatch_is_rejected(tmp_path, fake_file):
    with pytest.raises(ValueError, match="checksum"):
        load_mnist(data_dir=tmp_path, download=False, verify=True)


def test_sha256_matches_hashlib(fake_file):
    import hashlib

    assert sha256_of(fake_file) == hashlib.sha256(fake_file.read_bytes()).hexdigest()


def test_download_success_path(tmp_path, monkeypatch):
    source = tmp_path / "source.pkl.gz"
    write(source, fake_mnist())
    monkeypatch.setattr(datasets, "MNIST_SHA256", sha256_of(source))
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(source.read_bytes()))
    dest_dir = tmp_path / "cache"
    train, _, _ = load_mnist(data_dir=dest_dir)
    assert (dest_dir / MNIST_FILENAME).exists() and train[0].shape == (20, 784)
    assert not list(dest_dir.glob("*.part"))


def test_download_rejects_corrupted_data_and_leaves_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(b"not mnist"))
    dest_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="Could not download"):
        load_mnist(data_dir=dest_dir)
    assert not (dest_dir / MNIST_FILENAME).exists() and not list(dest_dir.glob("*.part"))


def test_download_falls_back_to_the_second_mirror(tmp_path, monkeypatch):
    source = tmp_path / "source.pkl.gz"
    write(source, fake_mnist())
    monkeypatch.setattr(datasets, "MNIST_SHA256", sha256_of(source))
    calls = []

    def flaky(url, timeout=0):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("mirror down")
        return io.BytesIO(source.read_bytes())

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    load_mnist(data_dir=tmp_path / "cache")
    assert len(calls) == 2 and calls[0] != calls[1]
