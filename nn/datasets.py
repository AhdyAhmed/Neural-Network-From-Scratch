"""Dataset loading. Day 8: MNIST.

MNIST is downloaded once from a public GitHub mirror (Michael Nielsen's
``neural-networks-and-deep-learning`` repository, the same dataset referenced in
design.md), cached under ``data/`` (git-ignored), and checked against a SHA-256
checksum before it is parsed.

The file is a Python pickle, and unpickling untrusted data can execute code. So:
  1. the download is verified against a pinned checksum, and
  2. it is parsed with a *restricted* unpickler that only allows the three
     NumPy classes the file actually uses; anything else raises an error.
"""

from __future__ import annotations

import gzip
import hashlib
import pickle
import shutil
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np

MNIST_FILENAME = "mnist.pkl.gz"
MNIST_SHA256 = "f11bb9e41d6c1b6c124aa38fd605497bdcfe2ee08cf7c2bb5a41ab5d759e1416"
MNIST_URLS = (
    "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz",
    "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz",
)

Split = tuple[np.ndarray, np.ndarray]


class _RestrictedUnpickler(pickle.Unpickler):
    """Only allows the NumPy objects that appear in the MNIST pickle."""

    _ALLOWED = {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"Refusing to load unexpected object {module}.{name}.")
        if module == "numpy.core.multiarray":  # renamed in NumPy 2; avoids a deprecation warning
            try:
                return super().find_class("numpy._core.multiarray", name)
            except (ImportError, AttributeError):
                pass
        return super().find_class(module, name)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mnist_pickle(path: Path) -> tuple[Split, Split, Split]:
    """Parse an MNIST pickle file with the restricted unpickler and validate its structure."""
    with gzip.open(path, "rb") as f, warnings.catch_warnings():
        # The old pickle passes align=0 (an int) to np.dtype, which newer NumPy warns about. Harmless,
        # and the filter only applies inside this block.
        warnings.simplefilter("ignore")
        # encoding="latin1" is required: the file was written by Python 2.
        data = _RestrictedUnpickler(f, encoding="latin1").load()

    if not (isinstance(data, tuple) and len(data) == 3):
        raise ValueError("Unexpected MNIST file layout: expected (train, validation, test).")
    splits = []
    for part in data:
        if not (isinstance(part, tuple) and len(part) == 2):
            raise ValueError("Unexpected MNIST file layout: each split must be (images, labels).")
        x, y = np.asarray(part[0]), np.asarray(part[1])
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError(f"Unexpected MNIST array shapes: {x.shape}, {y.shape}.")
        splits.append((x, y.astype(np.int64)))
    return tuple(splits)  # type: ignore[return-value]


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors = []
    for url in MNIST_URLS:
        try:
            print(f"Downloading MNIST from {url} ...", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out)
            if sha256_of(tmp) != MNIST_SHA256:
                raise ValueError("checksum mismatch")
            tmp.replace(dest)
            return
        except Exception as exc:  # try the next mirror
            errors.append(f"  {url}: {exc}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError(
        "Could not download MNIST:\n" + "\n".join(errors) + f"\nDownload {MNIST_URLS[0]} manually "
        f"and place it at {dest}."
    )


def load_mnist(
    data_dir: str | Path = "data",
    download: bool = True,
    verify: bool = True,
) -> tuple[Split, Split, Split]:
    """Load MNIST as ``((x_train, y_train), (x_val, y_val), (x_test, y_test))``.

    * ``x``: float32 array of shape (N, 784), pixel values scaled to [0, 1]
    * ``y``: int64 array of shape (N,), class labels 0-9
    * split sizes: 50,000 train / 10,000 validation / 10,000 test

    The file is cached in ``data_dir``. ``download=False`` raises
    ``FileNotFoundError`` instead of fetching it (useful offline / in CI).
    ``verify=False`` skips the checksum (only for your own test files).
    """
    path = Path(data_dir) / MNIST_FILENAME
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found (download disabled).")
        _download(path)
    if verify and sha256_of(path) != MNIST_SHA256:
        raise ValueError(f"{path} does not match the expected checksum; delete it to re-download.")
    return parse_mnist_pickle(path)
