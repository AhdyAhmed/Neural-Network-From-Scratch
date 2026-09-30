"""Layer base class and layers (Dense, Dropout).

Day 2: forward pass only. ``backward`` is implemented on Day 3.
Dropout is added on Day 11.
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Base class for everything that can sit in a network.

    Every layer follows the same two-method contract (see design.md):

    * ``forward(x)``  -> compute the output and cache what backward needs.
    * ``backward(grad_output)`` -> return dL/dx and store dL/dparams (Day 3).
    """

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError("backward() is implemented on Day 3.")

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return ``(value, gradient)`` pairs for the optimizer. No params by default."""
        return []

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(x, training=training)


class Dense(Layer):
    """Fully connected layer:  Z = X @ W + b.

    Shapes (rows = samples, columns = features):
        X: (N, in_features)   W: (in_features, out_features)
        b: (1, out_features)  Z: (N, out_features)

    Weights use a simple small-random initialization for now
    (He / Xavier arrive on Day 9). Pass ``seed`` or ``rng`` for reproducibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers.")

        self.in_features = in_features
        self.out_features = out_features

        rng = rng if rng is not None else np.random.default_rng(seed)
        self.W = rng.normal(loc=0.0, scale=0.01, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))

        # Gradients are filled in by backward() on Day 3.
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._x: np.ndarray | None = None  # cached input for backward

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Dense expected input of shape (N, {self.in_features}), got {x.shape}."
            )
        self._x = x
        return x @ self.W + self.b

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.dW), (self.b, self.db)]

    def __repr__(self) -> str:
        return f"Dense({self.in_features} -> {self.out_features})"
