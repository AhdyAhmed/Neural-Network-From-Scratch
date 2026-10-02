"""Sequential container with fit / predict / evaluate.

Day 4: full-batch training. Mini-batching and shuffling arrive on Day 7.
"""

from __future__ import annotations

import time

import numpy as np

from nn.layers import Layer
from nn.losses import Loss
from nn.optimizers import Optimizer


class Sequential:
    """A stack of layers applied in order (forward) and in reverse (backward).

        model = Sequential([Dense(1, 8), Tanh(), Dense(8, 1)])
        model.compile(loss=MSE(), optimizer=SGD(lr=0.1))
        history = model.fit(X, y, epochs=300)
        preds = model.predict(X)
    """

    def __init__(self, layers: list[Layer]) -> None:
        if not layers:
            raise ValueError("Sequential needs at least one layer.")
        self.layers = list(layers)
        self.loss: Loss | None = None
        self.optimizer: Optimizer | None = None

    # ------------------------------------------------------------ setup
    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        self.loss = loss
        self.optimizer = optimizer

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [p for layer in self.layers for p in layer.params()]

    def summary(self) -> str:
        lines = ["Layer                      Params", "-" * 34]
        total = 0
        for layer in self.layers:
            n = sum(v.size for v, _ in layer.params())
            total += n
            lines.append(f"{layer!r:<26} {n:>6}")
        lines += ["-" * 34, f"{'Total':<26} {total:>6}"]
        return "\n".join(lines)

    # ---------------------------------------------------------- passes
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # ------------------------------------------------------- training
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """One forward pass, one backward pass, one optimizer update. Returns the loss."""
        self._require_compiled()
        y_pred = self.forward(x, training=True)
        loss_value = self.loss.forward(y_pred, y)
        self.backward(self.loss.backward())
        self.optimizer.step(self.params())
        return loss_value

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: int = 1,
        log_every: int | None = None,
    ) -> dict[str, list[float]]:
        """Train on the full dataset each epoch. Returns a history dict.

        history["loss"]     training loss per epoch
        history["val_loss"] validation loss per epoch (only if validation_data given)
        """
        self._require_compiled()
        x, y = self._check_data(x, y)
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if validation_data is not None:
            x_val, y_val = self._check_data(*validation_data)

        log_every = log_every or max(1, epochs // 10)
        history: dict[str, list[float]] = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []

        start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            history["loss"].append(self.train_step(x, y))
            if validation_data is not None:
                history["val_loss"].append(self.evaluate(x_val, y_val))

            if verbose and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                msg = f"epoch {epoch:>4}/{epochs}  loss: {history['loss'][-1]:.6f}"
                if validation_data is not None:
                    msg += f"  val_loss: {history['val_loss'][-1]:.6f}"
                print(f"{msg}  ({time.perf_counter() - start:.2f}s)")
        return history

    # ------------------------------------------------------ inference
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass in evaluation mode (no parameter changes)."""
        return self.forward(np.asarray(x, dtype=float), training=False)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Loss on the given data, without updating anything."""
        self._require_compiled()
        x, y = self._check_data(x, y)
        return self.loss.forward(self.predict(x), y)

    # --------------------------------------------------------- helpers
    def _require_compiled(self) -> None:
        if self.loss is None or self.optimizer is None:
            raise RuntimeError("Call compile(loss=..., optimizer=...) before training or evaluating.")

    @staticmethod
    def _check_data(x, y) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:  # convenience: treat a flat target vector as (N, 1)
            y = y.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (N, features), got shape {x.shape}.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have the same number of samples, got {x.shape[0]} and {y.shape[0]}.")
        return x, y

    def __repr__(self) -> str:
        return "Sequential([" + ", ".join(repr(l) for l in self.layers) + "])"
