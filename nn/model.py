"""Sequential container with fit / predict / evaluate.

Day 4: full-batch training.  Day 7: shuffling, mini-batches, and the fused
softmax + cross-entropy backward pass.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from nn.activations import Softmax
from nn.layers import Dense, Layer
from nn.losses import CategoricalCrossEntropy, Loss
from nn.optimizers import Optimizer
from nn.utils import iterate_minibatches


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
        self.l2 = 0.0

    # ------------------------------------------------------------ setup
    def compile(self, loss: Loss, optimizer: Optimizer, l2: float = 0.0) -> None:
        """Configure training. ``l2`` is the non-negative L2 weight penalty.

        The objective adds ``0.5 * l2 * sum(W**2)`` over Dense weights; biases
        are not regularized. Its gradient ``l2 * W`` is added to each ``dW``.
        The default ``l2=0`` preserves the original behavior.
        """
        if not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be a finite non-negative number.")
        self.loss = loss
        self.optimizer = optimizer
        self.l2 = float(l2)

    def _regularization_penalty(self) -> float:
        return 0.5 * self.l2 * sum(float(np.sum(layer.W ** 2))
                                  for layer in self.layers if isinstance(layer, Dense))

    def _add_regularization_gradients(self) -> None:
        if self.l2:
            for layer in self.layers:
                if isinstance(layer, Dense):
                    layer.dW += self.l2 * layer.W

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
    def forward(self, x: np.ndarray, training: bool = True, record: list | None = None) -> np.ndarray:
        """Run the layers in order. If ``record`` is a list, every layer's output is appended to it
        (used by ``nn.diagnostics``; the arrays are the live outputs, so copy them if you keep them)."""
        for layer in self.layers:
            x = layer.forward(x, training=training)
            if record is not None:
                record.append(x)
        return x

    def backward(self, grad: np.ndarray, skip_last: bool = False, record: list | None = None) -> np.ndarray:
        """Chain rule through the layers in reverse.

        ``skip_last=True`` starts at the second-to-last layer (used by the fused
        softmax + cross-entropy path, where ``grad`` is already w.r.t. the logits).
        If ``record`` is a list, the gradient w.r.t. each processed layer's input is
        appended to it as backprop proceeds (so in reverse layer order).
        """
        layers = self.layers[:-1] if skip_last else self.layers
        for layer in reversed(layers):
            grad = layer.backward(grad)
            if record is not None:
                record.append(grad)
        return grad

    def backward_from_loss(self, record: list | None = None) -> np.ndarray:
        """Backpropagate starting from the loss that was just computed.

        If the network ends in ``Softmax`` and the loss is
        ``CategoricalCrossEntropy``, the two are fused: the gradient w.r.t. the
        logits is simply ``(p - y) / N`` and the Softmax layer is skipped.
        Otherwise this is ``backward(loss.backward())``.

        ``record``: optional list that receives the gradient arriving at each layer's
        *input*, in backward order (see ``backward``).
        """
        self._require_compiled()
        if self._is_fused():
            return self.backward(self.loss.backward_logits(), skip_last=True, record=record)
        return self.backward(self.loss.backward(), record=record)

    def _is_fused(self) -> bool:
        return isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossEntropy)

    # ------------------------------------------------------- training
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """One forward pass, one backward pass, one optimizer update. Returns the loss."""
        return self._train_step(x, y)[0]

    def _train_step(self, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        self._require_compiled()
        y_pred = self.forward(x, training=True)
        loss_value = self.loss.forward(y_pred, y) + self._regularization_penalty()
        self.backward_from_loss()
        self._add_regularization_gradients()
        self.optimizer.step(self.params())
        return loss_value, y_pred

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
        verbose: int = 1,
        log_every: int | None = None,
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> dict[str, list[float]]:
        """Train the model. Returns a history dict.

        batch_size  Samples per parameter update. ``None`` (or >= N) means
                    full-batch gradient descent. Smaller batches give more
                    (noisier) updates per epoch: this is the "stochastic" in SGD.
        shuffle     Reshuffle the samples every epoch (only matters when there
                    is more than one batch per epoch).
        seed        Seeds the shuffling, for reproducible runs.

        history["loss"]     training loss per epoch: the average of the batch
                            losses seen during that epoch, weighted by batch size
        history["val_loss"] validation loss per epoch (only if validation_data given)

        metrics     Optional ``{"name": fn(y_pred, y_true) -> float}``; histories
                    are recorded per epoch for training and validation.
        early_stopping  Stop when validation loss fails to improve; requires validation_data.
        patience    Number of consecutive non-improving epochs tolerated.
        min_delta   Minimum decrease in validation loss considered an improvement.
        restore_best_weights  Restore parameters from the best validation epoch.
        """
        self._require_compiled()
        x, y = self._check_data(x, y)
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if batch_size is not None and (not isinstance(batch_size, (int, np.integer)) or batch_size <= 0):
            raise ValueError("batch_size must be a positive integer or None.")
        if not isinstance(early_stopping, (bool, np.bool_)):
            raise ValueError("early_stopping must be a boolean.")
        if early_stopping and validation_data is None:
            raise ValueError("early_stopping=True requires validation_data.")
        if not isinstance(patience, (int, np.integer)) or patience <= 0:
            raise ValueError("patience must be a positive integer.")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a finite non-negative number.")
        n = x.shape[0]
        batch_size = n if batch_size is None else min(int(batch_size), n)
        rng = np.random.default_rng(seed)
        if validation_data is not None:
            x_val, y_val = self._check_data(*validation_data)

        log_every = log_every or max(1, epochs // 10)
        metrics = metrics or {}
        history: dict[str, list[float]] = {"loss": []}
        for name in metrics:
            history[name] = []
        if validation_data is not None:
            history["val_loss"] = []
            for name in metrics:
                history["val_" + name] = []

        start = time.perf_counter()
        best_val_loss = float("inf")
        best_params: list[tuple[np.ndarray, np.ndarray]] | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            totals = {name: 0.0 for name in metrics}
            if batch_size == n:  # full batch: no shuffling needed, one update per epoch
                epoch_loss, y_pred = self._train_step(x, y)
                for name, fn in metrics.items():
                    totals[name] = fn(y_pred, y)
            else:
                loss_sum = 0.0
                for xb, yb in iterate_minibatches(x, y, batch_size, shuffle=shuffle, rng=rng):
                    batch_loss, y_pred = self._train_step(xb, yb)
                    loss_sum += batch_loss * len(xb)
                    for name, fn in metrics.items():
                        totals[name] += fn(y_pred, yb) * len(xb)
                epoch_loss = loss_sum / n
                totals = {name: total / n for name, total in totals.items()}
            history["loss"].append(epoch_loss)
            for name in metrics:
                history[name].append(totals[name])
            if validation_data is not None:
                val_pred = self.predict(x_val)  # one forward pass serves the loss and every metric
                history["val_loss"].append(
                    self.loss.forward(val_pred, y_val) + self._regularization_penalty()
                )
                for name, fn in metrics.items():
                    history["val_" + name].append(fn(val_pred, y_val))
                if early_stopping:
                    current_val_loss = history["val_loss"][-1]
                    if current_val_loss < best_val_loss - min_delta:
                        best_val_loss = current_val_loss
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        best_params = [(value.copy(), grad.copy()) for value, grad in self.params()]
                    else:
                        epochs_without_improvement += 1

            if verbose and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                msg = f"epoch {epoch:>4}/{epochs}  loss: {history['loss'][-1]:.6f}"
                for name in metrics:
                    msg += f"  {name}: {history[name][-1]:.4f}"
                if validation_data is not None:
                    msg += f"  val_loss: {history['val_loss'][-1]:.6f}"
                    for name in metrics:
                        msg += f"  val_{name}: {history['val_' + name][-1]:.4f}"
                print(f"{msg}  ({time.perf_counter() - start:.2f}s)")
            if early_stopping and epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}; best validation loss {best_val_loss:.6f} at epoch {best_epoch}.")
                break
        if early_stopping and restore_best_weights and best_params is not None:
            for (parameter, _), (best_value, _) in zip(self.params(), best_params):
                parameter[...] = best_value
        return history

    # ------------------------------------------------------ inference
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass in evaluation mode (no parameter changes)."""
        return self.forward(np.asarray(x, dtype=float), training=False)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Loss on the given data, without updating anything."""
        self._require_compiled()
        x, y = self._check_data(x, y)
        return self.loss.forward(self.predict(x), y) + self._regularization_penalty()

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
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (N, outputs), got shape {y.shape}.")
        if x.shape[0] == 0:
            raise ValueError("x and y must contain at least one sample.")
        if x.shape[1] == 0 or y.shape[1] == 0:
            raise ValueError("x and y must each contain at least one feature/output column.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have the same number of samples, got {x.shape[0]} and {y.shape[0]}.")
        return x, y

    def __repr__(self) -> str:
        return "Sequential([" + ", ".join(repr(l) for l in self.layers) + "])"
