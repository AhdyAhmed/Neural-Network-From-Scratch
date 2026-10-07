"""Activation and gradient statistics, for spotting vanishing / exploding signals.

    stats = collect_statistics(model, x, y)
    print(format_statistics(stats))
    for warning in diagnose(stats):
        print("WARNING:", warning)

``collect_statistics`` runs one forward and one backward pass on the batch you
give it. It never updates parameters (it does overwrite the layers' stored
gradients, which every training step recomputes anyway).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nn.activations import ReLU, Sigmoid, Tanh
from nn.layers import Dense
from nn.model import Sequential

# Thresholds used by diagnose(). Deliberately loose: they flag real trouble, not mild imbalance.
# (For reference, a He-initialised ReLU net has dense-layer output std ~1 and dW/W between 0.01 and 0.5.)
VANISH_BELOW = 1e-3        # absolute output std below this, or first/last gradient ratio below it
EXPLODE_ABOVE = 1e3        # absolute output std above this, or first/last gradient ratio above it
UPDATE_TOO_SMALL = 1e-6    # rms(dW) / rms(W) below this: weights barely move, learning stalls
UPDATE_TOO_LARGE = 1e2     # rms(dW) / rms(W) above this: a normal step rewrites the weights
DEAD_ZERO_FRACTION = 0.3   # share of ReLU units that output 0 for every sample in the batch
SATURATED_FRACTION = 0.5   # share of tanh / sigmoid outputs pinned near their limits


@dataclass
class LayerStats:
    index: int
    name: str
    out_mean: float
    out_std: float
    out_abs_max: float
    frac_zero: float           # share of outputs that are exactly 0
    frac_dead_units: float     # share of units (columns) that are 0 for the whole batch
    frac_saturated: float      # tanh/sigmoid: share of outputs within 1% of a limit
    grad_in_rms: float         # rms of dL/d(layer input) arriving at this layer
    weight_rms: float | None = None
    weight_grad_rms: float | None = None

    @property
    def update_ratio(self) -> float | None:
        """rms(grad) / rms(weight): how big a gradient step is relative to the weights."""
        if self.weight_rms is None or self.weight_rms == 0:
            return None
        return self.weight_grad_rms / self.weight_rms


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a)))) if a.size else 0.0


def collect_statistics(model: Sequential, x: np.ndarray, y: np.ndarray) -> list[LayerStats]:
    """One forward + backward pass on ``(x, y)``; returns one ``LayerStats`` per layer."""
    if model.loss is None:
        raise RuntimeError("Model must be compiled with a loss before collecting statistics.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    outputs: list[np.ndarray] = []
    y_pred = model.forward(x, training=True, record=outputs)
    outputs = [o.copy() for o in outputs]
    model.loss.forward(y_pred, y)

    grads_reversed: list[np.ndarray] = []
    model.backward_from_loss(record=grads_reversed)
    # backward() records dL/d(layer input) for each layer it processes, last layer first.
    grads_in = list(reversed(grads_reversed))
    if model._is_fused():
        # The fused softmax + cross-entropy path skips the last layer; the gradient it
        # feeds in is exactly dL/d(that layer's input), so grads_in[i] is aligned with layer i again.
        grads_in.append(model.loss.backward_logits())

    stats = []
    for i, (layer, out) in enumerate(zip(model.layers, outputs)):
        if isinstance(layer, Tanh):
            saturated = float(np.mean(np.abs(out) > 0.99))
        elif isinstance(layer, Sigmoid):
            saturated = float(np.mean((out < 0.01) | (out > 0.99)))
        else:
            saturated = 0.0
        entry = LayerStats(
            index=i,
            name=repr(layer),
            out_mean=float(out.mean()),
            out_std=float(out.std()),
            out_abs_max=float(np.abs(out).max()),
            frac_zero=float(np.mean(out == 0)),
            frac_dead_units=float(np.mean(np.all(out == 0, axis=0))) if isinstance(layer, ReLU) else 0.0,
            frac_saturated=saturated,
            grad_in_rms=_rms(grads_in[i]) if i < len(grads_in) else float("nan"),
        )
        if isinstance(layer, Dense):
            entry.weight_rms = _rms(layer.W)
            entry.weight_grad_rms = _rms(layer.dW)
        stats.append(entry)
    return stats


def format_statistics(stats: list[LayerStats]) -> str:
    """A fixed-width table: output statistics on the left, gradient statistics on the right."""
    header = (f"{'#':>2} {'layer':<16} {'out mean':>9} {'out std':>9} {'|out| max':>10} {'%zero':>6} "
              f"{'%sat':>5} | {'grad-in rms':>11} {'weight rms':>11} {'dW rms':>10} {'dW/W':>9}")
    lines = [header, "-" * len(header)]
    for s in stats:
        w = f"{s.weight_rms:>11.3e} {s.weight_grad_rms:>10.3e} {s.update_ratio:>9.2e}" if s.weight_rms is not None else f"{'':>11} {'':>10} {'':>9}"
        lines.append(
            f"{s.index:>2} {s.name:<16} {s.out_mean:>9.2e} {s.out_std:>9.2e} {s.out_abs_max:>10.2e} "
            f"{100 * s.frac_zero:>5.0f}% {100 * s.frac_saturated:>4.0f}% | {s.grad_in_rms:>11.3e} {w}"
        )
    return "\n".join(lines)


def diagnose(stats: list[LayerStats]) -> list[str]:
    """Human-readable warnings (one per kind of problem); an empty list means nothing alarming."""
    warnings: list[str] = []

    bad = [s for s in stats if not all(np.isfinite(v) for v in (s.out_mean, s.out_std, s.out_abs_max, s.grad_in_rms))]
    if bad:
        warnings.append(f"non-finite (NaN/inf) values in outputs or gradients, first at layer {bad[0].index} {bad[0].name}")

    dense = [s for s in stats if s.weight_rms is not None]
    finite = [s for s in dense if np.isfinite(s.out_std)]

    vanishing = [s for s in finite if s.out_std < VANISH_BELOW]
    if vanishing:
        a, b = vanishing[0], vanishing[-1]
        warnings.append(
            f"vanishing signal: {len(vanishing)} Dense layer(s) output std < {VANISH_BELOW:g} "
            f"(layer {a.index}: {a.out_std:.1e} ... layer {b.index}: {b.out_std:.1e})"
        )
    exploding = [s for s in finite if s.out_std > EXPLODE_ABOVE]
    if exploding:
        a, b = exploding[0], exploding[-1]
        warnings.append(
            f"exploding signal: {len(exploding)} Dense layer(s) output std > {EXPLODE_ABOVE:g} "
            f"(layer {a.index}: {a.out_std:.1e} ... layer {b.index}: {b.out_std:.1e})"
        )

    if len(dense) >= 2 and all(np.isfinite(s.grad_in_rms) and s.grad_in_rms > 0 for s in (dense[0], dense[-1])):
        ratio = dense[0].grad_in_rms / dense[-1].grad_in_rms
        if ratio < VANISH_BELOW:
            warnings.append(f"vanishing gradients: gradient reaching the first layer is {ratio:.1e}x the last layer's")
        elif ratio > EXPLODE_ABOVE:
            warnings.append(f"exploding gradients: gradient reaching the first layer is {ratio:.1e}x the last layer's")

    tiny = [s for s in dense if s.update_ratio is not None and np.isfinite(s.update_ratio) and s.update_ratio < UPDATE_TOO_SMALL]
    if tiny:
        warnings.append(
            f"negligible updates: dW/W < {UPDATE_TOO_SMALL:g} in {len(tiny)} layer(s) (e.g. layer {tiny[0].index}: "
            f"{tiny[0].update_ratio:.1e}), so the weights will barely move and learning will stall"
        )
    huge = [s for s in dense if s.update_ratio is not None and np.isfinite(s.update_ratio) and s.update_ratio > UPDATE_TOO_LARGE]
    if huge:
        warnings.append(
            f"huge updates: dW/W > {UPDATE_TOO_LARGE:g} in {len(huge)} layer(s) (e.g. layer {huge[0].index}: "
            f"{huge[0].update_ratio:.1e}), so even a small learning rate would rewrite the weights"
        )

    dead = [s for s in stats if s.frac_dead_units > DEAD_ZERO_FRACTION]
    if dead:
        warnings.append("dead ReLU units: " + ", ".join(f"layer {s.index} ({100 * s.frac_dead_units:.0f}% always 0)" for s in dead))
    saturated = [s for s in stats if s.frac_saturated > SATURATED_FRACTION]
    if saturated:
        warnings.append("saturated units: " + ", ".join(f"layer {s.index} {s.name} ({100 * s.frac_saturated:.0f}% pinned)" for s in saturated))
    return warnings
