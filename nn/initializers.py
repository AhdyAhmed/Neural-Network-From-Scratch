"""Weight initializers (Xavier/Glorot, He).

Why initialization matters
--------------------------
Each Dense layer multiplies its input by W. If W is too small the signal shrinks
layer by layer (vanishing activations and gradients); if too large it grows
(exploding, or saturated sigmoid/tanh units). A good scheme keeps the *variance*
of activations roughly constant from layer to layer:

* Xavier / Glorot  Var(W) = 2 / (fan_in + fan_out)   suits tanh / sigmoid / softmax
* He               Var(W) = 2 / fan_in               suits ReLU (which zeroes half the signal)

Every initializer has the signature ``fn(fan_in, fan_out, rng) -> array of shape
(fan_in, fan_out)``. Biases are always initialised to zero by ``Dense``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Initializer = Callable[[int, int, np.random.Generator], np.ndarray]


def _check(fan_in: int, fan_out: int) -> None:
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive integers.")


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """N(0, 2 / fan_in)."""
    _check(fan_in, fan_out)
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def he_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """U(-a, a) with a = sqrt(6 / fan_in)  (variance 2 / fan_in)."""
    _check(fan_in, fan_out)
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def xavier_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """N(0, 2 / (fan_in + fan_out))."""
    _check(fan_in, fan_out)
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_in, fan_out))


def xavier_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """U(-a, a) with a = sqrt(6 / (fan_in + fan_out))  (variance 2 / (fan_in + fan_out))."""
    _check(fan_in, fan_out)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def normal(scale: float) -> Initializer:
    """Plain N(0, scale^2): the simple scheme used on Days 2 to 8 (``init_scale``)."""
    if scale <= 0:
        raise ValueError("scale must be positive.")

    def init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
        _check(fan_in, fan_out)
        return rng.normal(0.0, scale, size=(fan_in, fan_out))

    init.__name__ = f"normal({scale})"
    return init


_REGISTRY: dict[str, Initializer] = {
    "he_normal": he_normal,
    "he_uniform": he_uniform,
    "xavier_normal": xavier_normal,
    "xavier_uniform": xavier_uniform,
    "he": he_normal,
    "xavier": xavier_normal,
    "glorot": xavier_normal,
    "glorot_normal": xavier_normal,
    "glorot_uniform": xavier_uniform,
}


def get_initializer(spec: str | Initializer) -> Initializer:
    """Resolve a name like ``"he"`` or ``"xavier_uniform"`` (or pass a callable through)."""
    if callable(spec):
        return spec
    try:
        return _REGISTRY[spec.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"Unknown initializer {spec!r}. Choose from: {sorted(_REGISTRY)}.") from None


def initializer_for(activation) -> Initializer:
    """The recommended initializer for the layer that *feeds* ``activation``.

    ReLU  -> He.   Tanh / Sigmoid / Softmax / None / anything else -> Xavier.
    ``activation`` may be an instance, a class, or a name such as ``"relu"``.
    """
    if isinstance(activation, str):
        name = activation
    elif isinstance(activation, type):
        name = activation.__name__
    else:
        name = type(activation).__name__ if activation is not None else ""
    return he_normal if name.lower() == "relu" else xavier_normal
