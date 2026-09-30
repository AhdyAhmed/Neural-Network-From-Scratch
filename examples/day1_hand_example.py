"""Day 1: a single sigmoid neuron, worked by hand and verified with NumPy.

Model:   z = w . x + b,   a = sigmoid(z),   L = 0.5 * (a - y)^2

Run:  python examples/day1_hand_example.py
"""

import numpy as np

# Fixed toy example (same numbers as notes/day1_math_refresher.md)
X = np.array([1.0, 2.0])
W = np.array([0.5, -0.5])
B = 0.1
Y = 1.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(x, w, b):
    """Return the pre-activation z and the activation a."""
    z = x @ w + b
    return z, sigmoid(z)


def loss(a, y):
    """Half squared error."""
    return 0.5 * (a - y) ** 2


def gradients(x, w, b, y):
    """Backprop by the chain rule: dL/dw = dL/da * da/dz * dz/dw."""
    _, a = forward(x, w, b)
    dL_da = a - y
    da_dz = a * (1.0 - a)
    dL_dz = dL_da * da_dz
    dw = dL_dz * x          # dz/dw = x
    db = dL_dz              # dz/db = 1
    return dw, db


def main():
    z, a = forward(X, W, B)
    dw, db = gradients(X, W, B, Y)

    print("=== Forward pass ===")
    print(f"z = {z:.6f}")
    print(f"a = sigmoid(z) = {a:.6f}")
    print(f"L = {loss(a, Y):.6f}")

    print("\n=== Backward pass (analytic) ===")
    print(f"dL/dw = {dw}")
    print(f"dL/db = {db:.6f}")

    # Numerical check with centered finite differences
    eps = 1e-6

    def f(w, b):
        return loss(forward(X, w, b)[1], Y)

    num_dw = np.array([
        (f(W + eps * np.eye(2)[i], B) - f(W - eps * np.eye(2)[i], B)) / (2 * eps)
        for i in range(2)
    ])
    num_db = (f(W, B + eps) - f(W, B - eps)) / (2 * eps)

    print("\n=== Backward pass (numerical) ===")
    print(f"dL/dw = {num_dw}")
    print(f"dL/db = {num_db:.6f}")

    print("\nMax abs difference:",
          max(np.abs(dw - num_dw).max(), abs(db - num_db)))

    # One gradient-descent step: the loss should go down
    lr = 0.5
    w_new, b_new = W - lr * dw, B - lr * db
    print(f"\nAfter one SGD step (lr={lr}): L = {f(w_new, b_new):.6f} "
          f"(was {loss(a, Y):.6f})")


if __name__ == "__main__":
    main()
