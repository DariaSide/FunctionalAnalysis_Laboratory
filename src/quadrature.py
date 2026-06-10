from __future__ import annotations

import numpy as np


def rectangle_weights(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    return np.ones(n) / n


def trapezoid_weights(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    h = 1.0 / (n - 1)
    w = np.ones(n) * h
    w[0] = h / 2.0
    w[-1] = h / 2.0
    return w


def simpson_weights(n: int) -> np.ndarray:
    if n < 3:
        raise ValueError("n must be at least 3")
    if (n - 1) % 2 != 0:
        raise ValueError("Simpson rule requires an odd number of nodes")
    h = 1.0 / (n - 1)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * h / 3.0


def get_weights(n: int, method: str) -> np.ndarray:
    if method == "rectangle":
        return rectangle_weights(n)
    if method == "trapezoid":
        return trapezoid_weights(n)
    if method == "simpson":
        return simpson_weights(n)
    raise ValueError(f"Unknown quadrature method: {method}")
