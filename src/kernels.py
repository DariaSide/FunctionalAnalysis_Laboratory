from __future__ import annotations

import numpy as np


def brownian_kernel(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    return np.minimum.outer(x, t)


def make_uniform_grid(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    return np.linspace(0.0, 1.0, n)


def build_gram_matrix(x: np.ndarray) -> np.ndarray:
    return brownian_kernel(x, x)


def symmetry_error(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix - matrix.T)))


def random_quadratic_forms(
    matrix: np.ndarray,
    n_trials: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_trials):
        z = rng.normal(size=matrix.shape[0])
        values.append(z @ matrix @ z)
    return np.array(values)
