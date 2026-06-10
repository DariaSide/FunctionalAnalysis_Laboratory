from __future__ import annotations

import numpy as np
import pandas as pd

from kernels import brownian_kernel, make_uniform_grid
from quadrature import get_weights


def exact_eigenvalues(k_max: int) -> np.ndarray:
    k = np.arange(1, k_max + 1)
    return 1.0 / (np.pi**2 * (k - 0.5) ** 2)


def exact_eigenfunction(x: np.ndarray, k: int) -> np.ndarray:
    return np.sqrt(2.0) * np.sin((k - 0.5) * np.pi * x)


def build_operator_matrices(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_matrix = brownian_kernel(x, x)
    a_matrix = k_matrix * w[None, :]
    sqrt_w = np.sqrt(w)
    b_matrix = sqrt_w[:, None] * k_matrix * sqrt_w[None, :]
    return k_matrix, a_matrix, b_matrix


def compute_discrete_spectrum(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, _, b_matrix = build_operator_matrices(x, w)
    eigvals, eigvecs = np.linalg.eigh(b_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    phi = eigvecs / np.sqrt(w[:, None])
    return eigvals, phi


def discrete_l2_norm(phi: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.sum(phi**2 * w)))


def normalize_discrete_functions(phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    phi = phi.copy()
    for k in range(phi.shape[1]):
        norm = discrete_l2_norm(phi[:, k], w)
        if norm > 0:
            phi[:, k] /= norm
    return phi


def orient_function_signs(phi: np.ndarray, x: np.ndarray) -> np.ndarray:
    phi = phi.copy()
    midpoint = np.argmin(np.abs(x - 0.5))
    for k in range(phi.shape[1]):
        if phi[midpoint, k] < 0:
            phi[:, k] *= -1
    return phi


def eigenfunction_error(phi_num: np.ndarray, phi_exact: np.ndarray, w: np.ndarray) -> float:
    err_plus = np.sqrt(np.sum((phi_num - phi_exact) ** 2 * w))
    err_minus = np.sqrt(np.sum((phi_num + phi_exact) ** 2 * w))
    return float(min(err_plus, err_minus))


def spectrum_error_table(
    n_values: list[int],
    method: str = "trapezoid",
    k_max: int = 5,
) -> pd.DataFrame:
    rows = []
    exact = exact_eigenvalues(k_max)
    for n in n_values:
        x = make_uniform_grid(n)
        w = get_weights(n, method)
        eig_num, _ = compute_discrete_spectrum(x, w)
        for k in range(1, k_max + 1):
            rows.append(
                {
                    "method": method,
                    "n": n,
                    "k": k,
                    "lambda_exact": exact[k - 1],
                    "lambda_num": eig_num[k - 1],
                    "abs_error": abs(eig_num[k - 1] - exact[k - 1]),
                }
            )
    return pd.DataFrame(rows)


def quadrature_comparison_table(
    n_values: list[int],
    methods: list[str],
    k_max: int = 5,
) -> pd.DataFrame:
    frames = []
    for method in methods:
        valid_n = [n for n in n_values if method != "simpson" or (n - 1) % 2 == 0]
        frames.append(spectrum_error_table(valid_n, method=method, k_max=k_max))
    return pd.concat(frames, ignore_index=True)


def eigenfunction_error_table(
    n_values: list[int],
    method: str = "trapezoid",
    k_max: int = 4,
) -> pd.DataFrame:
    rows = []
    for n in n_values:
        x = make_uniform_grid(n)
        w = get_weights(n, method)
        _, phi = compute_discrete_spectrum(x, w)
        phi = normalize_discrete_functions(phi, w)
        phi = orient_function_signs(phi, x)
        for k in range(1, k_max + 1):
            exact = exact_eigenfunction(x, k)
            rows.append(
                {
                    "method": method,
                    "n": n,
                    "k": k,
                    "function_error": eigenfunction_error(phi[:, k - 1], exact, w),
                }
            )
    return pd.DataFrame(rows)


def truncated_kernel_approximation(x: np.ndarray, m: int) -> np.ndarray:
    k_approx = np.zeros((len(x), len(x)))
    lambdas = exact_eigenvalues(m)
    for k in range(1, m + 1):
        phi_k = exact_eigenfunction(x, k)
        k_approx += lambdas[k - 1] * np.outer(phi_k, phi_k)
    return k_approx


def frobenius_error(k_true: np.ndarray, k_approx: np.ndarray) -> float:
    return float(np.linalg.norm(k_true - k_approx, ord="fro"))
