from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kernels import build_gram_matrix, make_uniform_grid, random_quadratic_forms, symmetry_error
from quadrature import get_weights
from spectral_problem import (
    compute_discrete_spectrum,
    eigenfunction_error_table,
    exact_eigenvalues,
    frobenius_error,
    normalize_discrete_functions,
    orient_function_signs,
    quadrature_comparison_table,
    spectrum_error_table,
    truncated_kernel_approximation,
)
from plotting import ensure_dir, plot_eigenfunctions, plot_kernel_heatmap, plot_spectrum_errors


ROOT_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT_DIR / "results" / "tables"
FIGURE_DIR = ROOT_DIR / "results" / "figures"
RANDOM_STATE = 42


def run_case1() -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DIR)

    x_gram = make_uniform_grid(50)
    gram = build_gram_matrix(x_gram)
    gram_eigvals = np.linalg.eigvalsh(gram)
    quadratic_forms = random_quadratic_forms(gram, n_trials=1000, seed=RANDOM_STATE)

    gram_summary = pd.DataFrame(
        [
            {
                "n": 50,
                "symmetry_error": symmetry_error(gram),
                "min_eigenvalue": gram_eigvals.min(),
                "min_random_quadratic_form": quadratic_forms.min(),
                "negative_quadratic_forms": int(np.sum(quadratic_forms < -1e-10)),
            }
        ]
    )
    gram_summary.to_csv(TABLE_DIR / "case1_gram_summary.csv", index=False)

    n_values = [21, 51, 101, 201]
    spectrum_errors = spectrum_error_table(n_values, method="trapezoid", k_max=5)
    spectrum_errors.to_csv(TABLE_DIR / "case1_spectrum_errors.csv", index=False)

    quadrature_errors = quadrature_comparison_table(
        n_values=n_values,
        methods=["rectangle", "trapezoid", "simpson"],
        k_max=5,
    )
    quadrature_errors.to_csv(TABLE_DIR / "case1_quadrature_comparison.csv", index=False)

    function_errors = eigenfunction_error_table(n_values, method="trapezoid", k_max=4)
    function_errors.to_csv(TABLE_DIR / "case1_eigenfunction_errors.csv", index=False)

    n = 101
    x = make_uniform_grid(n)
    w = get_weights(n, "trapezoid")
    eig_num, phi_num = compute_discrete_spectrum(x, w)
    phi_num = normalize_discrete_functions(phi_num, w)
    phi_num = orient_function_signs(phi_num, x)

    eig_table = pd.DataFrame(
        {
            "k": np.arange(1, 7),
            "lambda_exact": exact_eigenvalues(6),
            "lambda_num": eig_num[:6],
            "abs_error": np.abs(eig_num[:6] - exact_eigenvalues(6)),
        }
    )
    eig_table.to_csv(TABLE_DIR / "case1_eigenvalues_n101.csv", index=False)

    plot_eigenfunctions(x, phi_num, FIGURE_DIR / "case1_eigenfunctions.pdf", k_max=4)
    plot_spectrum_errors(spectrum_errors, FIGURE_DIR / "case1_spectrum_errors.pdf")

    x_kernel = make_uniform_grid(100)
    k_true = build_gram_matrix(x_kernel)
    rows = []
    for m in [1, 2, 5, 10, 20, 40]:
        k_approx = truncated_kernel_approximation(x_kernel, m)
        rows.append({"m": m, "frobenius_error": frobenius_error(k_true, k_approx)})
    kernel_errors = pd.DataFrame(rows)
    kernel_errors.to_csv(TABLE_DIR / "case1_kernel_approximation_errors.csv", index=False)

    plot_kernel_heatmap(k_true, FIGURE_DIR / "case1_kernel_exact.pdf", "Exact kernel K(x,t)=min(x,t)")
    plot_kernel_heatmap(
        truncated_kernel_approximation(x_kernel, 10),
        FIGURE_DIR / "case1_kernel_approximation_m10.pdf",
        "Truncated kernel approximation, m=10",
    )

if __name__ == "__main__":
    run_case1()
