from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectral_problem import exact_eigenfunction


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_eigenfunctions(x: np.ndarray, phi_num: np.ndarray, output_path: str | Path, k_max: int = 4) -> None:
    fig, axes = plt.subplots(k_max, 1, figsize=(7, 2.8 * k_max), constrained_layout=True)
    if k_max == 1:
        axes = [axes]
    for k, ax in enumerate(axes, start=1):
        ax.plot(x, exact_eigenfunction(x, k), label=f"exact phi_{k}")
        ax.plot(x, phi_num[:, k - 1], "--", label=f"numeric phi_{k}")
        ax.set_title(f"Eigenfunction {k}")
        ax.set_xlabel("x")
        ax.set_ylabel("phi(x)")
        ax.grid(True)
        ax.legend()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_errors(error_table: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in sorted(error_table["k"].unique()):
        part = error_table[error_table["k"] == k]
        ax.plot(part["n"], part["abs_error"], marker="o", label=f"lambda_{k}")
    ax.set_yscale("log")
    ax.set_title("Eigenvalue error by grid size")
    ax.set_xlabel("n")
    ax.set_ylabel("absolute error")
    ax.grid(True)
    ax.legend()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_heatmap(matrix: np.ndarray, output_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("grid index")
    ax.set_ylabel("grid index")
    fig.colorbar(image, ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(results["model_name"], results["test_rmse"])
    ax.set_title("Model comparison by test RMSE")
    ax.set_xlabel("model")
    ax.set_ylabel("test RMSE")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_synthetic_scatter(x: np.ndarray, y: np.ndarray, output_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y, s=18)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(scatter, ax=ax, label="y")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
