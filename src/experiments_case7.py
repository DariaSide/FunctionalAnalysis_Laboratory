from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from additive_models import (
    additive_polynomial_features,
    centered_component_values,
    compare_models,
    make_additive_dataset,
    make_interaction_dataset,
)
from plotting import ensure_dir, plot_model_comparison, plot_synthetic_scatter


ROOT_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT_DIR / "results" / "tables"
FIGURE_DIR = ROOT_DIR / "results" / "figures"
RANDOM_STATE = 42


def fit_centered_components_table(x, y, degree: int = 7) -> pd.DataFrame:
    x_features = additive_polynomial_features(x, degree)
    x_train, _, y_train, _ = train_test_split(
        x_features, y, test_size=0.25, random_state=RANDOM_STATE
    )
    x_original_train, _, _, _ = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE
    )
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    return centered_component_values(model, x_original_train, degree)


def run_case7() -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DIR)

    x_additive, y_additive, _ = make_additive_dataset(random_state=RANDOM_STATE)
    x_interaction, y_interaction, _ = make_interaction_dataset(random_state=RANDOM_STATE + 1)

    additive_results = compare_models(
        x_additive,
        y_additive,
        additive_degree=7,
        full_degree=3,
        ridge_alpha=1.0,
        random_state=RANDOM_STATE,
    )
    interaction_results = compare_models(
        x_interaction,
        y_interaction,
        additive_degree=7,
        full_degree=3,
        ridge_alpha=1.0,
        random_state=RANDOM_STATE,
    )

    additive_results.to_csv(TABLE_DIR / "case7_additive_dataset_model_comparison.csv", index=False)
    interaction_results.to_csv(TABLE_DIR / "case7_interaction_dataset_model_comparison.csv", index=False)

    centered = fit_centered_components_table(x_additive, y_additive, degree=7)
    centered.to_csv(TABLE_DIR / "case7_centered_components.csv", index=False)

    plot_model_comparison(additive_results, FIGURE_DIR / "case7_additive_model_comparison.pdf")
    plot_model_comparison(interaction_results, FIGURE_DIR / "case7_interaction_model_comparison.pdf")
    plot_synthetic_scatter(
        x_additive,
        y_additive,
        FIGURE_DIR / "case7_additive_dataset.pdf",
        "Additive synthetic dataset",
    )
    plot_synthetic_scatter(
        x_interaction,
        y_interaction,
        FIGURE_DIR / "case7_interaction_dataset.pdf",
        "Interaction synthetic dataset",
    )

if __name__ == "__main__":
    run_case7()
