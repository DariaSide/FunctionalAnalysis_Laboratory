from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class RegressionResult:
    model_name: str
    train_rmse: float
    test_rmse: float
    train_r2: float
    test_r2: float
    n_features: int


def make_additive_dataset(
    n_samples: int = 700,
    noise_std: float = 0.10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(-2.0, 2.0, n_samples)
    x2 = rng.uniform(-2.0, 2.0, n_samples)
    y_clean = 1.0 + np.sin(2.0 * x1) + 0.35 * x2**2 - 0.4 * x2
    y = y_clean + rng.normal(0.0, noise_std, n_samples)
    x = np.column_stack([x1, x2])
    return x, y, y_clean


def make_interaction_dataset(
    n_samples: int = 700,
    noise_std: float = 0.12,
    random_state: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(-2.0, 2.0, n_samples)
    x2 = rng.uniform(-2.0, 2.0, n_samples)
    y_clean = 0.7 * np.sin(2.0 * x1 * x2) + 0.25 * x1**2 - 0.3 * x2
    y = y_clean + rng.normal(0.0, noise_std, n_samples)
    x = np.column_stack([x1, x2])
    return x, y, y_clean


def additive_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    columns = []
    for feature_index in range(x.shape[1]):
        feature = x[:, feature_index]
        for power in range(1, degree + 1):
            columns.append(feature**power)
    return np.column_stack(columns)


def full_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    transformer = PolynomialFeatures(degree=degree, include_bias=False)
    return transformer.fit_transform(x)


def fit_and_score(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model,
    random_state: int = 42,
) -> RegressionResult:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=random_state
    )
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    return RegressionResult(
        model_name=model_name,
        train_rmse=float(np.sqrt(mean_squared_error(y_train, pred_train))),
        test_rmse=float(np.sqrt(mean_squared_error(y_test, pred_test))),
        train_r2=float(r2_score(y_train, pred_train)),
        test_r2=float(r2_score(y_test, pred_test)),
        n_features=x.shape[1],
    )


def compare_models(
    x: np.ndarray,
    y: np.ndarray,
    additive_degree: int = 7,
    full_degree: int = 3,
    ridge_alpha: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    results = []

    results.append(
        fit_and_score(
            x,
            y,
            "linear",
            LinearRegression(),
            random_state=random_state,
        )
    )

    x_add = additive_polynomial_features(x, additive_degree)
    results.append(
        fit_and_score(
            x_add,
            y,
            f"additive polynomial degree={additive_degree}",
            LinearRegression(),
            random_state=random_state,
        )
    )

    results.append(
        fit_and_score(
            x_add,
            y,
            f"additive ridge degree={additive_degree}",
            Ridge(alpha=ridge_alpha),
            random_state=random_state,
        )
    )

    x_full = full_polynomial_features(x, full_degree)
    results.append(
        fit_and_score(
            x_full,
            y,
            f"full polynomial degree={full_degree}",
            LinearRegression(),
            random_state=random_state,
        )
    )

    return pd.DataFrame([result.__dict__ for result in results])


def centered_component_values(model: LinearRegression | Ridge, x: np.ndarray, degree: int) -> pd.DataFrame:
    if x.shape[1] != 2:
        raise ValueError("This helper expects exactly two features")

    coef = np.asarray(model.coef_)
    features = additive_polynomial_features(x, degree)
    comp_1 = features[:, :degree] @ coef[:degree]
    comp_2 = features[:, degree : 2 * degree] @ coef[degree : 2 * degree]

    comp_1_centered = comp_1 - np.mean(comp_1)
    comp_2_centered = comp_2 - np.mean(comp_2)
    intercept_centered = float(model.intercept_ + np.mean(comp_1) + np.mean(comp_2))

    return pd.DataFrame(
        {
            "x1": x[:, 0],
            "x2": x[:, 1],
            "component_1": comp_1_centered,
            "component_2": comp_2_centered,
            "intercept_centered": intercept_centered,
        }
    )
