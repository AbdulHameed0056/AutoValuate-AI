"""Train and benchmark multiple ML regression algorithms and persist the best model."""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from car_catalog import CURRENT_YEAR
from generate_dataset import save_dataset

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "body_type",
    "fuel_type",
    "transmission",
    "owner",
    "seller_type",
    "condition",
]

NUMERIC_FEATURES = [
    "car_age",
    "km_driven",
    "engine_cc",
    "max_power_bhp",
    "mileage_kmpl",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

FEATURE_LABELS = {
    "brand": "Brand",
    "model": "Model",
    "body_type": "Body Type",
    "fuel_type": "Fuel Type",
    "transmission": "Transmission",
    "owner": "Owner History",
    "seller_type": "Seller Type",
    "condition": "Condition",
    "car_age": "Vehicle Age",
    "km_driven": "Kilometers Driven",
    "engine_cc": "Engine Capacity (CC)",
    "max_power_bhp": "Max Power (BHP)",
    "mileage_kmpl": "Fuel Economy",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features from raw car listing columns."""
    engineered = df.copy()
    engineered["car_age"] = (CURRENT_YEAR - engineered["year"]).clip(lower=0)
    return engineered[FEATURE_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    """Create a fresh preprocessing transformer for categorical and numeric columns."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def get_candidate_models() -> dict[str, object]:
    """Return diverse regression estimators to benchmark on the used-car dataset."""
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=18, min_samples_leaf=3, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=24,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=280,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            random_state=42,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            random_state=42,
        ),
    }


def build_quantile_pipeline(alpha: float) -> Pipeline:
    """Create a quantile GradientBoostingRegressor pipeline for uncertainty bounds."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=alpha,
                    n_estimators=160,
                    learning_rate=0.07,
                    max_depth=4,
                    subsample=0.85,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(
    name: str,
    estimator: object,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, object]]:
    """Fit a candidate pipeline on log-price and compute evaluation metrics on INR scale."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", estimator),
        ]
    )

    started = time.perf_counter()
    pipeline.fit(X_train, y_train_log)
    elapsed = round(time.perf_counter() - started, 3)

    y_pred = np.expm1(pipeline.predict(X_test))

    metrics = {
        "algorithm": name,
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": int(round(float(mean_absolute_error(y_test, y_pred)))),
        "rmse": int(round(float(root_mean_squared_error(y_test, y_pred)))),
        "mape_percent": round(float(mean_absolute_percentage_error(y_test, y_pred) * 100), 2),
        "train_time_sec": elapsed,
    }
    return pipeline, metrics


def compute_permutation_importances(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test_log: np.ndarray
) -> list[dict[str, float | str]]:
    """Compute model-agnostic normalized permutation importance across input features."""
    result = permutation_importance(
        pipeline,
        X_test,
        y_test_log,
        n_repeats=6,
        random_state=42,
        n_jobs=-1,
    )
    raw_scores = [max(0.0, float(val)) for val in result.importances_mean]
    total = sum(raw_scores) or 1.0

    ranked = sorted(
        zip(FEATURE_COLUMNS, raw_scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "feature": col,
            "label": FEATURE_LABELS[col],
            "importance": round((score / total) * 100, 2),
        }
        for col, score in ranked
    ]


def train_and_save(base_dir: Path | None = None) -> dict[str, object]:
    """Benchmark candidate algorithms, pick the best one, and save artifacts."""
    root = base_dir or Path(__file__).parent
    data_path = root / "data" / "used_cars.csv"
    if not data_path.exists():
        save_dataset(data_path)

    raw_df = pd.read_csv(data_path)
    X = prepare_features(raw_df)
    y = raw_df["selling_price"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    evaluated: list[tuple[Pipeline, dict[str, object]]] = []
    for name, estimator in get_candidate_models().items():
        evaluated.append(
            evaluate_model(name, estimator, X_train, y_train_log, X_test, y_test)
        )

    # Select winner by highest R2, then lowest RMSE, then lowest MAE.
    evaluated.sort(
        key=lambda item: (-item[1]["r2"], item[1]["rmse"], item[1]["mae"])
    )

    best_pipeline, best_raw_metrics = evaluated[0]

    # Fit quantile estimators to provide adaptive low/high market confidence bounds.
    low_pipeline = build_quantile_pipeline(0.15).fit(X_train, y_train_log)
    high_pipeline = build_quantile_pipeline(0.85).fit(X_train, y_train_log)

    comparison = [
        {**entry_metrics, "is_best": idx == 0}
        for idx, (_, entry_metrics) in enumerate(evaluated)
    ]

    best_metrics = {
        **best_raw_metrics,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }

    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"point": best_pipeline, "low": low_pipeline, "high": high_pipeline},
        model_dir / "car_price_model.joblib",
    )

    metadata = {
        "metrics": best_metrics,
        "model_comparison": comparison,
        "feature_importances": compute_permutation_importances(
            best_pipeline, X_test, y_test_log
        ),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Model Evaluation Comparison ===")
    print(f"{'Algorithm':<24} {'R2':>8} {'MAE (INR)':>12} {'RMSE (INR)':>12} {'MAPE %':>8} {'Time(s)':>8}")
    print("-" * 76)
    for row in comparison:
        tag = " *BEST" if row["is_best"] else ""
        print(
            f"{row['algorithm']:<24} {row['r2']:>8.4f} {row['mae']:>12,} "
            f"{row['rmse']:>12,} {row['mape_percent']:>7.2f}% {row['train_time_sec']:>7.3f}{tag}"
        )

    return metadata


if __name__ == "__main__":
    train_and_save()
