"""Standalone implementation and test suite for the best-performing ML model.

Benchmarking 7 regression algorithms in train_model.py selected
HistGradientBoostingRegressor as the top model (highest R2 and lowest MAE/RMSE/MAPE).
This module implements and tests only that winning model.
"""

from pathlib import Path
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from car_catalog import CAR_CATALOG
from generate_dataset import save_dataset
from train_model import build_preprocessor, prepare_features

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "used_cars.csv"
MODEL_PATH = BASE_DIR / "model" / "car_price_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

SAMPLE_VEHICLES = [
    {
        "brand": "Hyundai",
        "model": "Creta",
        "body_type": CAR_CATALOG["Hyundai"]["Creta"]["body_type"],
        "year": 2021,
        "km_driven": 42000,
        "fuel_type": "Diesel",
        "transmission": "Automatic",
        "owner": "First Owner",
        "seller_type": "Dealer",
        "condition": "Excellent",
        "engine_cc": 1497,
        "max_power_bhp": 113.0,
        "mileage_kmpl": 18.5,
    },
    {
        "brand": "Maruti Suzuki",
        "model": "Swift",
        "body_type": CAR_CATALOG["Maruti Suzuki"]["Swift"]["body_type"],
        "year": 2019,
        "km_driven": 56000,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "owner": "First Owner",
        "seller_type": "Individual",
        "condition": "Good",
        "engine_cc": 1197,
        "max_power_bhp": 89.0,
        "mileage_kmpl": 22.4,
    },
    {
        "brand": "BMW",
        "model": "3 Series",
        "body_type": CAR_CATALOG["BMW"]["3 Series"]["body_type"],
        "year": 2020,
        "km_driven": 38000,
        "fuel_type": "Diesel",
        "transmission": "Automatic",
        "owner": "Second Owner",
        "seller_type": "Trustmark Dealer",
        "condition": "Good",
        "engine_cc": 1998,
        "max_power_bhp": 255.0,
        "mileage_kmpl": 16.1,
    },
]


def build_best_model() -> Pipeline:
    """Build the winning HistGradientBoostingRegressor pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    min_samples_leaf=10,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_evaluate_best_model(
    data_path: Path | None = None,
) -> tuple[Pipeline, dict[str, float | int | str]]:
    """Train only HistGradientBoostingRegressor and return its fitted pipeline and metrics."""
    csv_path = data_path or DATA_PATH
    if not csv_path.exists():
        save_dataset(csv_path)

    df = pd.read_csv(csv_path)
    X = prepare_features(df)
    y = df["selling_price"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_best_model()
    model.fit(X_train, np.log1p(y_train))

    preds = np.expm1(model.predict(X_test))
    metrics: dict[str, float | int | str] = {
        "algorithm": "HistGradientBoostingRegressor",
        "r2": round(float(r2_score(y_test, preds)), 4),
        "mae": int(round(float(mean_absolute_error(y_test, preds)))),
        "rmse": int(round(float(root_mean_squared_error(y_test, preds)))),
        "mape_percent": round(float(mean_absolute_percentage_error(y_test, preds) * 100), 2),
        "test_samples": int(len(X_test)),
    }
    return model, metrics


def predict_car_price(pipeline: Pipeline, vehicle: dict[str, object]) -> int:
    """Predict used-car resale value in INR using the winning model."""
    features = prepare_features(pd.DataFrame([vehicle]))
    predicted_inr = float(np.expm1(pipeline.predict(features)[0]))
    return int(round(predicted_inr, -2))


def run_best_model_report() -> None:
    """Display evaluation metrics and sample predictions for HistGradientBoostingRegressor."""
    model, metrics = train_and_evaluate_best_model()
    print("=== Best Model Implementation: HistGradientBoostingRegressor ===")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\nSample Predictions:")
    for car in SAMPLE_VEHICLES:
        price = predict_car_price(model, car)
        print(
            f"  - {car['year']} {car['brand']} {car['model']} "
            f"({car['km_driven']:,} km, {car['fuel_type']}, {car['transmission']}) "
            f"-> INR {price:,}"
        )


class BestCarPriceModelTests(unittest.TestCase):
    """Validate that only the highest-scoring algorithm is deployed and verified."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metrics = train_and_evaluate_best_model()

    def test_benchmark_selected_hist_gradient_boosting(self) -> None:
        self.assertTrue(METADATA_PATH.exists(), "Run train_model.py before testing saved artifacts.")
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        comparison = metadata["model_comparison"]
        best_entry = max(comparison, key=lambda row: (row["r2"], -row["rmse"], -row["mae"]))
        self.assertEqual(metadata["metrics"]["algorithm"], "Hist Gradient Boosting")
        self.assertEqual(best_entry["algorithm"], "Hist Gradient Boosting")

        saved_bundle = joblib.load(MODEL_PATH)
        self.assertIsInstance(
            saved_bundle["point"].named_steps["regressor"],
            HistGradientBoostingRegressor,
        )

    def test_best_model_evaluation_thresholds(self) -> None:
        self.assertGreaterEqual(float(self.metrics["r2"]), 0.985)
        self.assertLessEqual(float(self.metrics["mape_percent"]), 7.0)

    def test_sample_predictions_and_monotonic_depreciation(self) -> None:
        creta_price = predict_car_price(self.model, SAMPLE_VEHICLES[0])
        swift_price = predict_car_price(self.model, SAMPLE_VEHICLES[1])
        bmw_price = predict_car_price(self.model, SAMPLE_VEHICLES[2])

        self.assertGreater(creta_price, swift_price)
        self.assertGreater(bmw_price, creta_price)

        older_creta = {**SAMPLE_VEHICLES[0], "year": 2016, "km_driven": 95000, "owner": "Third Owner"}
        self.assertLess(predict_car_price(self.model, older_creta), creta_price)


if __name__ == "__main__":
    run_best_model_report()
    print()
    unittest.main(verbosity=2)
