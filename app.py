"""Flask web server and ML inference API for the Second-Hand Car Price Predictor."""

from pathlib import Path
from typing import Any
import json
import os

from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd

from car_catalog import (
    CAR_CATALOG,
    CONDITION_OPTIONS,
    CURRENT_YEAR,
    OWNER_OPTIONS,
    SELLER_OPTIONS,
    get_ui_catalog,
)
from train_model import FEATURE_COLUMNS, prepare_features, train_and_save

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "car_price_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

app = Flask(__name__, static_folder="static", static_url_path="")

_model_cache: dict[str, Any] = {}


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load the trained scikit-learn pipeline and metadata, training first if missing."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        train_and_save(BASE_DIR)

    if "pipeline" not in _model_cache:
        _model_cache["pipeline"] = joblib.load(MODEL_PATH)
        _model_cache["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    return _model_cache["pipeline"], _model_cache["metadata"]


def validate_and_normalize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate incoming car specification JSON and fill catalog defaults where omitted."""
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    brand = payload.get("brand")
    model_name = payload.get("model")
    if brand not in CAR_CATALOG:
        return None, f"Unsupported brand: {brand!r}."
    if model_name not in CAR_CATALOG[brand]:
        return None, f"Model {model_name!r} is not available for brand {brand!r}."

    spec = CAR_CATALOG[brand][model_name]

    try:
        year = int(payload["year"])
        km_driven = int(payload["km_driven"])
    except (KeyError, TypeError, ValueError):
        return None, "Fields 'year' and 'km_driven' must be valid integers."

    if not (1970 <= year <= CURRENT_YEAR):
        return None, f"Manufacturing year must be between 1970 to {CURRENT_YEAR}."
    if not (0 <= km_driven <= 10000000):
        return None, "Kilometers driven must be between 0 and 10000000."

    fuel_type = payload.get("fuel_type", spec["fuels"][0])
    transmission = payload.get("transmission", spec["transmissions"][0])
    owner = payload.get("owner", OWNER_OPTIONS[0])
    seller_type = payload.get("seller_type", SELLER_OPTIONS[0])
    condition = payload.get("condition", "Good")

    validations = [
        (fuel_type in spec["fuels"], f"Invalid fuel_type {fuel_type!r} for {brand} {model_name}."),
        (transmission in spec["transmissions"], f"Invalid transmission {transmission!r} for {brand} {model_name}."),
        (owner in OWNER_OPTIONS, f"Invalid owner {owner!r}."),
        (seller_type in SELLER_OPTIONS, f"Invalid seller_type {seller_type!r}."),
        (condition in CONDITION_OPTIONS, f"Invalid condition {condition!r}."),
    ]
    for is_valid, error_msg in validations:
        if not is_valid:
            return None, error_msg

    try:
        default_cc = 0 if fuel_type == "Electric" else spec["engine_cc"]
        engine_cc = int(payload.get("engine_cc", default_cc))
        max_power_bhp = float(payload.get("max_power_bhp", spec["max_power_bhp"]))
        mileage_kmpl = float(payload.get("mileage_kmpl", spec["mileage_kmpl"]))
    except (TypeError, ValueError):
        return None, "Numeric specs (engine_cc, max_power_bhp, mileage_kmpl) must be valid numbers."

    if engine_cc < 0 or max_power_bhp <= 0 or mileage_kmpl <= 0:
        return None, "Engine CC must be >= 0, and power/mileage must be positive."

    normalized = {
        "brand": brand,
        "model": model_name,
        "body_type": spec["body_type"],
        "year": year,
        "km_driven": km_driven,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "owner": owner,
        "seller_type": seller_type,
        "condition": condition,
        "engine_cc": engine_cc,
        "max_power_bhp": max_power_bhp,
        "mileage_kmpl": mileage_kmpl,
    }
    return normalized, None


def predict_with_uncertainty(model_bundle: dict[str, Any], record: dict[str, Any]) -> tuple[int, int, int]:
    """Return predicted price and 15th-85th percentile quantile interval."""
    input_df = prepare_features(pd.DataFrame([record]))
    point_price = float(np.expm1(model_bundle["point"].predict(input_df)[0]))
    low_price = float(np.expm1(model_bundle["low"].predict(input_df)[0]))
    high_price = float(np.expm1(model_bundle["high"].predict(input_df)[0]))

    return (
        int(round(point_price, -2)),
        int(round(min(low_price, point_price * 0.96), -2)),
        int(round(max(high_price, point_price * 1.04), -2)),
    )


def compute_depreciation_forecast(pipeline: Any, record: dict[str, Any], years_ahead: int = 5) -> list[dict[str, int]]:
    """Project resale value over the next N years assuming 11,500 km driven annually."""
    forecast_rows = []
    for offset in range(years_ahead + 1):
        row = dict(record)
        # Shift manufacturing year backward relative to CURRENT_YEAR to simulate aging.
        row["year"] = record["year"] - offset
        row["km_driven"] = min(400000, record["km_driven"] + offset * 11500)
        forecast_rows.append(row)

    df = prepare_features(pd.DataFrame(forecast_rows))
    predicted_prices = np.expm1(pipeline.predict(df))

    return [
        {
            "year": CURRENT_YEAR + offset,
            "age_years": (CURRENT_YEAR - record["year"]) + offset,
            "projected_km": int(forecast_rows[offset]["km_driven"]),
            "predicted_price": int(round(float(price), -2)),
        }
        for offset, price in enumerate(predicted_prices)
    ]


def compute_feature_impacts(pipeline: Any, record: dict[str, Any], predicted_price: int) -> list[dict[str, Any]]:
    """Measure marginal INR contribution of key attributes against a same-model neutral baseline."""
    spec = CAR_CATALOG[record["brand"]][record["model"]]
    reference_year = max(2012, CURRENT_YEAR - 5)

    comparisons = [
        ("car_age", f"Model Year ({record['year']})", {"year": reference_year}),
        ("km_driven", f"Odometer ({record['km_driven']:,} km)", {"km_driven": 55000}),
        ("fuel_type", f"Fuel ({record['fuel_type']})", {"fuel_type": spec["fuels"][0]}),
        ("transmission", f"Gearbox ({record['transmission']})", {"transmission": spec["transmissions"][0]}),
        ("owner", f"History ({record['owner']})", {"owner": "First Owner"}),
        ("condition", f"Condition ({record['condition']})", {"condition": "Good"}),
        ("seller_type", f"Seller ({record['seller_type']})", {"seller_type": "Individual"}),
    ]

    counterfactual_rows = []
    active_comparisons = []
    for key, label, neutral_override in comparisons:
        row = dict(record)
        changed = False
        for field, neutral_val in neutral_override.items():
            if row[field] != neutral_val:
                row[field] = neutral_val
                changed = True
        if changed:
            counterfactual_rows.append(row)
            active_comparisons.append((key, label))

    if not counterfactual_rows:
        return []

    df = prepare_features(pd.DataFrame(counterfactual_rows))
    baseline_prices = np.expm1(pipeline.predict(df))

    impacts = []
    for (key, label), alt_price in zip(active_comparisons, baseline_prices):
        delta = int(round(predicted_price - float(alt_price), -2))
        if abs(delta) >= 1500:
            impacts.append({"feature": key, "label": label, "impact_inr": delta})

    impacts.sort(key=lambda item: abs(item["impact_inr"]), reverse=True)
    return impacts


@app.after_request
def add_cors_headers(response: Any) -> Any:
    """Allow local file:// or dev-server origins to call the Flask ML API."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def serve_index():
    """Serve the single-page web application."""
    return app.send_static_file("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Return service health status."""
    return jsonify({"status": "ok"})


@app.route("/api/catalog", methods=["GET"])
def get_catalog():
    """Return dropdown options, model specifications, and trained model metrics."""
    _, metadata = load_artifacts()
    payload = get_ui_catalog()
    payload["model_metadata"] = metadata
    return jsonify(payload)


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict_price():
    """Predict used-car resale value, confidence interval, impacts, and 5-year curve."""
    if request.method == "OPTIONS":
        return ("", 204)
    model_bundle, _ = load_artifacts()
    data = request.get_json(silent=True)

    normalized, error = validate_and_normalize_payload(data)
    if error or normalized is None:
        return jsonify({"error": error}), 400

    spec = CAR_CATALOG[normalized["brand"]][normalized["model"]]
    point_pipeline = model_bundle["point"]
    predicted_price, low_price, high_price = predict_with_uncertainty(model_bundle, normalized)
    forecast = compute_depreciation_forecast(point_pipeline, normalized, years_ahead=5)
    impacts = compute_feature_impacts(point_pipeline, normalized, predicted_price)

    retained_pct = round((predicted_price / spec["base_price"]) * 100, 1)

    return jsonify(
        {
            "predicted_price": predicted_price,
            "price_range": {"low": low_price, "high": high_price},
            "new_car_reference_price": spec["base_price"],
            "retained_value_percent": retained_pct,
            "vehicle": normalized,
            "feature_impacts": impacts,
            "depreciation_forecast": forecast,
        }
    )


# Preload trained model artifacts when present (enables Gunicorn --preload CoW sharing).
if MODEL_PATH.exists() and METADATA_PATH.exists():
    load_artifacts()

if __name__ == "__main__":
    load_artifacts()
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
