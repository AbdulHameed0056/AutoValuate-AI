"""Automated tests for the Car Price Prediction ML pipeline and Flask API."""

import unittest

from app import app, load_artifacts
from car_catalog import CAR_CATALOG


class CarPricePredictorTests(unittest.TestCase):
    """Verify ML model artifacts, endpoints, validation, and forecast outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        load_artifacts()
        cls.client = app.test_client()

    def test_health_and_homepage(self) -> None:
        with self.client.get("/api/health") as health:
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.get_json(), {"status": "ok"})

        with self.client.get("/") as home:
            self.assertEqual(home.status_code, 200)
            self.assertIn(b"AutoValuate", home.data)

    def test_catalog_endpoint(self) -> None:
        res = self.client.get("/api/catalog")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(set(payload["brands"].keys()), set(CAR_CATALOG.keys()))
        self.assertGreater(payload["model_metadata"]["metrics"]["r2"], 0.90)

    def test_predict_valid_vehicle(self) -> None:
        res = self.client.post(
            "/api/predict",
            json={
                "brand": "Hyundai",
                "model": "Creta",
                "year": 2021,
                "km_driven": 42000,
                "fuel_type": "Diesel",
                "transmission": "Automatic",
                "owner": "First Owner",
                "seller_type": "Dealer",
                "condition": "Excellent",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        price = data["predicted_price"]
        self.assertGreater(price, 700000)
        self.assertLess(price, 1800000)
        self.assertLessEqual(data["price_range"]["low"], price)
        self.assertGreaterEqual(data["price_range"]["high"], price)
        self.assertEqual(len(data["depreciation_forecast"]), 6)

    def test_newer_and_better_condition_increases_value(self) -> None:
        base_payload = {
            "brand": "Maruti Suzuki",
            "model": "Swift",
            "year": 2018,
            "km_driven": 60000,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "owner": "Second Owner",
            "seller_type": "Individual",
            "condition": "Fair",
        }
        upgraded_payload = {
            **base_payload,
            "year": 2023,
            "km_driven": 20000,
            "owner": "First Owner",
            "condition": "Excellent",
        }
        low_res = self.client.post("/api/predict", json=base_payload).get_json()
        high_res = self.client.post("/api/predict", json=upgraded_payload).get_json()
        self.assertGreater(high_res["predicted_price"], low_res["predicted_price"])

    def test_predict_rejects_invalid_input(self) -> None:
        bad_brand = self.client.post(
            "/api/predict",
            json={"brand": "UnknownBrand", "model": "Swift", "year": 2020, "km_driven": 30000},
        )
        self.assertEqual(bad_brand.status_code, 400)

        bad_year = self.client.post(
            "/api/predict",
            json={"brand": "Maruti Suzuki", "model": "Swift", "year": 1950, "km_driven": 30000},
        )
        self.assertEqual(bad_year.status_code, 400)


if __name__ == "__main__":
    unittest.main()
