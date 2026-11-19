"""Black-box and white-box tests for the HiveSpace FastAPI forecasting service.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import unittest
from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.forecast import (
    DEFAULT_HORIZON_DAYS,
    build_forecast_table,
    load_model,
)
from app.main import app
from pipeline.process_split import (
    FEATURE_COLUMNS,
    add_features,
    load_raw_data,
)


class TestAPIBlackBox(unittest.TestCase):
    """Black-box tests calling the HTTP endpoints as a user or curl client would."""

    @classmethod
    def setUpClass(cls) -> None:
        # Enter the TestClient context so FastAPI's lifespan startup runs once.
        cls._client_ctx = TestClient(app)
        cls.client = cls._client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._client_ctx.__exit__(None, None, None)

    def test_root_endpoint(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "running")
        self.assertIn("supported_date_range", body)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertGreater(body["cached_dates"], 500)

    def test_predict_one_day_ahead_valid(self) -> None:
        # 2026-07-03 is 1 day after the last data date (2026-07-02).
        response = self.client.get("/predict", params={"date": "2026-07-03"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["date"], "2026-07-03")
        self.assertEqual(body["day_of_week"], "Friday")
        self.assertEqual(body["forecast_type"], "future")
        self.assertEqual(body["horizon_days"], 1)
        self.assertIsNone(body["warning"])
        self.assertIsInstance(body["predicted_occupancy"], int)
        self.assertTrue(0 <= body["predicted_occupancy"] <= 100)

    def test_predict_far_future_includes_warning(self) -> None:
        response = self.client.get("/predict", params={"date": "2026-11-20"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["forecast_type"], "future")
        self.assertGreater(body["horizon_days"], 1)
        self.assertIsNotNone(body["warning"])
        self.assertTrue(0 <= body["predicted_occupancy"] <= 100)

    def test_predict_bad_date_format_returns_422(self) -> None:
        response = self.client.get("/predict", params={"date": "20-11-2026"})
        self.assertEqual(response.status_code, 422)

    def test_predict_missing_date_returns_422(self) -> None:
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 422)

    def test_predict_date_too_early_returns_400(self) -> None:
        response = self.client.get("/predict", params={"date": "2020-01-01"})
        self.assertEqual(response.status_code, 400)

    def test_predict_date_past_horizon_returns_400(self) -> None:
        # One day past the 180-day horizon (2026-07-02 + 181 days = 2026-12-30).
        beyond_max = date(2026, 7, 2) + timedelta(days=DEFAULT_HORIZON_DAYS + 1)
        response = self.client.get("/predict", params={"date": str(beyond_max)})
        self.assertEqual(response.status_code, 400)

    def test_recursive_forecast_preserves_weekly_pattern(self) -> None:
        # 2026-11-18 is a Wednesday, 2026-11-22 is the following Sunday.
        wed = self.client.get("/predict", params={"date": "2026-11-18"}).json()
        sun = self.client.get("/predict", params={"date": "2026-11-22"}).json()
        self.assertEqual(wed["day_of_week"], "Wednesday")
        self.assertEqual(sun["day_of_week"], "Sunday")
        self.assertGreater(wed["predicted_occupancy"], sun["predicted_occupancy"] + 20)

    def test_predictions_are_deterministic(self) -> None:
        first = self.client.get("/predict", params={"date": "2026-11-20"}).json()
        second = self.client.get("/predict", params={"date": "2026-11-20"}).json()
        self.assertEqual(first, second)


class TestForecastWhiteBox(unittest.TestCase):
    """White-box tests verifying internal forecast table logic and zero leakage."""

    def test_historical_prediction_matches_direct_model_call(self) -> None:
        history = load_raw_data()
        model = load_model()
        table = build_forecast_table(history, model, horizon_days=10)

        features = add_features(history)
        sample_row = features.iloc[[100]]
        sample_date = sample_row["date"].iloc[0].date()

        direct_pred = int(round(float(model.predict(sample_row[FEATURE_COLUMNS])[0])))
        self.assertEqual(table[sample_date]["predicted_occupancy"], direct_pred)
        self.assertEqual(table[sample_date]["forecast_type"], "historical")
        self.assertEqual(table[sample_date]["horizon_days"], 0)

    def test_forecast_table_covers_exact_horizon(self) -> None:
        history = load_raw_data()
        model = load_model()
        horizon = 15
        table = build_forecast_table(history, model, horizon_days=horizon)

        last_data_date = history["date"].max().date()
        expected_max = last_data_date + timedelta(days=horizon)
        self.assertEqual(max(table.keys()), expected_max)
        self.assertEqual(table[expected_max]["horizon_days"], horizon)


if __name__ == "__main__":
    unittest.main(verbosity=2)
