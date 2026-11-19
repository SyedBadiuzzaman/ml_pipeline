"""Forecasting logic for the HiveSpace desk occupancy API.

This module is kept separate from FastAPI routes so it is easy to read and test.
At API startup, `build_forecast_table()` precomputes predictions for:
1. All historical dates with full lag history (one-step-ahead predictions).
2. Up to `DEFAULT_HORIZON_DAYS` (180) days after the last known data date,
   using a recursive forecast: predict day t, clip to [0, 100], feed that
   prediction back in as "yesterday", and repeat for day t + 1.

Feature engineering is imported directly from `pipeline.process_split` so
training and serving always compute identical features.
"""

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pipeline.process_split import (
    FEATURE_COLUMNS,
    LAG_DAYS,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    add_features,
    load_raw_data,
)

MODEL_DIR = PROJECT_ROOT / "model"
MODEL_JSON_PATH = MODEL_DIR / "xgb_occupancy.json"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

DEFAULT_HORIZON_DAYS = 180
TOTAL_DESKS = 100
MAX_LAG = max(LAG_DAYS)


def load_model(model_path: Path = MODEL_JSON_PATH) -> XGBRegressor:
    """Load the saved XGBoost model from its JSON file."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. "
            "Run `python pipeline/export_model.py` first."
        )
    model = XGBRegressor()
    model.load_model(str(model_path))
    return model


def load_model_info(info_path: Path = MODEL_INFO_PATH) -> dict:
    """Load model metadata saved alongside the XGBoost JSON file."""
    if not info_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {info_path}. "
            "Run `python pipeline/export_model.py` first."
        )
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_forecast_table(
    history_df: pd.DataFrame,
    model: XGBRegressor,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    total_desks: int = TOTAL_DESKS,
) -> dict[date, dict]:
    """Precompute predictions for all historical feature dates plus `horizon_days` ahead.

    Returns a dictionary keyed by `datetime.date` so API lookups are O(1) and
    deterministic.
    """
    if horizon_days < 0:
        raise ValueError("horizon_days must be 0 or greater.")

    # 1) Historical one-step-ahead predictions on all days with complete lag history.
    hist_features = add_features(history_df)
    hist_preds = np.clip(model.predict(hist_features[FEATURE_COLUMNS]), 0, total_desks)

    forecast_table: dict[date, dict] = {}
    for row_date, pred in zip(hist_features["date"], hist_preds):
        day = row_date.date()
        forecast_table[day] = {
            "date": day,
            "day_of_week": row_date.day_name(),
            "predicted_occupancy": int(round(float(pred))),
            "total_desks": total_desks,
            "forecast_type": "historical",
            "horizon_days": 0,
            "warning": None,
        }

    # 2) Recursive future forecasts: each day's prediction becomes input for the next.
    #    We only need the trailing `MAX_LAG` days of history to compute features for day t.
    working_series = (
        history_df[["date", TARGET_COLUMN]]
        .sort_values("date")
        .tail(MAX_LAG)
        .reset_index(drop=True)
        .copy()
    )
    working_series[TARGET_COLUMN] = working_series[TARGET_COLUMN].astype(float)

    for step in range(1, horizon_days + 1):
        next_timestamp = working_series["date"].iloc[-1] + pd.Timedelta(days=1)

        # Append day t with NaN target so add_features() can only use past days (< t).
        next_row = pd.DataFrame({"date": [next_timestamp], TARGET_COLUMN: [np.nan]})
        window = pd.concat([working_series.tail(MAX_LAG), next_row], ignore_index=True)

        features_row = add_features(window)
        raw_prediction = float(model.predict(features_row[FEATURE_COLUMNS])[0])
        clipped_prediction = float(np.clip(raw_prediction, 0, total_desks))

        # Feed the predicted occupancy back into the rolling history for step + 1.
        filled_row = pd.DataFrame(
            {"date": [next_timestamp], TARGET_COLUMN: [clipped_prediction]}
        )
        working_series = pd.concat(
            [working_series.tail(MAX_LAG), filled_row], ignore_index=True
        )

        warning = None
        if step > 1:
            warning = (
                f"This date is {step} days after the last known data. "
                "Only 1-day-ahead accuracy was tested (MAE about 6 desks); "
                "error is likely larger this far ahead."
            )

        day = next_timestamp.date()
        forecast_table[day] = {
            "date": day,
            "day_of_week": next_timestamp.day_name(),
            "predicted_occupancy": int(round(clipped_prediction)),
            "total_desks": total_desks,
            "forecast_type": "future",
            "horizon_days": step,
            "warning": warning,
        }

    return forecast_table


def initialize_forecast_state(
    csv_path: Path = RAW_DATA_PATH,
    model_path: Path = MODEL_JSON_PATH,
    info_path: Path = MODEL_INFO_PATH,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dict:
    """Load the model, metadata, and raw history, and build the lookup table."""
    model = load_model(model_path)
    model_info = load_model_info(info_path)
    history_df = load_raw_data(csv_path)
    total_desks = int(model_info.get("total_desks", TOTAL_DESKS))

    forecast_table = build_forecast_table(
        history_df=history_df,
        model=model,
        horizon_days=horizon_days,
        total_desks=total_desks,
    )
    allowed_dates = sorted(forecast_table.keys())

    return {
        "model": model,
        "model_info": model_info,
        "forecast_table": forecast_table,
        "min_date": allowed_dates[0],
        "max_date": allowed_dates[-1],
        "last_data_date": history_df["date"].max().date(),
    }
