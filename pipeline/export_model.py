"""Train the final XGBoost model on all available history and export it for serving.

Why train on all rows here?
- In `train_evaluate.py` and `backtest.py`, we strictly held out validation and
  test sets to measure honest out-of-sample error.
- Once the model design is verified, retraining on 100% of available historical
  rows (534 days) gives the production API the most recent office patterns.

The model is saved using XGBoost's native JSON format (`model/xgb_occupancy.json`)
rather than `pickle`, avoiding unsafe deserialization and keeping it text-based
for Hugging Face Spaces.

Run from the project root:
    python pipeline/export_model.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from xgboost import XGBRegressor

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from process_split import (
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    add_features,
    load_raw_data,
)

MODEL_DIR = PROJECT_ROOT / "model"
MODEL_JSON_PATH = MODEL_DIR / "xgb_occupancy.json"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
METRICS_PATH = PROJECT_ROOT / "metrics.json"

RANDOM_SEED = 42
TOTAL_DESKS = 100


def train_production_model(csv_path: Path = RAW_DATA_PATH) -> tuple[XGBRegressor, dict]:
    """Load all raw data, build lag features, and fit the production XGBoost model."""
    raw_data = load_raw_data(csv_path)
    feature_data = add_features(raw_data)

    # Use the exact same hyperparameters validated in train_evaluate.py and backtest.py.
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    model.fit(feature_data[FEATURE_COLUMNS], feature_data[TARGET_COLUMN])

    trained_at = datetime.now(timezone.utc)
    backtest_summary = _read_backtest_summary(METRICS_PATH)

    model_info = {
        "model_version": f"xgb-{trained_at.strftime('%Y-%m-%d')}",
        "trained_at_utc": trained_at.isoformat(),
        "model_type": "XGBRegressor",
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": RANDOM_SEED,
        },
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "total_desks": TOTAL_DESKS,
        "training_rows": int(len(feature_data)),
        "feature_start_date": str(feature_data["date"].min().date()),
        "last_data_date": str(feature_data["date"].max().date()),
        "backtest_90d_metrics": backtest_summary,
    }
    return model, model_info


def _read_backtest_summary(metrics_path: Path) -> dict:
    """Pull 90-day backtest MAE/RMSE from metrics.json if available."""
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        bt = metrics.get("backtest_90d", {})
        xgb = bt.get("xgboost_rolling_origin", {})
        return {
            "test_mae": xgb.get("test_mae"),
            "test_rmse": xgb.get("test_rmse"),
            "days_evaluated": bt.get("period", {}).get("days_evaluated"),
        }
    except (json.JSONDecodeError, OSError):
        return {}


def export_model(
    csv_path: Path = RAW_DATA_PATH,
    model_path: Path = MODEL_JSON_PATH,
    info_path: Path = MODEL_INFO_PATH,
) -> tuple[Path, Path]:
    """Train and save the model JSON and metadata JSON."""
    model, model_info = train_production_model(csv_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Save using XGBoost's native JSON format (safe text file, no pickle).
    model.save_model(str(model_path))

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    return model_path, info_path


def main() -> None:
    try:
        model_path, info_path = export_model()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    print(f"Exported production model to: {model_path}")
    print(f"Exported model metadata to : {info_path}")
    print(
        f"Trained on {info['training_rows']} rows "
        f"({info['feature_start_date']} -> {info['last_data_date']})"
    )


if __name__ == "__main__":
    main()
