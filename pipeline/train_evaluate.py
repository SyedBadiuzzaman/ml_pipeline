"""Train a Moving Average baseline and an XGBoost model, then compare MAE.

Steps:
1. Get the chronological train / validation / test splits from process_split.py.
2. Baseline: predict each day as the average of the previous 7 days.
3. XGBoost: learn from the lag features on the train split only.
4. Measure Mean Absolute Error (MAE) on validation and test splits.
5. Print the comparison and save it to metrics.json.

MAE = average of |actual - predicted|, measured in desks.
Example: MAE of 5 means predictions are off by about 5 desks per day.

Run from the project root:
    python pipeline/train_evaluate.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from process_split import (
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VALIDATION_CSV_PATH,
    load_and_split,
)

METRICS_PATH = PROJECT_ROOT / "metrics.json"
RANDOM_SEED = 42


def moving_average_predictions(split_data: pd.DataFrame) -> pd.Series:
    """Baseline: the trailing 7-day mean (days t-7 ... t-1) is the forecast.

    This column was already built with shift(1), so it never sees day t.
    """
    return split_data["rolling_mean_7"]


def train_xgboost(train_data: pd.DataFrame) -> XGBRegressor:
    """Fit an XGBoost regressor on the training split only."""
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])
    return model


def split_summary(split_data: pd.DataFrame) -> dict[str, str | int]:
    """Small summary of a split for metrics.json (proves the date order)."""
    return {
        "rows": int(len(split_data)),
        "start_date": str(split_data["date"].min().date()),
        "end_date": str(split_data["date"].max().date()),
    }


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/validation/test splits from data/ CSVs if available, or generate them."""
    if TRAIN_CSV_PATH.exists() and VALIDATION_CSV_PATH.exists() and TEST_CSV_PATH.exists():
        train_data = pd.read_csv(TRAIN_CSV_PATH)
        train_data["date"] = pd.to_datetime(train_data["date"])
        val_data = pd.read_csv(VALIDATION_CSV_PATH)
        val_data["date"] = pd.to_datetime(val_data["date"])
        test_data = pd.read_csv(TEST_CSV_PATH)
        test_data["date"] = pd.to_datetime(test_data["date"])
        return train_data, val_data, test_data
    return load_and_split(save_to_disk=True)


def evaluate_models() -> dict:
    """Train both models and return all metrics as a dictionary."""
    train_data, validation_data, test_data = load_splits()
    model = train_xgboost(train_data)

    metrics: dict = {"moving_average_7d": {}, "xgboost": {}}
    for split_name, split_data in [("validation", validation_data), ("test", test_data)]:
        actual = split_data[TARGET_COLUMN]
        baseline_mae = mean_absolute_error(actual, moving_average_predictions(split_data))
        xgboost_mae = mean_absolute_error(actual, model.predict(split_data[FEATURE_COLUMNS]))
        metrics["moving_average_7d"][f"{split_name}_mae"] = round(float(baseline_mae), 4)
        metrics["xgboost"][f"{split_name}_mae"] = round(float(xgboost_mae), 4)

    metrics["splits"] = {
        "train": split_summary(train_data),
        "validation": split_summary(validation_data),
        "test": split_summary(test_data),
    }
    metrics["features"] = FEATURE_COLUMNS
    return metrics


def save_metrics(metrics: dict, output_path: Path = METRICS_PATH) -> None:
    """Write metrics as readable JSON."""
    with output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def print_report(metrics: dict) -> None:
    """Print a small comparison table to the console."""
    print("Mean Absolute Error (desks, lower is better)")
    print(f"{'Model':<22}{'Validation':>12}{'Test':>10}")
    for label, key in [("Moving Average (7d)", "moving_average_7d"), ("XGBoost", "xgboost")]:
        scores = metrics[key]
        print(f"{label:<22}{scores['validation_mae']:>12.3f}{scores['test_mae']:>10.3f}")


def main() -> None:
    try:
        metrics = evaluate_models()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print_report(metrics)
    save_metrics(metrics)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
