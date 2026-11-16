"""Rolling-origin backtesting and honest model evaluation for HiveSpace desk occupancy.

This script simulates live production operation:
1. Loops through the last 90 days of the test set.
2. For each day t, retrains the XGBoost model on all historical data strictly prior to t.
3. Generates 1-day-ahead predictions for both XGBoost and the 7-day Moving Average baseline.
4. Computes in-sample train error across expanding windows to evaluate train vs. test overfitting.
5. Calculates MAE and RMSE over the 90-day backtest period.
6. Generates a line chart (data/backtest_predictions.png) showing predicted vs. actual occupancy.
7. Saves predictions to data/backtest_predictions.csv and updates metrics.json.

Run from project root:
    python pipeline/backtest.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

# Ensure pipeline directory is in sys.path
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from process_split import (
    DATA_DIR,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_and_split,
)

# Output paths
METRICS_PATH = PROJECT_ROOT / "metrics.json"
PREDICTIONS_CSV_PATH = DATA_DIR / "backtest_predictions.csv"
CHART_PNG_PATH = DATA_DIR / "backtest_predictions.png"

BACKTEST_DAYS = 90
RANDOM_SEED = 42


def get_full_chronological_data() -> pd.DataFrame:
    """Load and return the full dataset with engineered features sorted chronologically."""
    train_data, val_data, test_data = load_and_split(save_to_disk=True)
    full_data = (
        pd.concat([train_data, val_data, test_data], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    return full_data


def train_xgboost(train_data: pd.DataFrame) -> XGBRegressor:
    """Fit an XGBoost regressor on the provided training slice."""
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])
    return model


def run_rolling_origin_backtest(
    full_data: pd.DataFrame,
    backtest_days: int = BACKTEST_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """Run rolling-origin evaluation over the last `backtest_days` rows.

    For each step:
      - Prior data (< t) is used for training.
      - Next day (t) is predicted (1-step-ahead forecast).
      - In-sample train errors are tracked to quantify the overfitting gap.
    """
    total_rows = len(full_data)
    if total_rows <= backtest_days:
        raise ValueError(
            f"Dataset has {total_rows} rows, which is not enough for a {backtest_days}-day backtest."
        )

    start_idx = total_rows - backtest_days
    results = []
    train_maes = []
    train_rmses = []

    print(f"Starting rolling-origin backtest for the last {backtest_days} days...")
    print(f"Backtest period: {full_data['date'].iloc[start_idx].date()} to {full_data['date'].iloc[-1].date()}")

    for step, idx in enumerate(range(start_idx, total_rows), start=1):
        prior_data = full_data.iloc[:idx]
        target_row = full_data.iloc[[idx]]

        target_date = target_row["date"].iloc[0]
        actual_val = float(target_row[TARGET_COLUMN].iloc[0])

        # Anti-leakage assertion: training dates must strictly precede the target date
        assert prior_data["date"].max() < target_date, "Leakage detected: prior data contains target date!"

        # Retrain XGBoost on expanding historical window
        model = train_xgboost(prior_data)

        # 1-day ahead out-of-sample prediction
        xgb_pred = float(model.predict(target_row[FEATURE_COLUMNS])[0])

        # Baseline: trailing 7-day moving average (already built with shift(1))
        baseline_pred = float(target_row["rolling_mean_7"].iloc[0])

        # In-sample train performance check
        train_preds = model.predict(prior_data[FEATURE_COLUMNS])
        train_mae = mean_absolute_error(prior_data[TARGET_COLUMN], train_preds)
        train_rmse = root_mean_squared_error(prior_data[TARGET_COLUMN], train_preds)

        train_maes.append(train_mae)
        train_rmses.append(train_rmse)

        results.append(
            {
                "date": target_date,
                "actual_occupancy": actual_val,
                "baseline_prediction": baseline_pred,
                "xgboost_prediction": xgb_pred,
            }
        )

        if step % 30 == 0 or step == backtest_days:
            print(f"  Processed {step}/{backtest_days} days (latest date: {target_date.date()})")

    results_df = pd.DataFrame(results)

    # Compute overall evaluation metrics over the 90 days
    actuals = results_df["actual_occupancy"]
    xgb_preds = results_df["xgboost_prediction"]
    base_preds = results_df["baseline_prediction"]

    backtest_metrics = {
        "period": {
            "start_date": str(results_df["date"].min().date()),
            "end_date": str(results_df["date"].max().date()),
            "days_evaluated": int(len(results_df)),
        },
        "baseline_moving_average_7d": {
            "mae": round(float(mean_absolute_error(actuals, base_preds)), 4),
            "rmse": round(float(root_mean_squared_error(actuals, base_preds)), 4),
        },
        "xgboost_rolling_origin": {
            "test_mae": round(float(mean_absolute_error(actuals, xgb_preds)), 4),
            "test_rmse": round(float(root_mean_squared_error(actuals, xgb_preds)), 4),
            "avg_train_mae": round(float(np.mean(train_maes)), 4),
            "avg_train_rmse": round(float(np.mean(train_rmses)), 4),
            "mae_generalization_gap": round(
                float(mean_absolute_error(actuals, xgb_preds) - np.mean(train_maes)), 4
            ),
            "rmse_generalization_gap": round(
                float(root_mean_squared_error(actuals, xgb_preds) - np.mean(train_rmses)), 4
            ),
        },
    }

    return results_df, backtest_metrics


def save_chart(results_df: pd.DataFrame, output_path: Path = CHART_PNG_PATH) -> Path:
    """Generate and save publication-quality matplotlib chart comparing predicted vs actual."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.figure(figsize=(14, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    dates = pd.to_datetime(results_df["date"])

    # Plot actual occupancy curve
    plt.plot(
        dates,
        results_df["actual_occupancy"],
        label="Actual Occupancy",
        color="#1f2937",
        linewidth=2.2,
        alpha=0.9,
    )

    # Plot XGBoost rolling-origin predictions
    plt.plot(
        dates,
        results_df["xgboost_prediction"],
        label="XGBoost (Rolling-Origin)",
        color="#2563eb",
        linewidth=2.0,
        linestyle="--",
        alpha=0.95,
    )

    # Plot Moving Average baseline predictions
    plt.plot(
        dates,
        results_df["baseline_prediction"],
        label="7-Day Moving Average Baseline",
        color="#9ca3af",
        linewidth=1.8,
        linestyle=":",
        alpha=0.85,
    )

    plt.title(
        "HiveSpace Desk Occupancy — 90-Day Rolling-Origin Backtest\nPredicted vs. Actual Occupancy",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Date", fontsize=11, labelpad=10)
    plt.ylabel("Desk Occupancy (Number of Desks)", fontsize=11, labelpad=10)
    plt.ylim(-5, 110)

    # Format x-axis with weekly tick marks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(range(0, 101, 20), fontsize=9)

    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Chart saved to: {output_path}")
    return output_path


def update_metrics_json(backtest_metrics: dict, metrics_path: Path = METRICS_PATH) -> None:
    """Update metrics.json to include the 90-day backtesting evaluation."""
    metrics_data = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            try:
                metrics_data = json.load(f)
            except json.JSONDecodeError:
                metrics_data = {}

    metrics_data["backtest_90d"] = backtest_metrics

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Updated metrics in: {metrics_path}")


def print_backtest_report(metrics: dict) -> None:
    """Print a clean summary table of backtest MAE, RMSE, and overfitting analysis."""
    period = metrics["period"]
    base = metrics["baseline_moving_average_7d"]
    xgb = metrics["xgboost_rolling_origin"]

    print("\n" + "=" * 70)
    print(f"BACKTESTING RESULTS ({period['days_evaluated']} Days: {period['start_date']} to {period['end_date']})")
    print("=" * 70)
    print(f"{'Model':<35}{'MAE (Desks)':>15}{'RMSE (Desks)':>18}")
    print("-" * 70)
    print(f"{'7-Day Moving Average Baseline':<35}{base['mae']:>15.3f}{base['rmse']:>18.3f}")
    print(f"{'XGBoost (Rolling-Origin Retrained)':<35}{xgb['test_mae']:>15.3f}{xgb['test_rmse']:>18.3f}")
    print("-" * 70)

    # Overfitting analysis output
    print("\nOVERFITTING ANALYSIS (XGBoost Train vs. Test Generalization Gap):")
    print(f"  - Average In-Sample Train MAE : {xgb['avg_train_mae']:.3f} desks")
    print(f"  - Out-of-Sample Backtest MAE : {xgb['test_mae']:.3f} desks")
    print(f"  - Generalization Gap (MAE)   : {xgb['mae_generalization_gap']:.3f} desks")
    print(f"  - Average In-Sample Train RMSE: {xgb['avg_train_rmse']:.3f} desks")
    print(f"  - Out-of-Sample Backtest RMSE: {xgb['test_rmse']:.3f} desks")
    print(f"  - Generalization Gap (RMSE)  : {xgb['rmse_generalization_gap']:.3f} desks")

    if xgb["mae_generalization_gap"] <= 3.0:
        print("  - Diagnosis: NO HARMFUL OVERFITTING DETECTED.")
        print("    The gap between training error and backtest error is small and consistent")
        print("    with irreducible random noise (sigma = 6.0). Generalization is strong.")
    else:
        print("  - Diagnosis: Potential overfitting detected (gap > 3.0 desks).")
    print("=" * 70 + "\n")


def main() -> None:
    full_data = get_full_chronological_data()
    results_df, backtest_metrics = run_rolling_origin_backtest(full_data, BACKTEST_DAYS)

    # Save predictions CSV
    PREDICTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(PREDICTIONS_CSV_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Saved 90-day backtest predictions to: {PREDICTIONS_CSV_PATH}")

    # Generate and save chart
    save_chart(results_df, CHART_PNG_PATH)

    # Update metrics.json
    update_metrics_json(backtest_metrics, METRICS_PATH)

    # Print summary report
    print_backtest_report(backtest_metrics)


if __name__ == "__main__":
    main()
