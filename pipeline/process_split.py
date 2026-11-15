"""Load raw occupancy data, build lag features, and split it by time.

Two rules matter most here:
1. Features for day t may only use days BEFORE t (t-1, t-2, ...).
   That is why every feature starts with `.shift(...)`.
2. The split is chronological: train = oldest days, validation = middle,
   test = newest days. Rows are never shuffled.

Run from the project root to see the split report:
    python pipeline/process_split.py

`train_evaluate.py` imports `load_and_split()` from this file, so the
splits stay in memory and no extra CSV files are written.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_occupancy.csv"

TARGET_COLUMN = "occupancy"
LAG_DAYS = [1, 7, 14]
ROLLING_WINDOW = 7

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_std_7",
    "day_of_week",
    "is_weekend",
]

TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15
# The test set gets the remaining (newest) 15%.


def load_raw_data(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, check it, and return it sorted by date.

    Raises:
        FileNotFoundError: If the CSV does not exist yet.
        ValueError: If columns are missing or the values are not usable.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {csv_path}. "
            "Run `python pipeline/generate_data.py` first."
        )

    raw_data = pd.read_csv(csv_path)

    missing_columns = {"date", TARGET_COLUMN} - set(raw_data.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    # errors="raise" makes bad dates fail loudly instead of becoming NaT.
    raw_data["date"] = pd.to_datetime(raw_data["date"], errors="raise")
    raw_data[TARGET_COLUMN] = pd.to_numeric(raw_data[TARGET_COLUMN], errors="raise")

    if raw_data[TARGET_COLUMN].isna().any():
        raise ValueError("Occupancy column contains missing values.")
    if raw_data["date"].duplicated().any():
        raise ValueError("Dates must be unique (one row per day).")

    minimum_rows = max(LAG_DAYS) + ROLLING_WINDOW + 3
    if len(raw_data) < minimum_rows:
        raise ValueError(f"Need at least {minimum_rows} rows, found {len(raw_data)}.")

    # Sorting is essential: shift() and rolling() assume time order.
    return raw_data.sort_values("date").reset_index(drop=True)


def add_features(occupancy_data: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features using only past days.

    Rows at the start that do not have enough history (NaN features)
    are dropped, so no values are filled in or guessed.
    """
    features = occupancy_data.copy()
    target = features[TARGET_COLUMN]

    # Lag features: the value from N days ago.
    for lag in LAG_DAYS:
        features[f"lag_{lag}"] = target.shift(lag)

    # Rolling stats: shift(1) FIRST, so the 7-day window covers t-7 ... t-1
    # and never includes today's value (that would be leakage).
    past_values = target.shift(1)
    features["rolling_mean_7"] = past_values.rolling(window=ROLLING_WINDOW).mean()
    features["rolling_std_7"] = past_values.rolling(window=ROLLING_WINDOW).std()

    # Calendar features are known in advance, so they are safe to use.
    features["day_of_week"] = features["date"].dt.day_of_week
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

    return features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def split_by_time(
    feature_data: pd.DataFrame,
    train_share: float = TRAIN_SHARE,
    validation_share: float = VALIDATION_SHARE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows in date order into train, validation, and test sets.

    We slice by position on date-sorted data. No random sampling is used.
    """
    if not (0 < train_share < 1 and 0 < validation_share < 1):
        raise ValueError("Shares must be between 0 and 1.")
    if train_share + validation_share >= 1:
        raise ValueError("train_share + validation_share must leave room for a test set.")

    sorted_data = feature_data.sort_values("date").reset_index(drop=True)
    total_rows = len(sorted_data)
    train_end = int(total_rows * train_share)
    validation_end = int(total_rows * (train_share + validation_share))

    train_data = sorted_data.iloc[:train_end].copy()
    validation_data = sorted_data.iloc[train_end:validation_end].copy()
    test_data = sorted_data.iloc[validation_end:].copy()

    check_split_integrity(train_data, validation_data, test_data)
    return train_data, validation_data, test_data


def check_split_integrity(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Stop with an error if the split is not strictly chronological."""
    for name, part in [("train", train_data), ("validation", validation_data), ("test", test_data)]:
        if part.empty:
            raise ValueError(f"The {name} split is empty.")
        if not part["date"].is_monotonic_increasing:
            raise ValueError(f"The {name} split is not in date order.")

    if not train_data["date"].max() < validation_data["date"].min():
        raise ValueError("Leakage: train dates overlap or come after validation dates.")
    if not validation_data["date"].max() < test_data["date"].min():
        raise ValueError("Leakage: validation dates overlap or come after test dates.")


def load_and_split(
    csv_path: Path = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline: load -> add features -> chronological split."""
    raw_data = load_raw_data(csv_path)
    feature_data = add_features(raw_data)
    return split_by_time(feature_data)


def describe_split(name: str, part: pd.DataFrame) -> str:
    """Return one readable line describing a split."""
    return (
        f"{name:<11} {len(part):>4} rows  "
        f"{part['date'].min().date()} -> {part['date'].max().date()}"
    )


def main() -> None:
    try:
        train_data, validation_data, test_data = load_and_split()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print("Chronological split (no shuffling):")
    print(describe_split("Train", train_data))
    print(describe_split("Validation", validation_data))
    print(describe_split("Test", test_data))
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print("Integrity check passed: train < validation < test, no overlapping dates.")


if __name__ == "__main__":
    main()
