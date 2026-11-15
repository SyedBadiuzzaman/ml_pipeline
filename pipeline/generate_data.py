"""Generate 548 days of synthetic HiveSpace desk occupancy data.

The data copies a realistic office pattern:
- busy Tuesday to Thursday,
- quieter Monday and Friday,
- almost empty weekends,
- a gentle yearly wave (for example, summer holidays),
- random day-to-day noise.

Run from the project root:
    python pipeline/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Project paths are resolved from this file, so the script works
# no matter which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic_occupancy.csv"

START_DATE = "2025-01-01"
NUMBER_OF_DAYS = 548  # 18 months
TOTAL_DESKS = 100
RANDOM_SEED = 42  # Fixed seed so every run creates the same data.
NOISE_STD = 6.0  # Typical random change in desks used per day.

# Average desks used on each weekday (0 = Monday ... 6 = Sunday).
WEEKDAY_BASE_OCCUPANCY = {
    0: 52,  # Monday
    1: 76,  # Tuesday
    2: 80,  # Wednesday
    3: 74,  # Thursday
    4: 46,  # Friday
    5: 8,  # Saturday
    6: 5,  # Sunday
}


def generate_occupancy_data(
    start_date: str = START_DATE,
    number_of_days: int = NUMBER_OF_DAYS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create a DataFrame with one row per day: `date` and `occupancy`.

    Args:
        start_date: First date of the series (YYYY-MM-DD).
        number_of_days: How many consecutive days to create.
        seed: Random seed used for the noise, for reproducible output.

    Returns:
        DataFrame with columns `date` (datetime) and `occupancy` (int, 0-100).
    """
    if number_of_days <= 0:
        raise ValueError("number_of_days must be a positive number.")

    random_generator = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=number_of_days, freq="D")

    # 1) Weekly seasonality: look up the base value for each day of the week.
    weekly_pattern = dates.day_of_week.map(WEEKDAY_BASE_OCCUPANCY).to_numpy(dtype=float)

    # 2) Yearly wave: a small sine curve, lowest around mid-summer.
    #    Scaled by the weekly value so weekends stay close to empty.
    day_of_year = dates.day_of_year.to_numpy()
    yearly_factor = 1 + 0.08 * np.cos(2 * np.pi * day_of_year / 365.25)

    # 3) Random noise so no two weeks look exactly the same.
    noise = random_generator.normal(loc=0.0, scale=NOISE_STD, size=number_of_days)

    occupancy = weekly_pattern * yearly_factor + noise

    # Desk counts must be whole numbers between 0 and the total desks.
    occupancy = np.clip(np.round(occupancy), 0, TOTAL_DESKS).astype(int)

    return pd.DataFrame({"date": dates, "occupancy": occupancy})


def save_occupancy_data(occupancy_data: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> Path:
    """Save the data to CSV (creating the `data/` folder if needed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    occupancy_data.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return output_path


def main() -> None:
    occupancy_data = generate_occupancy_data()
    saved_path = save_occupancy_data(occupancy_data)

    print(f"Saved {len(occupancy_data)} days of data to {saved_path}")
    print(
        f"Date range: {occupancy_data['date'].min().date()} "
        f"to {occupancy_data['date'].max().date()}"
    )
    print(
        f"Occupancy: min={occupancy_data['occupancy'].min()}, "
        f"mean={occupancy_data['occupancy'].mean():.1f}, "
        f"max={occupancy_data['occupancy'].max()}"
    )


if __name__ == "__main__":
    main()
