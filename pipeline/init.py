"""Master runner for the entire HiveSpace desk occupancy pipeline.

This script executes all three pipeline stages in order:
1. Synthetic Data Generation (generate_data.py)
   Creates 548 days of occupancy data with weekly seasonality.
2. Feature Engineering & Chronological Split (process_split.py)
   Builds strictly past-looking lag features and validates split boundaries.
3. Model Training & Evaluation (train_evaluate.py)
   Fits the 7-day Moving Average baseline and XGBoost regressor,
   prints MAE scores, and saves metrics.json.

Run from the project root:
    python pipeline/init.py
"""

import sys
from pathlib import Path

# Ensure the pipeline directory is in sys.path so modules import reliably
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import generate_data
import process_split
import train_evaluate


def run_pipeline() -> None:
    """Run all pipeline stages in sequence."""
    print("=" * 65)
    print("STAGE 1: Generating Synthetic Occupancy Data")
    print("=" * 65)
    generate_data.main()
    print()

    print("=" * 65)
    print("STAGE 2: Feature Engineering & Chronological Split Verification")
    print("=" * 65)
    process_split.main()
    print()

    print("=" * 65)
    print("STAGE 3: Baseline & XGBoost Model Training & MAE Evaluation")
    print("=" * 65)
    train_evaluate.main()
    print()

    print("=" * 65)
    print("COMPLETE: Whole pipeline executed successfully.")
    print("=" * 65)


def main() -> None:
    """Main entrypoint for pipeline/init.py."""
    try:
        run_pipeline()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Pipeline failed with error: {error}") from error


if __name__ == "__main__":
    main()
