Problem Statement: HiveSpace Desk Occupancy Data Foundation

Core Goal: Establish a robust data foundation and baseline predictive models for HiveSpace desk occupancy by generating 18 months (548 days) of synthetic time-series data, building a feature-engineering pipeline, and training an initial XGBoost model while strictly preventing future data leakage.

Current Pain Points:

Lack of real-world historical data requires the creation of realistic synthetic data exhibiting weekly seasonality and random noise.

High risk of data leakage inherent in time-series machine learning, necessitating strict chronological train/validation/test splits (no random shuffling).

Need for a reproducible, well-documented workflow that proves the integrity of the data split to stakeholders or mentors.

Desired Outcome:

A Python script to generate the 548-day synthetic dataset.

A pipeline script to load raw data, engineer lag features, and perform a strict time-based split.

Training and evaluation of a Moving Average baseline and an XGBoost regressor, outputting MAE scores to the console and saving them to a metrics.json file.

A complete GitHub repository setup containing a README.md with a detailed "Data Split Strategy" section.

A predefined folder architecture and dependency list (libraries.md) to structure the repository.