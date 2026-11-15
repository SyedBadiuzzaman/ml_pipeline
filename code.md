High-level approach
The system is a sequence of standalone Python scripts that simulate desk usage, process it for machine learning, and train predictive models. First, we generate 18 months of synthetic daily desk occupancy data that mimics real-world weekly patterns (e.g., busy mid-week, empty weekends). Then, a data pipeline cleans this data and engineers "lag features" (using past days to predict future days) while strictly splitting the data chronologically into past, present, and future segments to avoid data leakage. Finally, we train a simple baseline model and a more complex XGBoost model on the past data, evaluating their accuracy on the unseen future data to prove the pipeline works.

Main components
Component 1: Data Generator (generate_data.py): Responsible for creating 548 days of synthetic time-series data using NumPy and Pandas, injecting weekly seasonality and random noise, and saving it as a raw CSV file.

Component 2: Data Pipeline (process_split.py): Responsible for loading the raw CSV, calculating rolling metrics/lag features, and strictly slicing the dataset by date into training, validation, and testing sets without random shuffling.

Component 3: Model Trainer (train_evaluate.py): Responsible for fitting a Moving Average baseline and an XGBoost regressor to the training data, calculating the Mean Absolute Error (MAE) on the test data, and saving the final metrics to metrics.json.

Data flow
Developer executes data generation: The system creates and saves synthetic_occupancy.csv in the data/ folder.

Developer triggers the pipeline: The app reads the raw CSV, applies lag feature engineering, and performs the strict time-based split.

Developer initiates training: The app processes the training split through the Moving Average and XGBoost algorithms.

App returns result: The system prints the MAE comparison to the console and serializes the exact scores into metrics.json for review.

Important rules
Strict chronological splitting: Never use random sampling or train_test_split with shuffle=True. Time-series data must be split linearly.

Zero future leakage: Ensure lag features (e.g., "yesterday's occupancy") never accidentally include data from the target prediction date.

Keep code simple: Rely on native Pandas vectorized operations for feature engineering rather than complex loops.