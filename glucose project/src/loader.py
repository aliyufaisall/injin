# src/data_loader.py
import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import EXPECTED_FEATURES, RANDOM_SEED, TARGET_COLUMN


def load_and_verify_data(file_path: str) -> pd.DataFrame:
    """Reads raw CSV data and runs safety checks to ensure structural alignment."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Project data missing at destination: {file_path}"
        )

    df = pd.read_csv(file_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Phase 1 target variable '{TARGET_COLUMN}' missing from source data."
        )

    # Sanity Check 2: Structural Verification
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            raise KeyError(
                f"Structural violation: Missing expected processing metric '{col}'."
            )

    return df


def split_to_three_way(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Splits data cleanly into Train (70%), Validation (15%), and an isolated Test Lockbox (15%)."""
    X = df[EXPECTED_FEATURES]
    y = df[TARGET_COLUMN]

    # Step 1: Separate out the strict Test Lockbox (15% of the total dataset)
    X_train_val, X_lockbox, y_train_val, y_lockbox = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED
    )

    # Step 2: Split the remaining 85% into Train and Validation
    # 0.1765 of 85% is approximately 15% of the total dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=RANDOM_SEED,
    )

    return X_train, X_val, X_lockbox, y_train, y_val, y_lockbox
