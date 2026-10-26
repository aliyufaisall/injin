import numpy as np
import sys
from pathlib import Path

# Resolve path mapping for VS Code execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation import compute_yield_production_metrics
from src.loader import load_and_verify_data, split_to_three_way


def evaluate_mean_baseline(y_train, y_val):
    """Predicts the training-set mean for every validation sample."""

    # Learn the mean from training data only to prevent data leakage
    mean_yield = np.mean(y_train)

    # Predict the same mean for every validation sample
    y_pred = np.full(len(y_val), mean_yield)

    # Evaluate using the same metrics as the production model
    metrics = compute_yield_production_metrics(y_val, y_pred)

    print("\n=== [BASELINE] MEAN PREDICTION ===")
    print(f"Mean yield prediction (g/L): {mean_yield:.4f}")
    print(f"Glucose RMSE (g/L):          {metrics['Glucose_RMSE_g_L']:.4f}")
    print(f"Financial Risk Loss:         {metrics['Financial_Risk_Loss']:.4f}")
    print(
        f"Over-Prediction Rate (%):    "
        f"{metrics['Yield_Over_Prediction_Rate']:.2f}%"
    )
    print("=================================\n")

    return metrics


if __name__ == "__main__":
    RAW_DATA_PATH = "glucose project/data/saccharification_process_data.csv"

    df = load_and_verify_data(RAW_DATA_PATH)

    X_train, X_val, X_lockbox, y_train, y_val, y_lockbox = (
        split_to_three_way(df)
    )

    print(f"[INFO] Training samples: {len(y_train)}")
    print(f"[INFO] Validation samples: {len(y_val)}")

    baseline_metrics = evaluate_mean_baseline(y_train, y_val)

