import sys
from pathlib import Path

# Resolve path mapping for VS Code execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import joblib
import xgboost as xgb

from src.config import RAW_DATA_PATH
from src.evaluation import (
    compute_yield_production_metrics,
    analyze_overprediction,
)
from src.loader import load_and_verify_data, split_to_three_way
from src.engine import build_production_engine


if __name__ == "__main__":
    print("Initializing Raw Data XGBoost Training, Validation & Test Pipeline...")

    # Step 1: Load data and create splits
    raw_df = load_and_verify_data(RAW_DATA_PATH)
    X_train, X_val, X_lockbox, y_train, y_val, y_lockbox = split_to_three_way(raw_df)

    print(
        f"[INFO] Splits isolated. Train: {X_train.shape} | "
        f"Val: {X_val.shape} | Lockbox: {X_lockbox.shape}"
    )

    # Step 2: Initialize XGBoost with its default objective
    xgb_regressor = build_production_engine()

    # Step 3: Train on raw features
    print("\nFitting standard squared-error XGBoost model...")
    xgb_regressor.fit(X_train, y_train)

    # PHASE 1: VALIDATION AUDIT
    print("\nAuditing candidate model on validation split...")
    val_predictions = xgb_regressor.predict(X_val)
    val_metrics = compute_yield_production_metrics(y_val, val_predictions)

    print("\n==================================================")
    print("CANDIDATE: VALIDATION SPLIT AUDIT REPORT")
    print("==================================================")
    print(f" - Standard RMSE:       {val_metrics['Glucose_RMSE_g_L']:.4f}")
    print(f" - Financial Risk Loss: {val_metrics['Financial_Risk_Loss']:.4f}")
    print(f" - Over-Prediction Rate: {val_metrics['Yield_Over_Prediction_Rate']:.1f}%")
    print("==================================================")

    # PHASE 2: LOCKBOX AUDIT
    print("\nEvaluating on the lockbox...")
    test_predictions = xgb_regressor.predict(X_lockbox)
    test_metrics = compute_yield_production_metrics(y_lockbox, test_predictions)

    print("\n==================================================")
    print("CANDIDATE: FINAL TEST LOCKBOX AUDIT REPORT")
    print("==================================================")
    print(f" - Standard RMSE:       {test_metrics['Glucose_RMSE_g_L']:.4f}")
    print(f" - Financial Risk Loss: {test_metrics['Financial_Risk_Loss']:.4f}")
    print(f" - Over-Prediction Rate: {test_metrics['Yield_Over_Prediction_Rate']:.1f}%")
    print("==================================================")

    # PHASE 2B: OVER-PREDICTION ANALYSIS
    overprediction_metrics = analyze_overprediction(
        y_lockbox,
        test_predictions,
    )

    print("\n=== LOCKBOX OVER-PREDICTION ANALYSIS ===")
    print(
        f"Average over-prediction: "
        f"{overprediction_metrics['Average_Overprediction_g_L']:.2f} g/L"
    )
    print(
        f"Worst over-prediction: "
        f"{overprediction_metrics['Worst_Overprediction_g_L']:.2f} g/L"
    )
    print(
        f"Number of over-predictions: "
        f"{overprediction_metrics['Overprediction_Count']}"
    )

    # PHASE 3: SAVE MODEL
    output_dir = "models"
    model_filename = os.path.join(
        output_dir,
        "xgboost_production_model2.joblib",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Saving model to '{model_filename}'...")
    joblib.dump(xgb_regressor, model_filename)
    print("[SUCCESS] Model artifact exported successfully!")
