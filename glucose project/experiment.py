
# experiment.py

import numpy as np
import mlflow
mlflow.set_tracking_uri(
    "sqlite:///C:/Users/HP/Documents/engine/glucose%20project/mlflow.db"
)
import mlflow.xgboost

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.loader import load_and_verify_data, split_to_three_way
from src.config import RAW_DATA_PATH
from src.engine import financial_risk_objective
from src.evaluation import (
    compute_yield_production_metrics,
    analyze_overprediction
)


# =====================================================
# 1. EXPERIMENT CONFIGURATION
# =====================================================

mlflow.set_experiment("Glucose Yield Prediction")


# =====================================================
# 2. LOAD AND SPLIT DATA
# =====================================================

df = load_and_verify_data(RAW_DATA_PATH)

(
    X_train,
    X_val,
    X_lockbox,
    y_train,
    y_val,
    y_lockbox,
) = split_to_three_way(df)


# =====================================================
# 3. MODEL HYPERPARAMETERS
# =====================================================

params = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.5,
    "min_child_weight": 3,
    "reg_alpha": 1,
    "reg_lambda": 5,
}


# =====================================================
# 4. START EXPERIMENT
# =====================================================

with mlflow.start_run():

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Active run:", mlflow.active_run().info.run_id)
    print("Experiment ID:", mlflow.active_run().info.experiment_id)
    # Initialize model
    model = XGBRegressor(
        **params,
        objective=financial_risk_objective,
        random_state=42,
        eval_metric="rmse",
    )

    # Train model
    model.fit(X_train, y_train)
    # =================================================
    # 5. VALIDATION
    # =================================================

    predictions = model.predict(X_val)

    metrics = compute_yield_production_metrics(
        y_val, predictions
    )

    overprediction_metrics = analyze_overprediction(
        y_val, predictions
    )

    all_metrics = {
        **metrics,
        **overprediction_metrics
    }

    # =================================================
    # 6. LOG PARAMETERS
    # =================================================

    mlflow.log_params(params)

    mlflow.log_param(
        "objective",
        "financial_risk_objective"
    )

    mlflow.log_param(
        "overprediction_penalty",
        3.0
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "training_rows",
        len(X_train)
    )

    mlflow.log_param(
        "validation_rows",
        len(X_val)
    )

    mlflow.log_param(
        "feature_count",
        X_train.shape[1]
    )

    # =================================================
    # 7. LOG METRICS
    # =================================================

    mlflow.log_metrics(all_metrics)

    # =================================================
    # 8. LOG MODEL
    # =================================================

    mlflow.xgboost.log_model(
    model,
    name="model"
)

    # =================================================
    # 9. LOG SOURCE CODE
    # =================================================

    mlflow.log_artifact("src/engine.py")

    # =================================================
    # 10. DISPLAY RESULTS
    # =================================================

    print("\nEXPERIMENT RESULTS")
    print("--------------------------")

    for metric, value in all_metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
