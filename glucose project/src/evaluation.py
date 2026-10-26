# src/evaluation.py
import numpy as np

def compute_yield_production_metrics(y_true, y_pred) -> dict:
    """
    Calculates standard physical accuracy metrics alongside our asymmetric financial 
    risk metrics tailored for glucose yield predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # XGBoost structural error convention: (predictions - actuals)
    errors = y_pred - y_true

    # Standard Symmetric Metric (Physical Mass Deviation in g/L)
    rmse = np.sqrt(np.mean(errors**2))

    # Asymmetric Financial Loss: Penalizes promising glucose we cannot actually produce
    # If error > 0 (over-predicted yield), multiply penalty weight by 3.0
    asymmetric_weights = np.where(errors > 0, 3.0, 1.0)
    financial_risk_score = np.mean(asymmetric_weights * (errors**2))

    # Track operational compliance: Count how many times the model over-estimated yield
    over_prediction_count = np.sum(errors > 0)
    total_predictions = len(errors)
    over_prediction_rate = (over_prediction_count / total_predictions) * 100

    return {
        "Glucose_RMSE_g_L": rmse,
        "Financial_Risk_Loss": financial_risk_score,
        "Yield_Over_Prediction_Rate": over_prediction_rate,
    }

import numpy as np

def analyze_overprediction(y_true, y_pred):
    errors = np.asarray(y_pred) - np.asarray(y_true)

    # Keep only over-predictions
    overpredictions = errors[errors > 0]

    if len(overpredictions) == 0:
        return {
            "Overprediction_Count": 0,
            "Average_Overprediction_g_L": 0.0,
            "Worst_Overprediction_g_L": 0.0,
        }

    return {
        "Overprediction_Count": len(overpredictions),
        "Average_Overprediction_g_L": np.mean(overpredictions),
        "Worst_Overprediction_g_L": np.max(overpredictions),
    }
