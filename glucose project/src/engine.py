# src/engine.py
import numpy as np
import xgboost as xgb
from src.config import RANDOM_SEED

def financial_risk_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Custom asymmetric loss function to penalize over-predictions.
    Ensures safe biological yield estimates by scaling over-predictions by 3.0.
    """
    # XGBoost internal convention: (predictions - actuals)
    errors = y_pred - y_true 
    
    # Apply 3.0 penalty weight when the model over-predicts (errors > 0)
    weights = np.where(errors > 0, 3.0, 1.0)
    
    # Compute first and second derivatives for gradient boosting
    gradient = 2 * weights * errors
    hessian = 2 * weights
    
    return gradient, hessian


def build_production_engine() -> xgb.XGBRegressor:
    """Initializes the XGBoost model with tuned hyperparameters."""
    return xgb.XGBRegressor(
        objective=None,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        gamma=0.5,
        reg_alpha=1,
        reg_lambda=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
