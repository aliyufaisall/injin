import sys
from pathlib import Path

# Resolve path mapping for VS Code execution
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import root_mean_squared_error

from src.config import RAW_DATA_PATH
from src.loader import load_and_verify_data, split_to_three_way
from src.engine import build_production_engine, financial_risk_objective
from src.evaluation import compute_yield_production_metrics


def tune_xgboost():
    print("[INFO] Loading data...")
    df = load_and_verify_data(RAW_DATA_PATH)

    X_train, X_val, X_lockbox, y_train, y_val, y_lockbox = (
        split_to_three_way(df)
    )

    # Keep the best configuration from your previous tuning round fixed.
    param_grid = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07],
    "n_estimators": [300, 500, 700, 1000],
    "max_depth": [4],
    "min_child_weight": [3],
    "gamma": [0.5],
    "reg_alpha": [1],
    "reg_lambda": [5],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    }
    best_rmse = float("inf")
    best_params = None
    results = []

    total = len(list(ParameterGrid(param_grid)))
    print(f"[INFO] Testing {total} configurations...")

    for i, params in enumerate(ParameterGrid(param_grid), start=1):
        model = build_production_engine()
        model.set_params(
            **params,
            objective=financial_risk_objective,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        metrics = compute_yield_production_metrics(y_val, predictions)
        rmse = metrics["Glucose_RMSE_g_L"]
        risk = metrics["Financial_Risk_Loss"]

        results.append({
            "rmse": rmse,
            "risk": risk,
            "params": params,
        })

        print(
            f"[{i}/{total}] "
            f"RMSE={rmse:.4f} | "
            f"Risk={risk:.4f}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    results.sort(key=lambda result: result["rmse"])

    print("\n=== TOP 5 CONFIGURATIONS ===")
    for rank, result in enumerate(results[:5], start=1):
        print(
            f"{rank}. RMSE={result['rmse']:.4f} | "
            f"Risk={result['risk']:.4f} | "
            f"Params={result['params']}"
        )

    print("\n=== BEST VALIDATION CONFIGURATION ===")
    print(f"RMSE: {best_rmse:.4f}")
    print(f"Parameters: {best_params}")

    return best_params, best_rmse


if __name__ == "__main__":
    tune_xgboost()
