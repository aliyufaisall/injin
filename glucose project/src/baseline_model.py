import sys
from pathlib import Path

# Resolve path mapping for VS Code execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.tree import DecisionTreeRegressor
from src.config import RANDOM_SEED
from src.evaluation import compute_yield_production_metrics
# Import your exact production ingestion functions
from src.loader import load_and_verify_data, split_to_three_way


def train_dumb_baseline(X_train_engineered, y_train, X_val_engineered, y_val):
    """Trains a baseline decision tree and evaluates it against our custom yield metrics."""
    # max_depth=3 keeps the tree small and easy to inspect
    model = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED)
    model.fit(X_train_engineered, y_train)

    # Generate predictions on validation data
    y_pred = model.predict(X_val_engineered)

    # Compute standard and financial risk metrics
    metrics = compute_yield_production_metrics(y_val, y_pred)

    print("\n=== [BASELINE] PERFORMANCE FLOOR ===")
    print(f"Glucose RMSE (g/L):        {metrics['Glucose_RMSE_g_L']:.4f}")
    print(f"Financial Risk Loss:       {metrics['Financial_Risk_Loss']:.4f}")
    print(f"Over-Prediction Rate (%):  {metrics['Yield_Over_Prediction_Rate']:.2f}%")
    print("====================================\n")

    return model


# Executes natively when you click the VS Code Play Button
if __name__ == "__main__":
    # TODO: Update this string to point directly to your raw data path
    RAW_DATA_PATH = "glucose project/data/saccharification_process_data.csv"

    print(f"[INFO] Ingesting and verifying source data from: {RAW_DATA_PATH}")

    try:
        # Pipeline execution matching your internal framework
        df = load_and_verify_data(RAW_DATA_PATH)
        
        # Unpack the 3-way splits directly
        X_train, X_val, X_lockbox, y_train, y_val, y_lockbox = split_to_three_way(df)
        
        print(f"[INFO] Data unpacked successfully. Training sample count: {X_train.shape[0]}")
        
        # Execute training (The lockbox sets are isolated and untouched)
        baseline_model = train_dumb_baseline(X_train, y_train, X_val, y_val)
        
    except FileNotFoundError as fnf_err:
        print(f"\n[ERROR] Data execution halted: {fnf_err}")
        print("Please ensure your local target file exists or update RAW_DATA_PATH in this script.\n")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed during processing: {str(e)}")
