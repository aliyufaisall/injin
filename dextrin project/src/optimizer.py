import pandas as pd
from scipy.optimize import differential_evolution


# =====================================================================
# STEP 1: Dextrin Economics Calculator (with Duration-Scaled Energy)
# =====================================================================
dextrin_price_per_kg: float = 0.0,

def calculate_dextrin_economics(
    predicted_yield_pct: float,
    batch_weight_kg: float,
    cassava_price_per_kg: float,
    water_ratio: float,
    enzyme_dosage_g_ml: float,
    enzyme_price: float,
    water_price: float,
    total_process_min: float,
    baseline_process_min: float,
    baseline_energy_kwh: float,
    energy_price: float,
    labour_cost: float,
) -> dict:
    """Calculates costs, revenue, and net profit (₦) for a dextrin batch."""

    # 1. Material & Energy Quantities
    total_water_liters = batch_weight_kg * water_ratio

    # Scale energy proportionally to total cooking + liquefaction minutes
    if baseline_process_min > 0:
        kwh_per_min = baseline_energy_kwh / baseline_process_min
        scaled_energy_kwh = kwh_per_min * total_process_min
    else:
        scaled_energy_kwh = baseline_energy_kwh

    # 2. Cost Breakdown (matching your exact Streamlit formulas)
    cost_raw_material = batch_weight_kg * cassava_price_per_kg
    cost_enzymes = (enzyme_dosage_g_ml * total_water_liters) * enzyme_price
    cost_water = total_water_liters * water_price
    cost_energy = scaled_energy_kwh * energy_price

    total_cost = (
        cost_raw_material
        + cost_enzymes
        + cost_water
        + cost_energy
        + labour_cost
    )

    # 3. Output Mass, Revenue & Net Profit
    dextrin_produced_kg = batch_weight_kg * (predicted_yield_pct / 100.0)

    return {
        "scaled_energy_kwh": scaled_energy_kwh,
        "cost_raw_material": cost_raw_material,
        "cost_enzymes": cost_enzymes,
        "cost_water": cost_water,
        "cost_energy": cost_energy,
        "total_cost": total_cost,
        "dextrin_produced_kg": dextrin_produced_kg,
    }

# =====================================================================
# STEP 2: Dextrin Feature Builder (Exact 13 Training Columns)
# =====================================================================
def build_dextrin_dataframe(fixed_features: dict, knobs: tuple) -> pd.DataFrame:
    """
    Combines the 6 fixed raw-material inputs and 7 controllable factory knobs
    into a single-row DataFrame matching your pipeline's exact column names.
    """
    (
        cook_temp,
        water_ratio,
        cook_duration,
        enzyme_dosage,
        liq_temp,
        liq_duration,
        process_ph,
    ) = knobs

    return pd.DataFrame(
        {
            "Cassava Starch Content (%)": [
                fixed_features["Cassava Starch Content (%)"]
            ],
            "Cassava Moisture Content (%)": [
                fixed_features["Cassava Moisture Content (%)"]
            ],
            "Raw Material Storage Duration (Days)": [
                fixed_features["Raw Material Storage Duration (Days)"]
            ],
            "Storage Temperature (°C)": [
                fixed_features["Storage Temperature (°C)"]
            ],
            "Storage Humidity (%)": [
                fixed_features["Storage Humidity (%)"]
            ],
            "Cooking Temperature (°C)": [cook_temp],
            "Water-to-Cassava Ratio": [water_ratio],
            "Cooking Duration (Min)": [cook_duration],
            "Alpha-Amylase Activity (U/g)": [
                fixed_features["Alpha-Amylase Activity (U/g)"]
            ],
            "Alpha-Amylase Dosage (g/mL)": [enzyme_dosage],
            "Liquefaction Temperature (°C)": [liq_temp],
            "Liquefaction Duration (Min)": [liq_duration],
            "pH During Enzymatic Treatment": [process_ph],
        }
    )

from pathlib import Path
import joblib
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, TransformerMixin


# (Your Step 1: calculate_dextrin_economics stays right here)
# (Your Step 2: build_dextrin_dataframe stays right here)


# =====================================================================
# STEP 3: The Dextrin Optimizer Runner (All 7 Controllable Knobs)
# =====================================================================
def optimize_dextrin_setpoints(
    model,
    fixed_features: dict,
    economics: dict,
    bounds: list[tuple[float, float]] | None = None,
) -> dict:
    """
    Searches for the 7 controllable factory setpoints that maximize
    Net Profit (₦) for a cassava dextrin liquefaction batch.
    """
    if bounds is None:
        # Order MUST match build_dextrin_dataframe's knobs unpacking:
        # (cook_temp, water_ratio, cook_duration, enzyme_dosage, liq_temp, liq_duration, process_ph)
        bounds = [
            (50.0, 120.0),   # Cooking Temperature (°C)
            (1.0, 10.0),     # Water-to-Cassava Ratio
            (5.0, 180.0),    # Cooking Duration (Min)
            (0.001, 0.100),  # Alpha-Amylase Dosage (g/mL)
            (50.0, 110.0),   # Liquefaction Temperature (°C)
            (10.0, 240.0),   # Liquefaction Duration (Min)
            (3.0, 9.0),      # pH During Enzymatic Treatment
        ]

    # The scorer function that scipy tries to minimize
    def objective(knobs):
        _, water_ratio, cook_duration, enzyme_dosage, _, liq_duration, _ = knobs
        total_process_min = cook_duration + liq_duration

        df = build_dextrin_dataframe(fixed_features, knobs)
        predicted_yield = float(model.predict(df)[0])

        econ_result = calculate_dextrin_economics(
            predicted_yield_pct=predicted_yield,
            batch_weight_kg=economics["batch_weight_kg"],
            cassava_price_per_kg=economics["cassava_price_per_kg"],
            water_ratio=water_ratio,
            enzyme_dosage_g_ml=enzyme_dosage,
            enzyme_price=economics["enzyme_price"],
            water_price=economics["water_price"],
            total_process_min=total_process_min,
            baseline_process_min=economics["baseline_process_min"],
            baseline_energy_kwh=economics["baseline_energy_kwh"],
            energy_price=economics["energy_price"],
            labour_cost=economics["labour_cost"],
        )

        # Return negative profit because differential_evolution minimizes

        if econ_result["total_cost"] > economics["max_batch_budget"]:
            overspend = econ_result["total_cost"] - economics["max_batch_budget"]
            return float(1_000_000 + overspend)
        return float(-predicted_yield)



    # Run Differential Evolution (~2,500 fast pipeline predictions in <0.5s)
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=45,
        popsize=8,
        seed=42,
        polish=False,  # Must stay False for tree-based models
    )

    # Unpack the winning 7 setpoints
    (
        opt_cook_temp,
        opt_water_ratio,
        opt_cook_min,
        opt_dosage,
        opt_liq_temp,
        opt_liq_min,
        opt_ph,
    ) = result.x

    # Run one final evaluation with the winning setpoints
    optimal_df = build_dextrin_dataframe(fixed_features, result.x)
    optimal_yield = float(model.predict(optimal_df)[0])
    optimal_economics = calculate_dextrin_economics(
        predicted_yield_pct=optimal_yield,
        batch_weight_kg=economics["batch_weight_kg"],
        cassava_price_per_kg=economics["cassava_price_per_kg"],
        water_ratio=opt_water_ratio,
        enzyme_dosage_g_ml=opt_dosage,
        enzyme_price=economics["enzyme_price"],
        water_price=economics["water_price"],
        total_process_min=opt_cook_min + opt_liq_min,
        baseline_process_min=economics["baseline_process_min"],
        baseline_energy_kwh=economics["baseline_energy_kwh"],
        energy_price=economics["energy_price"],
        labour_cost=economics["labour_cost"],
    )

    return {
        "setpoints": {
            "Cooking Temperature (°C)": round(float(opt_cook_temp), 1),
            "Water-to-Cassava Ratio": round(float(opt_water_ratio), 2),
            "Cooking Duration (Min)": round(float(opt_cook_min), 0),
            "Alpha-Amylase Dosage (g/mL)": round(float(opt_dosage), 4),
            "Liquefaction Temperature (°C)": round(float(opt_liq_temp), 1),
            "Liquefaction Duration (Min)": round(float(opt_liq_min), 0),
            "pH During Enzymatic Treatment": round(float(opt_ph), 2),
        },
        "predicted_yield_pct": round(optimal_yield, 2),
        "economics": optimal_economics,
    }


# =====================================================================
# STEP 4: Standalone Terminal Test Block
# =====================================================================
if __name__ == "__main__":
    # Needed only when running standalone so joblib can unpickle the pipeline
    class CassavaFeatureEngineer(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            X_out = X.copy()
            X_out["Storage_Spoilage_Penalty"] = (
                X_out["Raw Material Storage Duration (Days)"]
                * X_out["Storage Temperature (°C)"]
                * X_out["Storage Humidity (%)"]
            )
            cooking_heat_load = (
                X_out["Cooking Temperature (°C)"]
                * X_out["Cooking Duration (Min)"]
            )
            X_out["Thermal_Hydration_Index"] = cooking_heat_load / (
                X_out["Water-to-Cassava Ratio"] + 1e-5
            )
            X_out["Total_Enzyme_Power_Pool"] = (
                X_out["Alpha-Amylase Activity (U/g)"]
                * X_out["Alpha-Amylase Dosage (g/mL)"]
            )
            X_out["Liquefaction_Heat_Stress"] = (
                X_out["Liquefaction Temperature (°C)"]
                * X_out["Liquefaction Duration (Min)"]
            )
            return X_out

    root_dir = Path(__file__).resolve().parent.parent
    model_path = root_dir / "models" / "production_pipeline.joblib"
    loaded_model = joblib.load(model_path)

    sample_fixed = {
        "Cassava Starch Content (%)": 30.0,
        "Cassava Moisture Content (%)": 65.0,
        "Raw Material Storage Duration (Days)": 3,
        "Storage Temperature (°C)": 28.0,
        "Storage Humidity (%)": 75.0,
        "Alpha-Amylase Activity (U/g)": 1500.0,
    }

    sample_economics = {
        "batch_weight_kg": 1000.0,
        "cassava_price_per_kg": 150.0,
        "enzyme_price": 500.0,
        "water_price": 5.0,
        "baseline_process_min": 45 + 90,  # 135 total minutes
        "baseline_energy_kwh": 120.0,
        "energy_price": 110.0,
        "labour_cost": 25000.0,
        "dextrin_price_per_kg": 1200.0,
    }

    output = optimize_dextrin_setpoints(loaded_model, sample_fixed, sample_economics)

    print("=== OPTIMAL DEXTRIN SETPOINTS ===")
    for param, val in output["setpoints"].items():
        print(f"{param}: {val}")

    print(f"\nPredicted Dextrin Yield: {output['predicted_yield_pct']}%")
    print(f"Dextrin Output Mass:     {output['economics']['dextrin_produced_kg']:,.1f} kg")
    print(f"Scaled Energy Used:      {output['economics']['scaled_energy_kwh']:.1f} kWh")
    print(f"Total Batch Cost:        ₦{output['economics']['total_cost']:,.2f}")
    
