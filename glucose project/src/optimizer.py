import pandas as pd
from scipy.optimize import differential_evolution
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))



# =====================================================================
# STEP 1: Economics Calculator (with Duration-Scaled Energy)
# =====================================================================
def calculate_batch_economics(
    yield_g_L: float,
    substrate_vol: float,
    starch_conc: float,
    water_ratio: float,
    enzyme_dosage: float,
    duration_h: float,
    baseline_duration_h: float,
    baseline_energy_kwh: float,
    enzyme_price: float,
    water_price: float,
    energy_price: float,
    labour_cost: float,
    glucose_price_per_kg: float,
) -> dict:
    """Calculates costs, revenue, and net profit (₦) with scaled energy."""

    # 1. Material quantities
    total_water_liters = substrate_vol * water_ratio
    starch_mass_kg = (starch_conc * substrate_vol) / 1000.0
    enzyme_mass_g = enzyme_dosage * starch_mass_kg

    # 2. Scale energy proportionally to batch duration
    # Guard against division by zero if baseline_duration_h is 0
    if baseline_duration_h > 0:
        kwh_per_hour = baseline_energy_kwh / baseline_duration_h
        scaled_energy_kwh = kwh_per_hour * duration_h
    else:
        scaled_energy_kwh = baseline_energy_kwh

    # 3. Production costs
    cost_enzymes = enzyme_mass_g * enzyme_price
    cost_water = total_water_liters * water_price
    cost_energy = scaled_energy_kwh * energy_price
    total_cost = cost_enzymes + cost_water + cost_energy + labour_cost

    # 4. Revenue & Net Profit
    total_glucose_kg = (yield_g_L * substrate_vol) / 1000.0
    revenue = total_glucose_kg * glucose_price_per_kg
    net_profit = revenue - total_cost

    return {
        "scaled_energy_kwh": scaled_energy_kwh,
        "cost_enzymes": cost_enzymes,
        "cost_water": cost_water,
        "cost_energy": cost_energy,
        "total_cost": total_cost,
        "total_glucose_kg": total_glucose_kg,
        "revenue": revenue,
        "net_profit": net_profit,
    }


# =====================================================================
# STEP 2: XGBoost Feature Builder
# =====================================================================
def build_feature_dataframe(fixed_features: dict, knobs: tuple) -> pd.DataFrame:
    """
    Combines the 4 fixed batch inputs and 6 controllable knobs into a
    single-row DataFrame matching your XGBoost training columns exactly.
    """
    temp_c, ph, dosage, duration_h, water_ratio, rpm = knobs

    return pd.DataFrame(
        {
            "initial_starch_concentration_g_L": [
                fixed_features["initial_starch_concentration_g_L"]
            ],
            "initial_dextrin_avg_dp": [
                fixed_features["initial_dextrin_avg_dp"]
            ],
            "saccharification_temp_C": [temp_c],
            "pH": [ph],
            "glucoamylase_activity_U_g": [
                fixed_features["glucoamylase_activity_U_g"]
            ],
            "glucoamylase_dosage_g_kg": [dosage],
            "saccharification_duration_h": [duration_h],
            "water_to_substrate_ratio": [water_ratio],
            "initial_glucose_concentration_g_L": [
                fixed_features["initial_glucose_concentration_g_L"]
            ],
            "mixing_rate_rpm": [rpm],
        }
    )

from pathlib import Path
import joblib
import pandas as pd
from scipy.optimize import differential_evolution


# (Your Step 1: calculate_batch_economics stays right here)
# (Your Step 2: build_feature_dataframe stays right here)


# =====================================================================
# STEP 3: The Optimizer Runner (All 6 Controllable Knobs)
# =====================================================================
def optimize_batch_setpoints(
    model,
    fixed_features: dict,
    economics: dict,
    bounds: list[tuple[float, float]] | None = None,
) -> dict:
    """
    Searches for the 6 controllable setpoints (including mixing_rate_rpm)
    that maximize Net Profit (₦) using Differential Evolution.
    """
    if bounds is None:
        # Order MUST match build_feature_dataframe's knobs unpacking:
        # (temp_c, ph, dosage, duration_h, water_ratio, rpm)
        bounds = [
            (45.0, 75.0),   # saccharification_temp_C
            (3.5, 6.5),     # pH
            (0.1, 5.0),     # glucoamylase_dosage_g_kg
            (12.0, 96.0),   # saccharification_duration_h
            (1.0, 6.0),     # water_to_substrate_ratio
            (50.0, 400.0),  # mixing_rate_rpm
        ]

    # The scorer function that scipy tries to minimize
    def objective(knobs):
        _, _, dosage, duration_h, water_ratio, _ = knobs

        df = build_feature_dataframe(fixed_features, knobs)
        predicted_yield = float(model.predict(df)[0])

        econ_result = calculate_batch_economics(
            yield_g_L=predicted_yield,
            substrate_vol=economics["substrate_volume"],
            starch_conc=fixed_features["initial_starch_concentration_g_L"],
            water_ratio=water_ratio,
            enzyme_dosage=dosage,
            duration_h=duration_h,
            baseline_duration_h=economics["baseline_duration_h"],
            baseline_energy_kwh=economics["baseline_energy_kwh"],
            enzyme_price=economics["enzyme_price"],
            water_price=economics["water_price"],
            energy_price=economics["energy_price"],
            labour_cost=economics["labour_cost"],
            glucose_price_per_kg=economics["glucose_price_per_kg"],
        )

        # Return negative profit because differential_evolution minimizes
        return -econ_result["net_profit"]

    # Run Differential Evolution (~2,100 fast predictions in <0.4 seconds)
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=45,
        popsize=8,
        seed=42,
        polish=False,  # Must stay False for tree-based XGBoost models
    )

    # Extract the winning 6 setpoints
    opt_temp, opt_ph, opt_dosage, opt_duration, opt_water, opt_rpm = result.x

    # Run one final evaluation with the winning setpoints
    optimal_df = build_feature_dataframe(fixed_features, result.x)
    optimal_yield = float(model.predict(optimal_df)[0])
    optimal_economics = calculate_batch_economics(
        yield_g_L=optimal_yield,
        substrate_vol=economics["substrate_volume"],
        starch_conc=fixed_features["initial_starch_concentration_g_L"],
        water_ratio=opt_water,
        enzyme_dosage=opt_dosage,
        duration_h=opt_duration,
        baseline_duration_h=economics["baseline_duration_h"],
        baseline_energy_kwh=economics["baseline_energy_kwh"],
        enzyme_price=economics["enzyme_price"],
        water_price=economics["water_price"],
        energy_price=economics["energy_price"],
        labour_cost=economics["labour_cost"],
        glucose_price_per_kg=economics["glucose_price_per_kg"],
    )

    return {
        "setpoints": {
            "saccharification_temp_C": round(float(opt_temp), 2),
            "pH": round(float(opt_ph), 2),
            "glucoamylase_dosage_g_kg": round(float(opt_dosage), 3),
            "saccharification_duration_h": round(float(opt_duration), 1),
            "water_to_substrate_ratio": round(float(opt_water), 2),
            "mixing_rate_rpm": round(float(opt_rpm), 0),
        },
        "predicted_yield_g_L": round(optimal_yield, 2),
        "economics": optimal_economics,
    }


# =====================================================================
# STEP 4: Standalone Terminal Test Block
# =====================================================================
if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    model_path = root_dir / "models" / "xgboost_production_model.joblib"
    loaded_model = joblib.load(model_path)

    sample_fixed = {
        "initial_starch_concentration_g_L": 300.0,
        "initial_dextrin_avg_dp": 9.0,
        "glucoamylase_activity_U_g": 300.0,
        "initial_glucose_concentration_g_L": 5.0,
    }

    sample_economics = {
        "substrate_volume": 1000.0,
        "baseline_duration_h": 48.0,
        "baseline_energy_kwh": 120.0,
        "enzyme_price": 500.0,
        "water_price": 5.0,
        "energy_price": 110.0,
        "labour_cost": 25000.0,
        "glucose_price_per_kg": 1200.0,
    }

    output = optimize_batch_setpoints(loaded_model, sample_fixed, sample_economics)

    print("=== OPTIMAL SCADA SETPOINTS ===")
    for param, val in output["setpoints"].items():
        print(f"{param}: {val}")

    print(f"\nPredicted Glucose Yield: {output['predicted_yield_g_L']} g/L")
    print(f"Scaled Energy Used:      {output['economics']['scaled_energy_kwh']:.1f} kWh")
    print(f"Total Batch Cost:        ₦{output['economics']['total_cost']:,.2f}")
    print(f"Estimated Revenue:       ₦{output['economics']['revenue']:,.2f}")
    print(f"Maximum Net Profit:      ₦{output['economics']['net_profit']:,.2f}")

