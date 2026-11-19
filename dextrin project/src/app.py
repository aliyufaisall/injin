import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
import traceback

# =====================================================================
# Custom feature engineering pipeline 
# =====================================================================
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

# 1. Environment and Path setup
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.optimizer import optimize_dextrin_setpoints


@st.cache_resource
def load_model():
    model_path = root_dir / "models" / "production_pipeline.joblib"
    return joblib.load(model_path)

model = load_model()

# 2. Main Title Layout
st.title("Dextrin Yield Predictor")

# 3. Create user input widgets inside a single Form with multi-column layout
with st.form("processing_form"):
    st.subheader("Input Processing Parameters")
    st.write("Adjust the parameters below to match your production batch characteristics.")
        # --- Financial and Batch Scaling Inputs ---
    st.markdown("### Production Economics and Costs")
    col9, col10 = st.columns(2)
    with col9:
        batch_weight = st.number_input("Cassava Batch Weight (kg)", min_value=1.0, value=1000.0, step=50.0)
        cassava_price = st.number_input("Cassava Purchase Price (₦/kg)", min_value=0.0, value=150.0, step=5.0)
        enzyme_price = st.number_input("Enzyme Price (₦/g or ₦/mL)", min_value=0.0, value=500.0, step=10.0)

        max_batch_budget = st.number_input(
            "Maximum Batch Budget (₦)",
            min_value=1000.0,
            value=250000.0,
            step=5000.0,
        )

    with col10:
        water_price = st.number_input("Water Cost (₦/L)", min_value=0.0, value=5.0, step=0.5)
        energy_used = st.number_input("Energy Consumed (kWh)", min_value=0.0, value=120.0, step=5.0)
        energy_price = st.number_input("Electricity / Fuel Price (₦/kWh)", min_value=0.0, value=110.0, step=5.0)
        labour_cost = st.number_input("Labour Cost per Batch (₦)", min_value=0.0, value=25000.0, step=500.0)


    # --- Raw Material Characteristics ---
    st.markdown("### Raw Material Properties")
    col1, col2 = st.columns(2)
    with col1:
        starch_content = st.slider("Cassava Starch Content (%)", 10.0, 45.0, 30.0, 0.1)
    with col2:
        moisture_content = st.slider("Cassava Moisture Content (%)", 40.0, 80.0, 65.0, 0.1)

    # --- Storage Conditions ---
    st.markdown("### Storage Conditions")
    col3, col4 = st.columns(2)
    with col3:
        storage_duration = st.slider("Raw Material Storage Duration (Days)", 0, 30, 3, 1)
        storage_temp = st.slider("Storage Temperature (°C)", 15.0, 45.0, 28.0, 0.5)
    with col4:
        storage_humidity = st.slider("Storage Humidity (%)", 30.0, 95.0, 75.0, 1.0)

    # --- Gelatinization & Cooking ---
    st.markdown("### Gelatinization and Cooking")
    col5, col6 = st.columns(2)
    with col5:
        cooking_temp = st.slider("Cooking Temperature (°C)", 50.0, 120.0, 85.0, 0.5)
        cooking_duration = st.slider("Cooking Duration (Min)", 5, 180, 45, 1)
    with col6:
        water_ratio = st.slider("Water-to-Cassava Ratio", 1.0, 10.0, 3.5, 0.1)

    # --- Enzymatic Treatment & Liquefaction ---
    st.markdown("### Enzymatic Treatment and Liquefaction")
    col7, col8 = st.columns(2)
    with col7:
        enzyme_activity = st.slider("Alpha-Amylase Activity (U/g)", 100.0, 5000.0, 1500.0, 10.0)
        liquefaction_temp = st.slider("Liquefaction Temperature (°C)", 50.0, 110.0, 90.0, 0.5)
        process_ph = st.slider("pH During Enzymatic Treatment", 3.0, 9.0, 6.0, 0.1)
    with col8:
        enzyme_dosage = st.slider("Alpha-Amylase Dosage (g/mL)", 0.001, 0.100, 0.015, 0.001, format="%.3f")
        liquefaction_duration = st.slider("Liquefaction Duration (Min)", 10, 240, 90, 1)

    # Submit button 
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        submitted = st.form_submit_button(
            "Calculate Yield Prediction",
            use_container_width=True,
        )

    with btn_col2:
        optimize_clicked = st.form_submit_button(
            "Auto-Optimize for Max Profit",
            type="primary",
            use_container_width=True,
        )


# 4. Construct DataFrame matching column criteria
features_dict = {
    "Cassava Starch Content (%)": [starch_content],
    "Cassava Moisture Content (%)": [moisture_content],
    "Raw Material Storage Duration (Days)": [storage_duration],
    "Storage Temperature (°C)": [storage_temp],
    "Storage Humidity (%)": [storage_humidity],
    "Cooking Temperature (°C)": [cooking_temp],
    "Water-to-Cassava Ratio": [water_ratio],
    "Cooking Duration (Min)": [cooking_duration],
    "Alpha-Amylase Activity (U/g)": [enzyme_activity],
    "Alpha-Amylase Dosage (g/mL)": [enzyme_dosage],
    "Liquefaction Temperature (°C)": [liquefaction_temp],
    "Liquefaction Duration (Min)": [liquefaction_duration],
    "pH During Enzymatic Treatment": [process_ph],
}

input_df = pd.DataFrame(features_dict)

# 5. Prediction Execution
if submitted:
    prediction = model.predict(input_df)
    
    st.success("Calculation Complete!")
    
    if hasattr(prediction, "__len__") and not isinstance(prediction, (str, bytes)):
        final_val = prediction[0]
    else:
        final_val = prediction
        
    st.metric(label="Estimated Cassava Yield", value=f"{final_val:.2f}")

        # --- Financial Accounting Computations ---
    # 1. Derive total water needed using the user's Water-to-Cassava Ratio slider
    total_water_liters = batch_weight * water_ratio
    
    # 2. Total cost aggregations
    cost_raw_material = batch_weight * cassava_price
    cost_enzymes = (enzyme_dosage * total_water_liters) * enzyme_price
    cost_water = total_water_liters * water_price
    cost_energy = energy_used * energy_price
    
    total_production_cost = cost_raw_material + cost_enzymes + cost_water + cost_energy + labour_cost
    
    
    # --- Display Financial Metrics ---
    st.markdown("---")
    st.subheader("Economic Efficiency Analysis")
    
    # Create 3 columns with the middle one holding your metric
    m_col1, m_col2, m_col3 = st.columns([1, 2, 1])

    with m_col2:
        st.metric(label="Total Batch Cost", value=f"₦{total_production_cost:,.2f}")

    with st.expander("View Cost Breakdown"):
        st.write(
            f"**Enzyme Cost:** ₦{cost_enzymes:,.2f}"
        )
        st.write(
            f"**Water Cost:** ₦{cost_water:,.2f}"
        )
        st.write(
            f"**Energy Cost:** ₦{cost_energy:,.2f}"
        )
        st.write(
            f"**Labour Cost:** ₦{labour_cost:,.2f}"
        )
if optimize_clicked:
    try:
        with st.spinner("Finding highest dextrin yield within your batch budget..."):
            # 1. Pack the 6 fixed raw material & storage inputs
            fixed_features = {
                "Cassava Starch Content (%)": starch_content,
                "Cassava Moisture Content (%)": moisture_content,
                "Raw Material Storage Duration (Days)": storage_duration,
                "Storage Temperature (°C)": storage_temp,
                "Storage Humidity (%)": storage_humidity,
                "Alpha-Amylase Activity (U/g)": enzyme_activity,
            }

            # 2. Pack the cost & budget inputs
            economics_inputs = {
                "batch_weight_kg": batch_weight,
                "cassava_price_per_kg": cassava_price,
                "enzyme_price": enzyme_price,
                "water_price": water_price,
                "baseline_process_min": cooking_duration + liquefaction_duration,
                "baseline_energy_kwh": energy_used,
                "energy_price": energy_price,
                "labour_cost": labour_cost,
                "max_batch_budget": max_batch_budget,
            }

            # 3. Call your optimizer
            opt_result = optimize_dextrin_setpoints(
                model=model,
                fixed_features=fixed_features,
                economics=economics_inputs,
            )

        # 4. Unpack results
        sp = opt_result["setpoints"]
        opt_yield = opt_result["predicted_yield_pct"]
        opt_econ = opt_result["economics"]

        if opt_econ["total_cost"] > max_batch_budget:
            st.warning(
                f"Note: Fixed raw material & labour costs already exceed ₦{max_batch_budget:,.2f}. "
                "Showing the lowest-cost recipe possible."
            )
        else:
            st.success("Optimization Complete! Maximum Yield Within Budget Found.")

        # Top KPI Metrics
        k1, k2, k3 = st.columns(3)
        k1.metric("Optimized Dextrin Yield", f"{opt_yield:.2f}%")
        k2.metric("Dextrin Output Mass", f"{opt_econ['dextrin_produced_kg']:,.1f} kg")
        k3.metric("Optimized Batch Cost", f"₦{opt_econ['total_cost']:,.2f}")

        # Side-by-Side Comparison Table
        st.subheader("Recommended Setpoints vs. Current Input")
        comparison_df = pd.DataFrame(
            {
                "Parameter": [
                    "Cooking Temperature (°C)",
                    "Water-to-Cassava Ratio",
                    "Cooking Duration (Min)",
                    "Alpha-Amylase Dosage (g/mL)",
                    "Liquefaction Temperature (°C)",
                    "Liquefaction Duration (Min)",
                    "pH During Enzymatic Treatment",
                ],
                "Your Input": [
                    cooking_temp,
                    water_ratio,
                    cooking_duration,
                    enzyme_dosage,
                    liquefaction_temp,
                    liquefaction_duration,
                    process_ph,
                ],
                "AI Recommended": [
                    sp["Cooking Temperature (°C)"],
                    sp["Water-to-Cassava Ratio"],
                    sp["Cooking Duration (Min)"],
                    sp["Alpha-Amylase Dosage (g/mL)"],
                    sp["Liquefaction Temperature (°C)"],
                    sp["Liquefaction Duration (Min)"],
                    sp["pH During Enzymatic Treatment"],
                ],
            }
        )
        st.table(comparison_df)

    except Exception as error:
        st.error(f"Optimization failed: {error}")
        st.code(traceback.format_exc())

        

