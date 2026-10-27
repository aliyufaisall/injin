import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =====================================================================
# 1. Environment and Path Setup
# =====================================================================
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


@st.cache_resource
def load_model():
    model_path = root_dir / "models" /"xgboost_production_model.joblib"
    return joblib.load(model_path)


model = load_model()

# =====================================================================
# 2. Main Title Layout
# =====================================================================
st.title("Glucose Yield Predictor")

st.write(
    "Estimate glucose yield from starch saccharification "
    "process parameters and calculate batch production cost."
)

# =====================================================================
# 3. User Input Form
# =====================================================================
with st.form("saccharification_form"):

    st.subheader("Input Processing Parameters")

    st.write(
        "Adjust the parameters below to match your "
        "saccharification batch conditions."
    )

    # --- Production Economics and Costs ---
    st.markdown("### Production Economics and Costs")

    col9, col10 = st.columns(2)

    with col9:
        substrate_volume = st.number_input(
            "Substrate Volume (L)",
            min_value=1.0,
            value=1000.0,
            step=50.0,
        )

        enzyme_price = st.number_input(
            "Enzyme Price (₦/g)",
            min_value=0.0,
            value=500.0,
            step=10.0,
        )

        water_price = st.number_input(
            "Water Cost (₦/L)",
            min_value=0.0,
            value=5.0,
            step=0.5,
        )

    with col10:
        energy_used = st.number_input(
            "Energy Consumed (kWh)",
            min_value=0.0,
            value=120.0,
            step=5.0,
        )

        energy_price = st.number_input(
            "Electricity / Fuel Price (₦/kWh)",
            min_value=0.0,
            value=110.0,
            step=5.0,
        )

        labour_cost = st.number_input(
            "Labour Cost per Batch (₦)",
            min_value=0.0,
            value=25000.0,
            step=500.0,
        )

    # --- Initial Substrate Characteristics ---
    st.markdown("### Initial Substrate Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        initial_starch_concentration = st.number_input(
            "Initial Starch Concentration (g/L)",
            min_value=1.0,
            value=300.0,
            step=10.0,
        )

    with col2:
        initial_dextrin_avg_dp = st.number_input(
            "Initial Dextrin Average DP",
            min_value=1.0,
            value=9.0,
            step=1.0,
        )

    # --- Saccharification Conditions ---
    st.markdown("### Saccharification Conditions")

    col3, col4 = st.columns(2)

    with col3:
        saccharification_temp = st.number_input(
            "Saccharification Temperature (°C)",
            min_value=0.0,
            value=60.0,
            step=0.5,
        )

        process_ph = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=4.5,
            step=0.1,
        )

    with col4:
        saccharification_duration = st.number_input(
            "Saccharification Duration (Hours)",
            min_value=0.0,
            value=48.0,
            step=1.0,
        )

        water_to_substrate_ratio = st.number_input(
            "Water-to-Substrate Ratio",
            min_value=0.1,
            value=3.0,
            step=0.1,
        )

    # --- Enzymatic Treatment ---
    st.markdown("### Enzymatic Treatment")

    col5, col6 = st.columns(2)

    with col5:
        glucoamylase_activity = st.number_input(
            "Glucoamylase Activity (U/g)",
            min_value=1.0,
            value=300.0,
            step=10.0,
        )

    with col6:
        glucoamylase_dosage = st.number_input(
            "Glucoamylase Dosage (g/kg)",
            min_value=0.01,
            value=1.0,
            step=0.1,
        )

    # --- Background Glucose and Mixing ---
    st.markdown("### Background Glucose and Mixing")

    col7, col8 = st.columns(2)

    with col7:
        initial_glucose_concentration = st.number_input(
            "Initial Glucose Concentration (g/L)",
            min_value=0.0,
            value=5.0,
            step=1.0,
        )

    with col8:
        mixing_rate = st.number_input(
            "Mixing Rate (RPM)",
            min_value=0.0,
            value=200.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Predict Glucose Yield",
        type="primary",
        use_container_width=True,
    )

# =====================================================================
# 4. Construct Input DataFrame
# =====================================================================
features_dict = {
    "initial_starch_concentration_g_L": [
        initial_starch_concentration
    ],
    "initial_dextrin_avg_dp": [
        initial_dextrin_avg_dp
    ],
    "saccharification_temp_C": [
        saccharification_temp
    ],
    "pH": [process_ph],
    "glucoamylase_activity_U_g": [
        glucoamylase_activity
    ],
    "glucoamylase_dosage_g_kg": [
        glucoamylase_dosage
    ],
    "saccharification_duration_h": [
        saccharification_duration
    ],
    "water_to_substrate_ratio": [
        water_to_substrate_ratio
    ],
    "initial_glucose_concentration_g_L": [
        initial_glucose_concentration
    ],
    "mixing_rate_rpm": [mixing_rate],
}

input_df = pd.DataFrame(features_dict)

# =====================================================================
# 5. Prediction and Production Cost Calculation
# =====================================================================
if submitted:

    try:
        prediction = model.predict(input_df)
        final_value = float(prediction[0])

        # --- Estimate Material Requirements ---
        total_water_liters = substrate_volume * water_to_substrate_ratio

        # Convert starch concentration to kg for the specified volume.
        starch_mass_kg = (
            initial_starch_concentration * substrate_volume / 1000
        )

        # Enzyme dosage is expressed in g per kg of substrate.
        enzyme_mass_g = glucoamylase_dosage * starch_mass_kg

        # --- Calculate Individual Costs ---
        cost_enzymes = enzyme_mass_g * enzyme_price
        cost_water = total_water_liters * water_price
        cost_energy = energy_used * energy_price

        total_production_cost = (
            cost_enzymes
            + cost_water
            + cost_energy
            + labour_cost
        )

        # --- Display Glucose Yield ---
        st.success("Prediction Complete!")

        st.metric(
            label="Predicted Glucose Yield",
            value=f"{final_value:.2f} g/L",
        )

        # --- Display Production Cost ---
        st.markdown("---")
        st.subheader("Saccharification Batch Cost")

        st.metric(
            label="Total Batch Production Cost",
            value=f"₦{total_production_cost:,.2f}",
        )

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
            st.write(
                f"**Estimated Water Requirement:** "
                f"{total_water_liters:,.2f} L"
            )
            st.write(
                f"**Estimated Enzyme Requirement:** "
                f"{enzyme_mass_g:,.2f} g"
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
    
