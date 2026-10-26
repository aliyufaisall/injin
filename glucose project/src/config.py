import os

# Root & Data Paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # Points to project root

RAW_DATA_PATH = os.path.join(
    BASE_DIR, "data", "saccharification_process_data.csv"
)

# Phase 1: Target Definition and Global Reproducibility Hyperparameters
TARGET_COLUMN = "glucose_yield_g_L"
RANDOM_SEED = 42

# Column Verification List
EXPECTED_FEATURES = [
    "initial_starch_concentration_g_L",
    "initial_dextrin_avg_dp",
    "saccharification_temp_C",
    "pH",
    "glucoamylase_activity_U_g",
    "glucoamylase_dosage_g_kg",
    "saccharification_duration_h",
    "water_to_substrate_ratio",
    "initial_glucose_concentration_g_L",
    "mixing_rate_rpm",
]
