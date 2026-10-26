import os
import numpy as np
import pandas as pd

def generate_base_features(num_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Generates independent baseline process features based on industrial 
    starch saccharification operational envelopes.
    """
    np.random.seed(seed)
    
    data = {
        "initial_starch_concentration_g_L": np.random.uniform(200, 400, num_samples),
        "initial_dextrin_avg_dp": np.random.uniform(3, 15, num_samples),           # Average Degree of Polymerization
        "saccharification_temp_C": np.random.normal(60, 5, num_samples),           # Industrial optima centered around 60C
        "pH": np.random.normal(4.5, 0.4, num_samples),                             # Optima window ~ 4.3-4.5
        "glucoamylase_activity_U_g": np.random.uniform(200, 400, num_samples),     # Warehouse assay activity
        "glucoamylase_dosage_g_kg": np.random.uniform(0.5, 2.0, num_samples),      # Mass added per substrate mass
        "saccharification_duration_h": np.random.uniform(24, 72, num_samples),     # Retention time window
        "water_to_substrate_ratio": np.random.uniform(2.0, 4.0, num_samples),      # Dilution metric
        "initial_glucose_concentration_g_L": np.random.exponential(5, num_samples),# Residual background sugars
        "mixing_rate_rpm": np.random.uniform(100, 300, num_samples)                # Mechanical agitation intensity
    }
    return pd.DataFrame(data)

def calculate_enzyme_efficiency(df: pd.DataFrame) -> np.ndarray:
    """
    Computes non-linear enzyme activity scaling factors based on 
    environmental and physical operational constraints (Bell curve pH,
    asymmetric thermal denaturation, and mixing boundaries).
    """
    # pH Profile: Symmetric Gaussian drop-off away from optimal 4.5
    ph_effect = np.exp(-0.5 * ((df["pH"] - 4.5) / 0.3) ** 2)
    
    # Temperature Profile: Asymmetric decay modeling sharp denaturation past 60°C
    temp_effect = np.where(
        df["saccharification_temp_C"] <= 60,
        np.exp(-0.5 * ((df["saccharification_temp_C"] - 60) / 6) ** 2),
        np.exp(-0.5 * ((df["saccharification_temp_C"] - 60) / 3) ** 2)  # High-temp degradation decay rate
    )
    
    # Mixing Profile: Exponential plateau modeling mass transfer boundary layer saturation
    mixing_effect = 0.7 + 0.3 * (1 - np.exp(-df["mixing_rate_rpm"] / 150))
    
    return ph_effect * temp_effect * mixing_effect

def calculate_reaction_kinetics(df: pd.DataFrame, efficiency: np.ndarray) -> np.ndarray:
    """
    Simulates time-course target yield using saturation kinetics, 
    accounting for substrate complexity (DP) and product inhibition.
    """
    # Compute active internal functional catalyst power
    effective_enzyme = (df["glucoamylase_dosage_g_kg"] * df["glucoamylase_activity_U_g"]) * efficiency
    
    # Hydrolysis speed penalization based on average dextrin chain lengths
    dp_factor = 1.0 / (1.0 + 0.05 * (df["initial_dextrin_avg_dp"] - 3))
    
    # Mass conservation wall (1.11g glucose maximum theoretical limit per 1g dry starch)
    max_hydrolysis_yield = df["initial_starch_concentration_g_L"] * 1.11
    
    # Saturation over duration
    k = 0.05 * effective_enzyme * dp_factor
    time_progression = 1 - np.exp(-k * df["saccharification_duration_h"] / 100)
    
    # Product feedback inhibition kinetics
    inhibition_factor = 1.0 / (1.0 + 0.001 * df["initial_glucose_concentration_g_L"])
    
    yield_predicted = max_hydrolysis_yield * time_progression * inhibition_factor
    
    # Strictly enforce thermodynamic boundaries
    return np.clip(yield_predicted, 0, max_hydrolysis_yield)

def add_measurement_noise(yield_array: np.ndarray, noise_level: float = 0.02, seed: int = 42) -> np.ndarray:
    """
    Applies proportional heteroscedastic Gaussian noise to simulate analytical sensor errors.
    """
    np.random.seed(seed)
    noise = np.random.normal(0, noise_level * yield_array)
    return np.maximum(0, yield_array + noise)

def pipeline_generate_dataset(num_samples: int = 1000, output_path: str = "saccharification_process_data.csv") -> pd.DataFrame:
    """
    Orchestrates the individual decoupled components of the generation pipeline
    and persists the observable dataframe to a CSV file.
    """
    # 1. Generate features representing verifiable plant measurements
    df = generate_base_features(num_samples)
    
    # 2. Compute hidden process interactions (Not outputted to preserve ML learning objectives)
    efficiency = calculate_enzyme_efficiency(df)
    theoretical_yield = calculate_reaction_kinetics(df, efficiency)
    
    # 3. Apply observation error to yield and build target label
    df["glucose_yield_g_L"] = add_measurement_noise(theoretical_yield)
    
    # 4. Serialize to disk
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created and saved to '{output_path}'. Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    dataset = pipeline_generate_dataset(num_samples=5000)
    
