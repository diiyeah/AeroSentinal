import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "landing_gear" / "landing_gear_synthetic.csv"
EXPORT_DIR = PROJECT_ROOT / "ml" / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Features we want to use
FEATURE_NAMES = [
    "aircraft_mass_kg",
    "touchdown_speed_kts",
    "brake_pressure_psi",
    "runway_friction_coeff",
    "braking_fraction",
    "brake_energy_mj",
    "peak_brake_temp_c",
    "temp_rise_c",
    "cumulative_cycles",
    "wear_index"
]

def train_landing_gear():
    print("Loading synthetic landing gear dataset...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing Landing Gear CSV dataset at {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    
    X = df[FEATURE_NAMES].values
    y = df["severity_class"].values
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler parameters and default values (means) to JSON
    scaler_params = {
        "feature_names": FEATURE_NAMES,
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        # Store default feature vector to simplify partial telemetry requests in API
        "defaults": df[FEATURE_NAMES].mean().to_dict()
    }
    with open(EXPORT_DIR / "landing_gear_scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    print("[OK] Saved scaler and default values to landing_gear_scaler.json")
    
    # Train XGBClassifier
    print("Training XGBoost Classifier on Landing Gear dataset...")
    model = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_scaled, y)
    
    # Export to ONNX
    print("Exporting XGBoost model to ONNX...")
    initial_type = [("input", FloatTensorType([None, len(FEATURE_NAMES)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=12)
    
    onnx_path = EXPORT_DIR / "landing_gear_model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[OK] Saved Landing Gear model to {onnx_path}")

if __name__ == "__main__":
    train_landing_gear()
