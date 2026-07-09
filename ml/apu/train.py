import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "apu" / "apu_synthetic.csv"
EXPORT_DIR = PROJECT_ROOT / "ml" / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "egt_steady_state_c",
    "startup_time_constant_s",
    "fuel_flow_kg_h",
    "turbine_inlet_temp_c",
    "rpm_pct",
    "time_to_90pct_s",
    "max_egt_rate_c_per_s",
    "egt_margin_c"
]

def train_apu():
    print("Loading synthetic APU dataset...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing APU CSV dataset at {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    
    X = df[FEATURE_NAMES].values
    y = df["health_class"].values
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler parameters and default values (means) to JSON
    scaler_params = {
        "feature_names": FEATURE_NAMES,
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "defaults": df[FEATURE_NAMES].mean().to_dict()
    }
    with open(EXPORT_DIR / "apu_scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    print("[OK] Saved scaler and default values to apu_scaler.json")
    
    # Train RandomForestClassifier
    print("Training Random Forest Classifier on APU dataset...")
    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=5,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Export to ONNX
    print("Exporting Random Forest model to ONNX...")
    initial_type = [("input", FloatTensorType([None, len(FEATURE_NAMES)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    
    onnx_path = EXPORT_DIR / "apu_model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[OK] Saved APU model to {onnx_path}")

if __name__ == "__main__":
    train_apu()
