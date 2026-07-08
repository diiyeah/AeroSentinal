import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "hydraulics"
EXPORT_DIR = PROJECT_ROOT / "ml" / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

class ConvAutoencoder1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Input shape: (batch_size, 1, 60)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2), # -> (batch, 16, 30)
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=5, stride=2, padding=2), # -> (batch, 8, 15)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=5, stride=2, padding=2, output_padding=1), # -> (batch, 16, 30)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1), # -> (batch, 1, 60)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_hydraulics():
    print("Loading UCI Hydraulics dataset...")
    ps1_path = DATA_DIR / "PS1.txt"
    profile_path = DATA_DIR / "profile.txt"
    
    if not ps1_path.exists() or not profile_path.exists():
        raise FileNotFoundError("Missing UCI Hydraulics PS1.txt or profile.txt files.")
        
    # Read profile for labels
    # Col 0: Cooler, Col 1: Valve, Col 2: Pump leak, Col 3: Accumulator
    profile = pd.read_csv(profile_path, sep=r"\s+", header=None)
    
    # Read PS1 sensor (sampled at 100 Hz, 60 seconds per cycle = 6000 values)
    print("Reading PS1 telemetry...")
    ps1 = pd.read_csv(ps1_path, sep=r"\s+", header=None)
    
    # Downsample from 6000 to 60 values (average of every 100 readings)
    print("Downsampling PS1 to 1Hz (60 values per cycle)...")
    ps1_downsampled = np.zeros((ps1.shape[0], 60), dtype=np.float32)
    for i in range(60):
        ps1_downsampled[:, i] = ps1.iloc[:, i*100:(i+1)*100].mean(axis=1)
        
    # Identify nominal (healthy) cycles
    # Cooler = 100 (optimal), Valve = 100 (optimal), Leak = 0 (no leak), Accumulator = 130 (optimal)
    nominal_idx = profile[
        (profile[0] == 100) & 
        (profile[1] == 100) & 
        (profile[2] == 0) & 
        (profile[3] == 130)
    ].index.tolist()
    
    print(f"Found {len(nominal_idx)} nominal cycles out of {ps1.shape[0]} total cycles.")
    
    # Min-max scaling
    mins = ps1_downsampled.min()
    maxs = ps1_downsampled.max()
    ps1_scaled = (ps1_downsampled - mins) / (maxs - mins)
    
    # Split nominal data for autoencoder training
    nominal_data = ps1_scaled[nominal_idx]
    
    # Train model
    print("Training 1D Conv Autoencoder on nominal data...")
    model = ConvAutoencoder1D()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Format data for Conv1D: (samples, channels=1, length=60)
    X_tensor = torch.tensor(nominal_data[:, np.newaxis, :], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    model.train()
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(loader):.6f}")
        
    # Pick a baseline healthy and baseline faulty sequence to store in JSON
    # This helps the API simulate a realistic 60-second cycle reconstruction
    healthy_baseline = nominal_data[0].tolist()
    
    # Find a highly degraded cycle (e.g. accumulator close to failure, severe pump leak)
    faulty_idx = profile[
        (profile[2] == 2) | (profile[3] == 90)
    ].index.tolist()
    faulty_baseline = ps1_scaled[faulty_idx[0]].tolist() if faulty_idx else ps1_scaled[-1].tolist()
    
    # Save scaler and baselines
    scaler_params = {
        "mins": float(mins),
        "maxs": float(maxs),
        "healthy_baseline": healthy_baseline,
        "faulty_baseline": faulty_baseline
    }
    with open(EXPORT_DIR / "hydraulics_scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    print("[OK] Saved scaler and baseline cycles to hydraulics_scaler.json")
    
    # Export to ONNX
    print("Exporting model to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 1, 60)
    onnx_path = EXPORT_DIR / "hydraulics_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12
    )
    print(f"[OK] Saved Hydraulics model to {onnx_path}")

if __name__ == "__main__":
    train_hydraulics()
