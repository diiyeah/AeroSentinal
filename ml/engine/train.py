import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "cmapss"
EXPORT_DIR = PROJECT_ROOT / "ml" / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Select the 14 standard sensors
SENSOR_COLS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
SENSOR_NAMES = [f"s_{i}" for i in SENSOR_COLS]

class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        # Bidirectional LSTM has hidden_dim * 2 output
        self.attention_w = nn.Parameter(torch.randn(hidden_dim * 2, 1))
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x) # shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Simple self-attention over sequence dimension
        # score = tanh(lstm_out * W)
        attn_scores = torch.matmul(lstm_out, self.attention_w) # shape: (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) # shape: (batch_size, seq_len, 1)
        
        # Weighted context vector
        context = torch.sum(lstm_out * attn_weights, dim=1) # shape: (batch_size, hidden_dim * 2)
        
        out = self.fc(context) # shape: (batch_size, output_dim)
        return out.squeeze(-1)

def load_cmapss_data():
    # Load train data
    train_path = DATA_DIR / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing CMAPSS train file at {train_path}")
        
    col_names = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s_{i}" for i in range(1, 22)]
    df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    
    # Calculate RUL
    # Piecewise linear RUL target: cap maximum RUL at 125
    max_cycles = df.groupby("unit")["cycle"].max().reset_index()
    max_cycles.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycles, on="unit")
    df["RUL_raw"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL_raw"].clip(upper=125)
    return df

def preprocess_and_train():
    print("Loading C-MAPSS FD001 dataset...")
    df = load_cmapss_data()
    
    # Extract sensors
    X_raw = df[SENSOR_NAMES].values
    y = df["RUL"].values
    
    # Calculate min-max parameters for scaling
    mins = X_raw.min(axis=0)
    maxs = X_raw.max(axis=0)
    # Avoid division by zero
    maxs = np.where(maxs == mins, maxs + 1e-5, maxs)
    
    # Save scaler parameters to JSON
    scaler_params = {
        "sensor_cols": SENSOR_COLS,
        "sensor_names": SENSOR_NAMES,
        "mins": mins.tolist(),
        "maxs": maxs.tolist()
    }
    with open(EXPORT_DIR / "engine_scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    print("[OK] Saved scaling parameters to engine_scaler.json")
    
    # Normalize features
    X_scaled = (X_raw - mins) / (maxs - mins)
    
    # Create sliding windows
    seq_len = 30
    features = []
    targets = []
    
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit]
        unit_X = X_scaled[unit_df.index]
        unit_y = y[unit_df.index]
        
        for i in range(len(unit_df) - seq_len + 1):
            features.append(unit_X[i:i+seq_len])
            targets.append(unit_y[i+seq_len-1])
            
    X_seq = np.array(features, dtype=np.float32)
    y_seq = np.array(targets, dtype=np.float32)
    
    # Train PyTorch model
    print(f"Prepared {X_seq.shape[0]} training sequences. Training model...")
    model = BiLSTMAttention(input_dim=len(SENSOR_COLS))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_seq)
    y_tensor = torch.tensor(y_seq)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    epochs = 4
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(loader):.4f}")
        
    # Export to ONNX
    print("Exporting trained model to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, seq_len, len(SENSOR_COLS))
    onnx_path = EXPORT_DIR / "engine_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12
    )
    print(f"[OK] Saved Engine model to {onnx_path}")

if __name__ == "__main__":
    preprocess_and_train()
