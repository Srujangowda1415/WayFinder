"""
test_inference.py — Phase A: ML Inference Verification
WayFinder / IDR Navigation System

Verifies that the trained PyTorch model can correctly predict speed
from raw IMU sequences by applying the exact training preprocessing.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/srujangowda/Desktop/WayFinder')
from src.preprocessing.loader import discover_sequences, load_sequence
from src.preprocessing.alignment import estimate_alignment_matrix, apply_alignment
from src.ai_models.models import SpeedEstimator

METRICS_DIR = Path('/Users/srujangowda/Desktop/WayFinder/results/metrics')
PLOTS_DIR = Path('/Users/srujangowda/Desktop/WayFinder/results/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Phase A — ML Inference Verification")
print("=" * 60)

# Load stats and model
norm_stats_path = METRICS_DIR / 'normalization_stats.json'
model_path = METRICS_DIR / 'phase6_speed_model.pth'

if not model_path.exists():
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

with open(norm_stats_path, 'r') as f:
    norm_stats = json.load(f)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Loading model on {device}...")

model = SpeedEstimator().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Select test sequence S1
seqs = discover_sequences('/Users/srujangowda/Desktop/WayFinder/data/IO-VNBD-master')
s1_seq = next((s for s in seqs if s['sequence'] == 'S1'), None)
if s1_seq is None:
    print("Test sequence S1 not found.")
    sys.exit(1)

print(f"\nProcessing Sequence: {s1_seq['sequence']}")
vdf, sdf = load_sequence(s1_seq)

times = vdf['time_s'].values
accel = sdf[['accel_x_ms2', 'accel_y_ms2', 'accel_z_ms2']].values
gyro = sdf[['gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads']].values
gt_speed = vdf['gps_velocity_ms'].values

# 1. Alignment
print("Estimating alignment matrix...")
R = estimate_alignment_matrix(accel, gt_speed)
aligned_accel, aligned_gyro = apply_alignment(accel, gyro, R)
features = np.hstack([aligned_accel, aligned_gyro])

# 2. Normalisation
print("Normalising features...")
keys = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
for i, k in enumerate(keys):
    if k in norm_stats:
        features[:, i] = (features[:, i] - norm_stats[k]['mean']) / norm_stats[k]['std']

# 3. Inference (Sliding Window)
window_size = 50
n = len(features)
pred_speed = np.zeros(n)

print(f"Running inference (window={window_size})...")
X_batch = []
indices = []

for i in range(window_size, n):
    x_win = features[i - window_size: i]
    X_batch.append(x_win)
    indices.append(i - 1)  # the prediction is for the END of the window

    if len(X_batch) == 128 or i == n - 1:
        X_tensor = torch.tensor(np.array(X_batch, dtype=np.float32)).to(device)
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy().flatten()
        for idx, p in zip(indices, preds):
            pred_speed[idx] = max(0.0, float(p))  # speed cannot be negative
        X_batch = []
        indices = []

# Fill initial window with 0s or first prediction
pred_speed[:window_size] = pred_speed[window_size]

# Log snippet (first 10 valid predictions)
print("\n[LOG] Verification of Inference Inputs/Outputs:")
for i in range(window_size, window_size + 10):
    print(f"Time: {times[i]:.2f}s | "
          f"Acc[0]: {features[i,:3].round(2)} | "
          f"Gyr[0]: {features[i,3:].round(2)} | "
          f"Pred Speed: {pred_speed[i]:.2f} m/s | "
          f"GT Speed: {gt_speed[i]:.2f} m/s")

# Plot
print("\nGenerating inference plot...")
plt.figure(figsize=(10, 5))
plt.plot(times, gt_speed, 'b-', lw=1.5, label='GT GPS Speed', alpha=0.8)
plt.plot(times, pred_speed, 'r--', lw=1.5, label='ML Pred Speed', alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title(f'ML Inference Verification — {s1_seq["sequence"]}')
plt.legend()
plt.grid(True, alpha=0.3)
plot_path = PLOTS_DIR / 'phaseA_inference_verification.png'
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()

rmse = np.sqrt(np.mean((pred_speed[window_size:] - gt_speed[window_size:])**2))
print(f"\nInference RMSE: {rmse:.3f} m/s")
print(f"Plot saved: {plot_path}")
print("Phase A STATUS: COMPLETE")
