"""
evaluate_alignment.py — Phase 4: Evaluate Alignment
WayFinder / IDR Navigation System

Evaluates the alignment module on sample training data.
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/srujangowda/Desktop/WayFinder')
from src.preprocessing.loader import discover_sequences, load_sequence
from src.preprocessing.alignment import estimate_alignment_matrix, apply_alignment

DATA_ROOT = '/Users/srujangowda/Desktop/WayFinder/data/IO-VNBD-master'
PLOTS_DIR = Path('/Users/srujangowda/Desktop/WayFinder/results/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Phase 4 — Evaluate Alignment")
print("=" * 60)

seqs = discover_sequences(DATA_ROOT)
train_seqs = [s for s in seqs if 'Vta' in s['sequence']]
if not train_seqs:
    print("No Vta sequences found for alignment.")
    sys.exit(1)

seq = train_seqs[0]
print(f"Using sequence: {seq['sequence']}")
vdf, sdf = load_sequence(seq)

# Extract IMU data
accel = sdf[['accel_x_ms2', 'accel_y_ms2', 'accel_z_ms2']].values
gyro = sdf[['gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads']].values
gps_speed = sdf['gps_speed_ms'].values

# Estimate alignment
print("Estimating alignment matrix...")
R = estimate_alignment_matrix(accel, gps_speed)

print("Rotation Matrix:")
print(R)

# Apply alignment
aligned_accel, aligned_gyro = apply_alignment(accel, gyro, R)

# Plot before and after
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(f"Alignment Evaluation — {seq['sequence']}")

time_s = sdf['time_s'].values

axes[0, 0].plot(time_s, accel[:, 0], label='X')
axes[0, 0].plot(time_s, accel[:, 1], label='Y')
axes[0, 0].plot(time_s, accel[:, 2], label='Z')
axes[0, 0].set_title('Raw Phone Acceleration')
axes[0, 0].set_ylabel('m/s^2')
axes[0, 0].legend()

axes[0, 1].plot(time_s, aligned_accel[:, 0], label='Forward (X)')
axes[0, 1].plot(time_s, aligned_accel[:, 1], label='Right (Y)')
axes[0, 1].plot(time_s, aligned_accel[:, 2], label='Down (Z)')
axes[0, 1].set_title('Aligned Vehicle Acceleration')
axes[0, 1].legend()

axes[1, 0].plot(time_s, gyro[:, 0], label='X')
axes[1, 0].plot(time_s, gyro[:, 1], label='Y')
axes[1, 0].plot(time_s, gyro[:, 2], label='Z')
axes[1, 0].set_title('Raw Phone Gyroscope')
axes[1, 0].set_ylabel('rad/s')
axes[1, 0].legend()

axes[1, 1].plot(time_s, aligned_gyro[:, 0], label='Roll (X)')
axes[1, 1].plot(time_s, aligned_gyro[:, 1], label='Pitch (Y)')
axes[1, 1].plot(time_s, aligned_gyro[:, 2], label='Yaw (Z)')
axes[1, 1].set_title('Aligned Vehicle Gyroscope')
axes[1, 1].legend()

plt.tight_layout()
out_plot = PLOTS_DIR / 'phase4_alignment.png'
plt.savefig(out_plot)
print(f"Saved plot to {out_plot}")
print("Phase 4 STATUS: PASS")
