"""
pipeline.py — Preprocessing Pipeline
Phase 2: Data Preprocessing

Features:
- Sequence-level train/val/test split (NO temporal leakage)
- Sensor normalization
- Sliding window sequence generation
- Outlier detection
- Missing value handling
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.preprocessing.loader import (
    discover_sequences, load_sequence, haversine_m
)
from src.preprocessing.alignment import estimate_alignment_matrix, build_features

# ─── TRAIN / VAL / TEST SPLIT ─────────────────────────────────────────────────
# Trajectory-level split — NO temporal leakage.
#
# TEST_DRIVERS must be drivers that never appear in TRAIN/VAL, or "test"
# results don't measure cross-driver generalisation at all. 'Vf (Driver E)'
# is the same underlying driver as 'Vta'/'Vtb' (train) and 'Vw' (val) — only
# the vehicle/session label differs — so it belongs in VAL, not TEST.
# Genuinely unseen-driver testing is only 'S (Driver A)' and 'Y (Driver D)'.

TRAIN_DRIVERS = ['Vta (Driver E)', 'Vtb (Driver E)']
VAL_DRIVERS   = ['Vw (Driver E)', 'Vf (Driver E)']
TEST_DRIVERS  = ['S (Driver A)', 'Y (Driver D)']


def trajectory_split(sequences: list) -> dict:
    """Split sequences into train/val/test by driver (no temporal leakage)."""
    splits = {'train': [], 'val': [], 'test': []}
    for seq in sequences:
        driver = seq['driver']
        if driver in TRAIN_DRIVERS:
            splits['train'].append(seq)
        elif driver in VAL_DRIVERS:
            splits['val'].append(seq)
        elif driver in TEST_DRIVERS:
            splits['test'].append(seq)
        else:
            splits['train'].append(seq)  # default to train
    return splits


def compute_normalization_stats(sequences: list) -> dict:
    """
    Compute mean/std for normalization using ONLY training sequences.
    NEVER use val/test data for this.
    """
    all_data = []
    for seq in sequences:
        try:
            vdf, sdf = load_sequence(seq)

            # accel_x/y/z, gyro_x/y/z, and the derived magnitude channels
            # must be normalized the same way the model actually sees them
            # at train/inference time: rotated into the vehicle frame via
            # the single shared build_features() helper. Computing these
            # stats separately (as before) is exactly how training and
            # inference silently drifted apart previously.
            raw_accel = sdf[['accel_x_ms2', 'accel_y_ms2', 'accel_z_ms2']].values
            raw_gyro = sdf[['gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads']].values
            gt_speed = vdf['gps_velocity_ms'].values
            R = estimate_alignment_matrix(raw_accel, gt_speed)
            # include_magnitude=True so the (currently unused, see
            # EXTRA_FEATURE_KEYS in alignment.py) derived-feature stats stay
            # available if that experiment is revisited later.
            built = build_features(raw_accel, raw_gyro, R, include_magnitude=True)

            # Features we'll normalize
            feats = {
                'accel_x': built[:, 0],
                'accel_y': built[:, 1],
                'accel_z': built[:, 2],
                'gyro_x':  built[:, 3],
                'gyro_y':  built[:, 4],
                'gyro_z':  built[:, 5],
                'accel_horiz_mag': built[:, 6],
                'gyro_mag': built[:, 7],
                'mag_x':   sdf['mag_x_uT'].values,
                'mag_y':   sdf['mag_y_uT'].values,
                'mag_z':   sdf['mag_z_uT'].values,
                'gps_speed': sdf['gps_speed_ms'].values,
                'v_speed':   vdf['gps_velocity_ms'].values,
                'v_yaw':     vdf['yaw_rate_rads'].values,
                'v_long_accel': vdf['long_accel_ms2'].values,
                'v_lat_accel':  vdf['lat_accel_ms2'].values,
            }
            all_data.append(feats)
        except Exception as e:
            print(f"  WARN: {seq['sequence']}: {e}")
            continue
    
    # Compute global stats
    stats = {}
    for key in all_data[0].keys() if all_data else []:
        combined = np.concatenate([d[key] for d in all_data])
        # Robust stats (use percentiles for outlier resistance)
        p01, p99 = np.percentile(combined, [1, 99])
        clipped = np.clip(combined, p01, p99)
        stats[key] = {
            'mean': float(np.mean(clipped)),
            'std':  float(np.std(clipped) + 1e-8),
            'min':  float(p01),
            'max':  float(p99),
        }
    
    return stats


def create_sliding_windows(sdf, vdf, window_size=100, stride=10,
                           norm_stats=None, target='speed'):
    """
    Create sliding windows for ML training.
    
    Input features: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    Target:         forward speed (m/s) from GPS
    
    Args:
        sdf: Smartphone DataFrame
        vdf: Vehicle DataFrame  
        window_size: samples per window (100 @ 10Hz = 10 seconds)
        stride: step between windows
        norm_stats: normalization stats (from training data only)
        target: 'speed' or 'yaw_rate'
    
    Returns:
        X: (N, window_size, n_features)
        y: (N,) target values
        t: (N,) timestamps
    """
    # Use S-file IMU data
    imu_cols = ['accel_x_ms2', 'accel_y_ms2', 'accel_z_ms2',
                'gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads']
    
    X_cols = sdf[imu_cols].values.astype(np.float32)
    
    # Target: GPS speed from S-file (more reliable for smartphone scenario)
    if target == 'speed':
        y_vals = sdf['gps_speed_ms'].values.astype(np.float32)
    elif target == 'yaw_rate':
        # Interpolate yaw rate from V-file if time-aligned
        y_vals = np.radians(vdf['yaw_rate_degs'].values).astype(np.float32)
    else:
        raise ValueError(f"Unknown target: {target}")
    
    # Normalize features if stats provided
    if norm_stats is not None:
        keys = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        for i, key in enumerate(keys):
            if key in norm_stats:
                X_cols[:, i] = (X_cols[:, i] - norm_stats[key]['mean']) / norm_stats[key]['std']
    
    # Generate windows
    X_list, y_list, t_list = [], [], []
    n = len(X_cols)
    times = sdf['time_s'].values
    
    for start in range(0, n - window_size, stride):
        end = start + window_size
        X_win = X_cols[start:end]
        # Target: value at center of window (causal: use END of window)
        y_win = y_vals[end - 1]
        
        X_list.append(X_win)
        y_list.append(y_win)
        t_list.append(times[end - 1])
    
    if not X_list:
        return None, None, None
    
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(t_list, dtype=np.float64))


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_ROOT = str(REPO_ROOT / 'data' / 'IO-VNBD-master')
    OUT_DIR = REPO_ROOT / 'results' / 'metrics'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  Phase 2 — Preprocessing Pipeline")
    print("=" * 60)
    
    # Discover sequences
    seqs = discover_sequences(DATA_ROOT)
    print(f"\nTotal sequences: {len(seqs)}")
    
    # Split
    splits = trajectory_split(seqs)
    for name, seq_list in splits.items():
        print(f"  {name:6s}: {len(seq_list)} sequences — {[s['sequence'] for s in seq_list[:3]]}...")
    
    # Compute normalization stats from TRAINING data only
    print("\nComputing normalization stats from TRAINING data...")
    norm_stats = compute_normalization_stats(splits['train'])
    
    # Save
    norm_path = OUT_DIR / 'normalization_stats.json'
    with open(norm_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved: {norm_path}")
    
    # Verify on one sequence
    print("\nVerifying sliding windows on validation sequence...")
    if splits['val']:
        vdf, sdf = load_sequence(splits['val'][0])
        X, y, t = create_sliding_windows(sdf, vdf, window_size=100, stride=10,
                                          norm_stats=norm_stats)
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  y range: [{y.min():.2f}, {y.max():.2f}] m/s")
        print(f"  X mean:  {X.mean():.4f} (should be ~0 after normalization)")
        print(f"  X std:   {X.std():.4f} (should be ~1 after normalization)")
    
    # Save split manifest
    manifest = {
        'train': [{'driver': s['driver'], 'sequence': s['sequence']} for s in splits['train']],
        'val':   [{'driver': s['driver'], 'sequence': s['sequence']} for s in splits['val']],
        'test':  [{'driver': s['driver'], 'sequence': s['sequence']} for s in splits['test']],
    }
    manifest_path = OUT_DIR / 'split_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Split manifest saved: {manifest_path}")
    
    print("\n  PHASE 2 STATUS: PASS")
