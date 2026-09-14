"""
inference.py — AI Model Inference
WayFinder / IDR Navigation System

Provides functions to run the trained AI models on raw IMU data.
"""

import json
import torch
import numpy as np
from pathlib import Path

from src.preprocessing.alignment import estimate_alignment_matrix, apply_alignment
from src.ai_models.models import SpeedEstimator

METRICS_DIR = Path('/Users/srujangowda/Desktop/WayFinder/results/metrics')

_model = None
_norm_stats = None
_device = None

def load_ai_model():
    global _model, _norm_stats, _device
    if _model is not None:
        return

    norm_stats_path = METRICS_DIR / 'normalization_stats.json'
    model_path = METRICS_DIR / 'phase6_speed_model.pth'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_phase6.py first.")

    with open(norm_stats_path, 'r') as f:
        _norm_stats = json.load(f)

    _device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    _model = SpeedEstimator().to(_device)
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.eval()

def predict_speed_sequence(vdf, sdf):
    """
    Predict vehicle speed for an entire sequence using the trained CNN-GRU model.
    """
    load_ai_model()

    accel = sdf[['accel_x_ms2', 'accel_y_ms2', 'accel_z_ms2']].values
    gyro = sdf[['gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads']].values
    gt_speed = vdf['gps_velocity_ms'].values

    # Alignment
    R = estimate_alignment_matrix(accel, gt_speed)
    aligned_accel, aligned_gyro = apply_alignment(accel, gyro, R)
    features = np.hstack([aligned_accel, aligned_gyro])

    # Normalisation
    keys = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for i, k in enumerate(keys):
        if k in _norm_stats:
            features[:, i] = (features[:, i] - _norm_stats[k]['mean']) / _norm_stats[k]['std']

    # Inference (Sliding Window)
    window_size = 50
    n = len(features)
    pred_speed = np.zeros(n)
    
    if n <= window_size:
        return pred_speed

    X_batch = []
    indices = []

    for i in range(window_size, n):
        x_win = features[i - window_size: i]
        X_batch.append(x_win)
        indices.append(i - 1)  # predicting for the end of the window

        if len(X_batch) == 128 or i == n - 1:
            X_tensor = torch.tensor(np.array(X_batch, dtype=np.float32)).to(_device)
            with torch.no_grad():
                preds = _model(X_tensor).cpu().numpy().flatten()
            for idx, p in zip(indices, preds):
                pred_speed[idx] = max(0.0, float(p))
            X_batch = []
            indices = []

    # Fill initial window
    pred_speed[:window_size] = pred_speed[window_size]
    
    return pred_speed
