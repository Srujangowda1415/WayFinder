"""
dataset.py — PyTorch Dataset for IO-VNBD
WayFinder / IDR Navigation System

Generates sliding windows of IMU data and corresponding targets.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.preprocessing.loader import load_sequence
from src.preprocessing.alignment import estimate_alignment_matrix, build_features, FEATURE_KEYS

class IMUDataset(Dataset):
    def __init__(self, sequences, window_size=100, stride=10, norm_stats=None, target_type='speed'):
        """
        Args:
            sequences: List of sequence dicts.
            window_size: Number of samples in a window (100 = 10s @ 10Hz).
            stride: Step size between windows.
            norm_stats: Normalization stats.
            target_type: 'speed' (regression), 'motion' (binary classification)
        """
        self.window_size = window_size
        self.stride = stride
        self.target_type = target_type
        
        self.X = []
        self.y = []
        
        for seq in sequences:
            try:
                vdf, sdf = load_sequence(seq)
                
                # We align IMU to vehicle frame
                accel = sdf[['accel_x_ms2', 'accel_y_ms2', 'accel_z_ms2']].values
                gyro = sdf[['gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads']].values
                
                # Ground truth speed comes from V-file (ECU)
                # However, S-file and V-file are synchronized and have same length.
                # Let's use V-file gps_velocity_ms
                gt_speed = vdf['gps_velocity_ms'].values
                
                # Estimate alignment
                R = estimate_alignment_matrix(accel, gt_speed)
                features = build_features(accel, gyro, R)

                if norm_stats is not None:
                    for i, k in enumerate(FEATURE_KEYS):
                        if k in norm_stats:
                            features[:, i] = (features[:, i] - norm_stats[k]['mean']) / norm_stats[k]['std']
                
                n = len(features)
                for start in range(0, n - window_size, stride):
                    end = start + window_size
                    x_win = features[start:end]
                    
                    if target_type == 'speed':
                        # Predict speed at the END of the window (causal)
                        y_val = gt_speed[end - 1]
                    elif target_type == 'motion':
                        # Predict if moving (> 0.5 m/s) at the END of the window
                        y_val = 1.0 if gt_speed[end - 1] > 0.5 else 0.0
                        
                    self.X.append(x_win)
                    self.y.append(y_val)
                    
            except Exception as e:
                print(f"Skipping sequence {seq['sequence']} due to error: {e}")
                
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        # PyTorch expects input shape (channels, length) for 1D CNNs
        # but we'll return (length, channels) and let the model handle it
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def _inverse_frequency_sample_weights(y: np.ndarray, bin_edges) -> np.ndarray:
    """
    Per-sample weight = 1 / (frequency of that sample's speed bin).

    Training windows are heavily imbalanced toward cruising speed (in this
    dataset, ~47% of windows fall in [10,20) m/s while <10% are near-
    stationary). An unweighted loss lets the majority bin dominate,
    producing "regression to the mean": low speeds get systematically
    over-predicted and high speeds under-predicted — exactly the failure
    mode that matters most for drift, since a slow/stationary window with
    an inflated speed prediction integrates directly into position error.
    """
    bin_idx = np.digitize(y, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
    counts = np.bincount(bin_idx, minlength=len(bin_edges) - 1)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts[bin_idx]
    return weights


def get_dataloaders(manifest_path, norm_stats_path, target_type='speed', batch_size=64,
                     balance_speed_bins=True):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    from src.preprocessing.loader import discover_sequences
    all_seqs = discover_sequences(str(REPO_ROOT / 'data' / 'IO-VNBD-master'))
    seq_map = {s['sequence']: s for s in all_seqs}

    train_seqs = [seq_map[s['sequence']] for s in manifest['train'] if s['sequence'] in seq_map]
    val_seqs = [seq_map[s['sequence']] for s in manifest['val'] if s['sequence'] in seq_map]

    train_dataset = IMUDataset(train_seqs, window_size=50, stride=10, norm_stats=norm_stats, target_type=target_type)
    val_dataset = IMUDataset(val_seqs, window_size=50, stride=50, norm_stats=norm_stats, target_type=target_type)

    if balance_speed_bins and target_type == 'speed' and len(train_dataset) > 0:
        bin_edges = [0, 0.5, 2, 5, 10, 20, 1000]
        weights = _inverse_frequency_sample_weights(train_dataset.y, bin_edges)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
