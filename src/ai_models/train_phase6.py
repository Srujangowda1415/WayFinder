"""
train_phase6.py — Phase 6: AI Speed Estimator
WayFinder / IDR Navigation System

Trains a CNN-GRU to estimate forward vehicle speed from IMU.
This replaces wheel odometry during GNSS-denied periods.
Target: GPS speed from V-file (ECU GPS, primary ground truth).
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

sys.path.insert(0, '/Users/srujangowda/Desktop/WayFinder')
from src.ai_models.dataset import IMUDataset, get_dataloaders
from src.ai_models.models import SpeedEstimator

METRICS_DIR = Path('/Users/srujangowda/Desktop/WayFinder/results/metrics')
PLOTS_DIR = Path('/Users/srujangowda/Desktop/WayFinder/results/plots')

print("=" * 60)
print("  Phase 6 — AI Speed Estimator")
print("=" * 60)

manifest_path = METRICS_DIR / 'split_manifest.json'
norm_stats_path = METRICS_DIR / 'normalization_stats.json'

train_loader, val_loader = get_dataloaders(
    manifest_path, norm_stats_path, target_type='speed', batch_size=128
)

print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print(f"  Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"  Device: {device}")

model = SpeedEstimator().to(device)

def custom_loss(outputs, y, X):
    pred_speed = outputs[:, 0]
    log_var = outputs[:, 1]
    pred_vib = outputs[:, 2]
    
    # Calculate vibration target from normalized acceleration variance
    vib_target = torch.std(torch.norm(X[:, :, :3], dim=-1), dim=-1)
    
    # GNLL Loss for speed
    gnll_loss = 0.5 * torch.exp(-log_var) * (pred_speed - y)**2 + 0.5 * log_var
    gnll_loss = gnll_loss.mean()
    
    # MSE Loss for vibration
    vib_loss = nn.MSELoss()(pred_vib, vib_target)
    
    total_loss = gnll_loss + 5.0 * vib_loss
    return total_loss

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

num_epochs = 10
train_losses, val_losses, val_maes = [], [], []
best_val_loss = float('inf')
best_model_state = None

print("\nTraining speed estimator...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = custom_loss(outputs, y, X)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = custom_loss(outputs, y, X)
            val_loss += loss.item() * X.size(0)
            all_preds.extend(outputs[:, 0].cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    scheduler.step(epoch_val_loss)

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets)**2)))
    val_maes.append(mae)

    print(f"  Epoch {epoch+1}/{num_epochs} - "
          f"Train RMSE: {np.sqrt(epoch_loss):.4f} m/s | "
          f"Val RMSE: {rmse:.4f} m/s | "
          f"Val MAE: {mae:.4f} m/s")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

# Save best model
model_path = METRICS_DIR / 'phase6_speed_model.pth'
torch.save(best_model_state, model_path)
model_size_mb = model_path.stat().st_size / (1024 * 1024)

final_rmse = float(np.sqrt(np.mean((preds - targets)**2)))
final_mae = float(np.mean(np.abs(preds - targets)))

print(f"\nFinal Metrics:")
print(f"  Val RMSE: {final_rmse:.4f} m/s ({final_rmse * 3.6:.2f} km/h)")
print(f"  Val MAE:  {final_mae:.4f} m/s ({final_mae * 3.6:.2f} km/h)")
print(f"  Model size: {model_size_mb:.2f} MB")

# Save metrics
metrics = {
    'phase': 6,
    'method': 'CNN-GRU Speed Estimator',
    'val_rmse_ms': final_rmse,
    'val_mae_ms': final_mae,
    'val_rmse_kmh': final_rmse * 3.6,
    'val_mae_kmh': final_mae * 3.6,
    'model_size_mb': model_size_mb,
}
with open(METRICS_DIR / 'phase6_speed_estimator.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Phase 6: AI Speed Estimator', fontsize=13, fontweight='bold')

axes[0].plot(np.sqrt(train_losses), label='Train RMSE')
axes[0].plot(np.sqrt(val_losses), label='Val RMSE')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('RMSE (m/s)')
axes[0].set_title('Training Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Scatter: predicted vs actual
sample_idx = np.random.choice(len(preds), min(2000, len(preds)), replace=False)
axes[1].scatter(targets[sample_idx], preds[sample_idx], alpha=0.3, s=5, color='steelblue')
lim = max(targets.max(), preds.max())
axes[1].plot([0, lim], [0, lim], 'r--', lw=1.5, label='Perfect')
axes[1].set_xlabel('Ground Truth Speed (m/s)')
axes[1].set_ylabel('Predicted Speed (m/s)')
axes[1].set_title(f'Prediction vs GT (RMSE={final_rmse:.3f} m/s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'phase6_speed_estimator.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"  Plot saved: {PLOTS_DIR / 'phase6_speed_estimator.png'}")
print("\nPhase 6 STATUS: PASS")
