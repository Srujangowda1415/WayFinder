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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.ai_models.dataset import IMUDataset, get_dataloaders
from src.ai_models.models import SpeedEstimator

METRICS_DIR = REPO_ROOT / 'results' / 'metrics'
PLOTS_DIR = REPO_ROOT / 'results' / 'plots'

# Fixed seed: without it, runs of this *exact same* config produced val RMSE
# and downstream drift that varied more than the differences between actual
# architecture changes being compared — making experiment-to-experiment
# comparisons unreliable. Not a substitute for evaluating on more than one
# seed before trusting a conclusion, but removes one axis of noise.
torch.manual_seed(42)
np.random.seed(42)

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
    # Clamp log_var: unclamped, a GNLL loss can be minimized by inflating
    # variance instead of improving the point estimate (the quadratic term
    # shrinks faster than the linear penalty grows whenever the error is
    # large, which it always is early in training) — the model can "cheat"
    # instead of learning. Clamping bounds how much it can lean on that.
    log_var = torch.clamp(outputs[:, 1], min=-3.0, max=3.0)
    pred_vib = outputs[:, 2]

    # Calculate vibration target from normalized acceleration variance
    vib_target = torch.std(torch.norm(X[:, :, :3], dim=-1), dim=-1)

    # GNLL Loss for speed
    gnll_loss = 0.5 * torch.exp(-log_var) * (pred_speed - y)**2 + 0.5 * log_var
    gnll_loss = gnll_loss.mean()

    # MSE Loss for vibration — this is an auxiliary head the mobile app
    # doesn't currently rely on for navigation; it must not compete with the
    # primary speed task. (It previously had a 5.0x weight, which could
    # dominate the gradient signal for the head that actually matters.)
    vib_loss = nn.MSELoss()(pred_vib, vib_target)

    total_loss = gnll_loss + 0.05 * vib_loss
    return total_loss

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

num_epochs = 60
patience_epochs = 10
epochs_since_improvement = 0
train_losses, val_losses, val_maes = [], [], []
best_val_loss = float('inf')
best_model_state = None

print("\nTraining speed estimator...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_sq_err = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = custom_loss(outputs, y, X)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        train_sq_err += float(((outputs[:, 0] - y) ** 2).sum().item())

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_rmse = float(np.sqrt(train_sq_err / len(train_loader.dataset)))

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
          f"Train RMSE: {train_rmse:.4f} m/s | "
          f"Val RMSE: {rmse:.4f} m/s | "
          f"Val MAE: {mae:.4f} m/s")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_preds, best_targets = preds, targets
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1
        if epochs_since_improvement >= patience_epochs:
            print(f"  Early stopping: no val improvement in {patience_epochs} epochs.")
            break

# Save best model
model_path = METRICS_DIR / 'phase6_speed_model.pth'
torch.save(best_model_state, model_path)
model_size_mb = model_path.stat().st_size / (1024 * 1024)

final_rmse = float(np.sqrt(np.mean((best_preds - best_targets)**2)))
final_mae = float(np.mean(np.abs(best_preds - best_targets)))

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

# Scatter: predicted vs actual (best epoch)
sample_idx = np.random.choice(len(best_preds), min(2000, len(best_preds)), replace=False)
axes[1].scatter(best_targets[sample_idx], best_preds[sample_idx], alpha=0.3, s=5, color='steelblue')
lim = max(best_targets.max(), best_preds.max())
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
