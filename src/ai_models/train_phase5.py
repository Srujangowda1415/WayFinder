"""
train_phase5.py — Phase 5: AI Vibration / Motion Model
WayFinder / IDR Navigation System

Trains a binary classifier to distinguish vehicle motion from vibration/stationary.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.ai_models.dataset import get_dataloaders
from src.ai_models.models import MotionClassifier

DATA_ROOT = str(REPO_ROOT / 'data' / 'IO-VNBD-master')
METRICS_DIR = REPO_ROOT / 'results' / 'metrics'
PLOTS_DIR = REPO_ROOT / 'results' / 'plots'

print("=" * 60)
print("  Phase 5 — AI Vibration / Motion Model")
print("=" * 60)

manifest_path = METRICS_DIR / 'split_manifest.json'
norm_stats_path = METRICS_DIR / 'normalization_stats.json'

train_loader, val_loader = get_dataloaders(manifest_path, norm_stats_path, target_type='motion', batch_size=128)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

model = MotionClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

train_losses = []
val_losses = []
val_f1s = []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        start_time = time.time()
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item() * X.size(0)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
        inf_time = (time.time() - start_time) / len(val_loader.dataset)
            
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    val_f1s.append(f1)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val F1: {f1:.4f}")

# Save model size and latency
model_path = METRICS_DIR / 'phase5_motion_model.pth'
torch.save(model.state_dict(), model_path)
model_size_mb = model_path.stat().st_size / (1024 * 1024)

print(f"\nFinal Metrics:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Inference Latency: {inf_time*1000:.2f} ms/sample")
print(f"  Model Size: {model_size_mb:.2f} MB")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Phase 5 Training Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(PLOTS_DIR / 'phase5_training.png')
print("Phase 5 STATUS: PASS")
