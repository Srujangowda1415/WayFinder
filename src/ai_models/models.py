"""
models.py — PyTorch Models for IO-VNBD
WayFinder / IDR Navigation System

Lightweight models for smartphone deployment:
1. MotionClassifier (Phase 5)
2. SpeedEstimator (Phase 6)
"""

import torch
import torch.nn as nn

class LightweightCNNGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, num_classes=1):
        super(LightweightCNNGRU, self).__init__()
        
        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # GRU for temporal context
        self.gru = nn.GRU(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # GRU
        gru_out, _ = self.gru(x)
        
        # Take the last output
        last_out = gru_out[:, -1, :]
        
        out = self.fc(last_out)
        return out.squeeze(-1)

class MotionClassifier(LightweightCNNGRU):
    def __init__(self):
        super(MotionClassifier, self).__init__(num_classes=1)
        
class SpeedEstimator(LightweightCNNGRU):
    def __init__(self):
        super(SpeedEstimator, self).__init__(num_classes=3)
