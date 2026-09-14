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
    def __init__(self, input_dim=6, conv1_ch=24, conv2_ch=48, hidden_dim=48,
                 num_classes=1, dropout=0.2):
        super(LightweightCNNGRU, self).__init__()

        # 1D CNN for feature extraction. Kernel 5 (vs. the original 3) widens
        # the receptive field over the 50-sample (5s @ 10Hz) window, and the
        # wider channel counts give the regressor more capacity to separate
        # genuine vehicle-dynamics signal from vibration noise.
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv1_ch, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=conv1_ch, out_channels=conv2_ch, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout)

        # GRU for temporal context
        self.gru = nn.GRU(input_size=conv2_ch, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

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
        x = self.dropout1(x)

        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # GRU
        gru_out, _ = self.gru(x)

        # Take the last timestep, not a mean-pool over the window: this task
        # is causal (predict speed AT the end of the window), and averaging
        # over the whole window smears in earlier, higher-speed dynamics
        # whenever a window transitions into a stop or slowdown — verified
        # empirically to bias predictions upward specifically at low/mid
        # true speeds (the case that matters most for drift, since errors
        # there integrate directly into position error).
        last_out = gru_out[:, -1, :]
        last_out = self.dropout2(last_out)

        out = self.fc(last_out)
        return out.squeeze(-1)

class MotionClassifier(LightweightCNNGRU):
    def __init__(self):
        super(MotionClassifier, self).__init__(num_classes=1)

class SpeedEstimator(LightweightCNNGRU):
    def __init__(self):
        super(SpeedEstimator, self).__init__(num_classes=3)
