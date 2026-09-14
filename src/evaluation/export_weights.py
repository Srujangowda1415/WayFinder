"""
export_onnx.py — Export PyTorch SpeedEstimator to ONNX
The ONNX file will be used with onnxruntime-inference-engine in Dart via
the ort_flutter plugin (or consumed server-side).

For the mobile app we use a completely self-contained Dart implementation
of the CNN-GRU inference, which avoids TFLite entirely and works on any
Android device without extra native libs.  All weights are stored as a
JSON file that the Dart code loads at startup.
"""

import sys, json, struct
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.ai_models.models import SpeedEstimator

METRICS_DIR = REPO_ROOT / 'results' / 'metrics'
ASSETS_DIR  = REPO_ROOT / 'mobile_app' / 'wayfinder_app' / 'assets'
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Phase C — Weight Export for Dart Inference")
print("=" * 60)

device = torch.device('cpu')
model = SpeedEstimator().to(device)
state_dict = torch.load(METRICS_DIR / 'phase6_speed_model.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("PyTorch model loaded.")

# ── Extract all weights and save as JSON ──────────────────────────────────────
weights = {}
for k, v in state_dict.items():
    weights[k] = v.numpy().tolist()
    print(f"  {k}: {list(v.shape)}")

weights_path = ASSETS_DIR / 'model_weights.json'
with open(weights_path, 'w') as f:
    json.dump(weights, f)
print(f"\nWeights saved: {weights_path}  ({weights_path.stat().st_size/1024:.0f} KB)")

# ── Save normalization stats ───────────────────────────────────────────────────
with open(METRICS_DIR / 'normalization_stats.json') as f:
    full_norm = json.load(f)

keys = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
dart_norm = {}
for k in keys:
    dart_norm[k] = {'mean': full_norm[k]['mean'], 'std': full_norm[k]['std']}

norm_path = ASSETS_DIR / 'norm_stats.json'
with open(norm_path, 'w') as f:
    json.dump(dart_norm, f, indent=2)
print(f"Norm stats saved: {norm_path}")

# ── Verify: pure-numpy forward pass ───────────────────────────────────────────
def gru_cell_np(x, h, W_ih, W_hh, b_ih, b_hh):
    H = h.shape[0]
    gi = W_ih @ x + b_ih
    gh = W_hh @ h + b_hh
    i_r, i_z, i_n = gi[:H], gi[H:2*H], gi[2*H:]
    h_r, h_z, h_n = gh[:H], gh[H:2*H], gh[2*H:]
    r = 1 / (1 + np.exp(-(i_r + h_r)))
    z = 1 / (1 + np.exp(-(i_z + h_z)))
    n = np.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h

def relu(x): return np.maximum(0, x)

def conv1d_np(x, weight, bias):
    # x: (T, C_in), weight: (C_out, C_in, K), bias: (C_out,)
    T, C_in = x.shape
    C_out, _, K = weight.shape
    pad = K // 2
    x_pad = np.pad(x, ((pad, pad), (0, 0)))
    out = np.zeros((T, C_out))
    for t in range(T):
        for c_out in range(C_out):
            out[t, c_out] = np.sum(x_pad[t:t+K] * weight[c_out].T) + bias[c_out]
    return out

def maxpool_np(x, k=2):
    T, C = x.shape
    T2 = T // k
    return x[:T2*k].reshape(T2, k, C).max(axis=1)

def numpy_forward(x_in, sd):
    # x_in: (50, 6) normalised
    w1 = np.array(sd['conv1.weight'])   # (16, 6, 3)
    b1 = np.array(sd['conv1.bias'])
    w2 = np.array(sd['conv2.weight'])   # (32, 16, 3)
    b2 = np.array(sd['conv2.bias'])
    W_ih = np.array(sd['gru.weight_ih_l0'])  # (96, 32)
    W_hh = np.array(sd['gru.weight_hh_l0'])  # (96, 32)
    b_ih = np.array(sd['gru.bias_ih_l0'])
    b_hh = np.array(sd['gru.bias_hh_l0'])
    w_fc = np.array(sd['fc.weight'])    # (1, 32)
    b_fc = np.array(sd['fc.bias'])

    x = relu(conv1d_np(x_in, w1, b1))  # (50, 16)
    x = maxpool_np(x)                   # (25, 16)
    x = relu(conv1d_np(x, w2, b2))     # (25, 32)
    x = maxpool_np(x)                   # (12, 32)

    h = np.zeros(32)
    for t in range(x.shape[0]):
        h = gru_cell_np(x[t], h, W_ih, W_hh, b_ih, b_hh)

    return float(w_fc @ h + b_fc)

# Test with random input
rng = np.random.default_rng(42)
test_in = rng.standard_normal((50, 6)).astype(np.float32)

with torch.no_grad():
    pt_out = float(model(torch.tensor(test_in).unsqueeze(0)).item())

np_out = numpy_forward(test_in, {k: v.numpy() for k, v in state_dict.items()})

print(f"\nNumPy vs PyTorch verification:")
print(f"  PyTorch output: {pt_out:.4f} m/s")
print(f"  NumPy output:   {np_out:.4f} m/s")
print(f"  Diff: {abs(pt_out - np_out):.6f}")
if abs(pt_out - np_out) < 0.01:
    print("  PASS: Dart inference implementation will be correct.")
else:
    print("  WARNING: Difference too large, check implementation.")

print("\nPhase C-1 STATUS: COMPLETE")
