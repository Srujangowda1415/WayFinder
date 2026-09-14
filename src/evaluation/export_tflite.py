"""
export_tflite.py  — Phase C Step 1: Export trained model to TFLite
WayFinder / IDR Navigation System

Re-implements the CNN-GRU SpeedEstimator in TensorFlow/Keras (identical
architecture to the PyTorch model), loads the trained PyTorch weights,
transfers them layer by layer, then converts to TFLite and quantizes.

Output:
  assets/speed_estimator.tflite
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/srujangowda/Desktop/WayFinder')
from src.ai_models.models import SpeedEstimator

METRICS_DIR = Path('/Users/srujangowda/Desktop/WayFinder/results/metrics')
ASSETS_DIR  = Path('/Users/srujangowda/Desktop/WayFinder/mobile_app/wayfinder_app/assets')
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Phase C — TFLite Export")
print("=" * 60)

# ── 1. Load PyTorch model ──────────────────────────────────────────────────────
device = torch.device('cpu')
pt_model = SpeedEstimator().to(device)
state_dict = torch.load(METRICS_DIR / 'phase6_speed_model.pth', map_location=device)
pt_model.load_state_dict(state_dict)
pt_model.eval()
print("PyTorch model loaded.")

# ── 2. Extract weights ─────────────────────────────────────────────────────────
# CNN layers (PyTorch format: out_ch, in_ch, kernel)
w_conv1 = state_dict['conv1.weight'].numpy()       # (16, 6, 3)
b_conv1 = state_dict['conv1.bias'].numpy()         # (16,)
w_conv2 = state_dict['conv2.weight'].numpy()       # (32, 16, 3)
b_conv2 = state_dict['conv2.bias'].numpy()         # (32,)

# GRU weights (PyTorch packs W_ir, W_iz, W_in then W_hr, W_hz, W_hn)
w_ih   = state_dict['gru.weight_ih_l0'].numpy()    # (3*hidden, input=32)
w_hh   = state_dict['gru.weight_hh_l0'].numpy()    # (3*hidden, hidden)
b_ih   = state_dict['gru.bias_ih_l0'].numpy()      # (3*hidden,)
b_hh   = state_dict['gru.bias_hh_l0'].numpy()      # (3*hidden,)

# FC
w_fc = state_dict['fc.weight'].numpy()             # (1, 32)
b_fc = state_dict['fc.bias'].numpy()               # (1,)

print("Weights extracted.")

# ── 3. Re-implement in TF/Keras with transferred weights ──────────────────────
import tensorflow as tf

def build_keras_model():
    inputs = tf.keras.Input(shape=(50, 6), name='imu_input')

    # Conv block 1 — PyTorch uses (N, C, L), Keras uses (N, L, C)
    x = tf.keras.layers.Conv1D(16, kernel_size=3, padding='same',
                                use_bias=True, name='conv1')(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Conv block 2
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same',
                                use_bias=True, name='conv2')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # GRU
    x = tf.keras.layers.GRU(32, return_sequences=False, name='gru')(x)

    # Output
    outputs = tf.keras.layers.Dense(1, name='fc')(x)
    outputs = tf.keras.layers.Flatten()(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = build_keras_model()
print("Keras model built.")

# ── 4. Transfer Conv weights ───────────────────────────────────────────────────
# PyTorch Conv1d weight shape: (out_ch, in_ch, kernel_size)
# Keras  Conv1D weight shape:  (kernel_size, in_ch, out_ch)
model.get_layer('conv1').set_weights([
    w_conv1.transpose(2, 1, 0),
    b_conv1,
])
model.get_layer('conv2').set_weights([
    w_conv2.transpose(2, 1, 0),
    b_conv2,
])
print("Conv weights transferred.")

# ── 5. Transfer GRU weights ────────────────────────────────────────────────────
# PyTorch GRU stores gates as [r, z, n]; Keras stores as [z, r, h]
# PyTorch ih: (3*H, input), hh: (3*H, H)
H = 32
def reorder_pytorch_to_keras_gru(W, H):
    """Reorder from PyTorch [r,z,n] gate order to Keras [z,r,h]."""
    Wr, Wz, Wn = W[:H], W[H:2*H], W[2*H:3*H]
    return np.concatenate([Wz, Wr, Wn], axis=0)

# Keras GRU kernel:     (input_dim, 3*H)    — transposed
# Keras GRU recurrent:  (H, 3*H)            — transposed
# Keras GRU bias:       (2, 3*H) where [0]=input bias, [1]=recurrent bias
w_ih_reordered = reorder_pytorch_to_keras_gru(w_ih, H)    # (3H, 32)
w_hh_reordered = reorder_pytorch_to_keras_gru(w_hh, H)    # (3H, H)
b_ih_reordered = reorder_pytorch_to_keras_gru(b_ih, H)    # (3H,)
b_hh_reordered = reorder_pytorch_to_keras_gru(b_hh, H)    # (3H,)

keras_gru_kernel    = w_ih_reordered.T           # (input, 3H)
keras_gru_recurrent = w_hh_reordered.T           # (H, 3H)
keras_gru_bias      = np.stack([b_ih_reordered, b_hh_reordered], axis=0)  # (2, 3H)

model.get_layer('gru').set_weights([
    keras_gru_kernel,
    keras_gru_recurrent,
    keras_gru_bias,
])
print("GRU weights transferred.")

# ── 6. Transfer FC weights ─────────────────────────────────────────────────────
model.get_layer('fc').set_weights([w_fc.T, b_fc])
print("FC weights transferred.")

# ── 7. Sanity check: compare PyTorch vs Keras on random input ─────────────────
rng = np.random.default_rng(42)
test_input = rng.standard_normal((4, 50, 6)).astype(np.float32)

with torch.no_grad():
    pt_out = pt_model(torch.tensor(test_input)).numpy()

keras_out = model.predict(test_input, verbose=0).flatten()

diff = np.abs(pt_out - keras_out).max()
print(f"\nSanity check — max abs diff PyTorch vs Keras: {diff:.6f}")
if diff < 1e-3:
    print("PASS: models agree to <1 m/s")
else:
    print(f"WARNING: models differ by {diff:.4f} m/s — check weight transfer")

# ── 8. Convert to TFLite ──────────────────────────────────────────────────────
print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # dynamic-range quantisation
tflite_model = converter.convert()

out_path = ASSETS_DIR / 'speed_estimator.tflite'
out_path.write_bytes(tflite_model)
size_kb = len(tflite_model) / 1024
print(f"TFLite model saved: {out_path}  ({size_kb:.1f} KB)")

# ── 9. Verify TFLite inference ────────────────────────────────────────────────
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

tflite_preds = []
for i in range(4):
    interp.set_tensor(inp_det['index'], test_input[i:i+1])
    interp.invoke()
    tflite_preds.append(interp.get_tensor(out_det['index'])[0])

tflite_preds = np.array(tflite_preds).flatten()
tflite_diff  = np.abs(pt_out - tflite_preds).max()
print(f"TFLite sanity check — max abs diff vs PyTorch: {tflite_diff:.4f} m/s")

# ── 10. Save normalization stats for Dart ─────────────────────────────────────
with open(METRICS_DIR / 'normalization_stats.json', 'r') as f:
    norm = json.load(f)

dart_norm = {
    "accel_x_mean": norm['accel_x']['mean'],
    "accel_x_std":  norm['accel_x']['std'],
    "accel_y_mean": norm['accel_y']['mean'],
    "accel_y_std":  norm['accel_y']['std'],
    "accel_z_mean": norm['accel_z']['mean'],
    "accel_z_std":  norm['accel_z']['std'],
    "gyro_x_mean":  norm['gyro_x']['mean'],
    "gyro_x_std":   norm['gyro_x']['std'],
    "gyro_y_mean":  norm['gyro_y']['mean'],
    "gyro_y_std":   norm['gyro_y']['std'],
    "gyro_z_mean":  norm['gyro_z']['mean'],
    "gyro_z_std":   norm['gyro_z']['std'],
}

norm_path = ASSETS_DIR / 'norm_stats.json'
with open(norm_path, 'w') as f:
    json.dump(dart_norm, f, indent=2)
print(f"Norm stats saved: {norm_path}")

print("\nPhase C-1 STATUS: COMPLETE")
