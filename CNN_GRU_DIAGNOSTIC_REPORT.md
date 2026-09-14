# CNN-GRU Diagnostic Report

## 1. Model Checkpoint Analysis

The trained model checkpoint `phase6_speed_model.pth` has been evaluated.
- **Validation RMSE:** 9.03 m/s (32.52 km/h)
- **Validation MAE:** 7.20 m/s (25.94 km/h)

The validation RMSE of 32 km/h is extremely poor for speed estimation, indicating the model is effectively useless at making accurate predictions on the validation set.

## 2. Normalization Stats

The normalization statistics used to standardise the model inputs:
- `accel_x` mean: -0.03 m/s²
- `accel_y` mean: -0.15 m/s²
- `accel_z` mean: 9.83 m/s²

**Critical Observation:** The `accel_z` mean is `9.83`, which represents the acceleration due to gravity. The model was trained on *aligned* vehicle-frame data where the Z-axis always points straight down (gravity).

## 3. Stationary Tests

We tested the PyTorch model with simulated stationary inputs.

### Test 1: Phone Flat (Vehicle Aligned)
- **Input:** `accel = [0, 0, 9.81]`
- **Normalized Input:** `accel_z = (9.81 - 9.83) / 0.71 = -0.03` (In Distribution)
- **Predicted Speed:** 0.44 m/s (1.58 km/h)
- **Result:** The model behaves reasonably when the phone is flat and correctly aligned.

### Test 2: Phone Upright (Portrait Mode, Typical Use)
- **Input:** `accel = [0, 9.81, 0]` (Gravity is in the Y-axis)
- **Normalized Input:** 
  - `accel_y = (9.81 - (-0.15)) / 1.94 = 5.13` (Out of Distribution)
  - `accel_z = (0 - 9.83) / 0.71 = -13.85` (Catastrophically Out of Distribution)
- **Predicted Speed:** 11.47 m/s (41.28 km/h)
- **Result:** The model hallucinates 41 km/h of speed when perfectly stationary simply because the phone is held upright!

## 4. Root Cause of Artificial Movement

The Dart application uses an alignment matrix `_alignR`. Until this matrix is properly populated (which requires 200 GPS samples), it defaults to the Identity matrix.

During this time, raw phone accelerations (e.g. gravity on the Y-axis for portrait mode) are passed into the model. The model normalizes this using the training statistics which expect gravity on the Z-axis. This results in standard scores of `-13.8` sigma being fed into the neural network, triggering massive false speed predictions (~11.5 m/s or 41 km/h).

The EKF then unconditionally integrates this hallucinated speed, causing the marker to rapidly shoot forward. Map Matching then snaps this wild movement to the nearest roads, creating the illusion that the vehicle is driving along a route.
