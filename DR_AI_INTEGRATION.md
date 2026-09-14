# DR AI Integration Architecture

## 1. IMU Input
- Source: Smartphone Accelerometer (`ax`, `ay`, `az`) and Gyroscope (`gx`, `gy`, `gz`) via `sensors_plus` at ~10 Hz.
- Reference Frame: Calibrated to Vehicle Frame using the `_alignR` rotation matrix derived from GNSS track and gravity vector before GNSS loss.

## 2. Preprocessing
- Sliding window of 50 samples (~5 seconds of history).
- Input shape: `[1, 6, 50]` (Batch, Channels, Sequence Length).
- Channels: `ax, ay, az, gx, gy, gz`.
- Normalized using `normalization_stats.json` computed solely from the training set to prevent data leakage.

## 3. CNN-GRU Architecture
- **CNN:** 1D Convolutions (`Conv1d`) to extract local temporal features from the 6 IMU channels.
- **GRU:** 1-layer GRU to capture long-term temporal dependencies of vehicle dynamics.
- **Output:** Dense layer predicting a single scalar: `forward_velocity` (m/s).

## 4. Velocity Prediction
- The ONNX model `phase6_speed_model.onnx` is invoked every 100ms via `SpeedEstimator`.
- Output is clamped to realistic vehicle speeds (0 to 55 m/s).

## 5. EKF Measurement Update
- **CRITICAL:** The AI velocity does NOT directly drive the marker.
- The AI velocity is fed into the EKF as a measurement update: `_ekf.updateSpeed(validatedSpeed, variance)`.
- The EKF decides how much to trust this update based on the AI uncertainty (currently modeled or fixed).

## 6. ZUPT (Zero Velocity Update)
- If `_confirmedStationary` is true (based on low acceleration variance and low gyro magnitude), ZUPT is triggered.
- `_ekf.updateZupt()` forces the EKF internal velocity state to 0.0 with extremely low variance.
- AI predictions are ignored while stationary.

## 7. NHC (Non-Holonomic Constraint)
- Applied at every EKF prediction step: `_ekf.updateNhc()`.
- Constrains lateral (sideways) velocity to 0, ensuring the vehicle only travels forward along its heading.

## 8. DR/INS Integration
- `ekf.predict(dt, correctedYawRate)` integrates the corrected speed and yaw rate to update `x` and `y` position coordinates.

## 9. Map Matching
- Applied ONLY when `_confirmedStationary` is false and AI speed > 0.5 m/s.
- `MapMatcher.snapToRoute()` snaps the raw EKF output to the nearest route geometry.
- Prevents map-matching from inventing movement when the vehicle is stationary.

## 10. Final Marker Position
- The raw (or snapped) EKF target position is smoothed via an Exponential Moving Average (EMA) to prevent jitter.
- The UI binds to `pdrLat` and `pdrLon`, which are the authoritative sources for the Drive Mode marker.
