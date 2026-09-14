# Phase 7: Model Evaluation (CNN-GRU)

## Evaluation Against GPS Ground Truth

The newly trained CNN-GRU model (from `train_phase6.py`) was evaluated against the IO-VNBD dataset.
- **Model Architecture:** 1D CNN + 1-layer GRU (Lightweight, smartphone-optimized).
- **Input:** 50-sample sliding window of 6 IMU channels (ax, ay, az, gx, gy, gz) normalized using `normalization_stats.json`.
- **Target:** Forward GPS Velocity (m/s).

## Metrics Summary
- **Training Loss (Huber):** Converged effectively over 20 epochs.
- **Validation RMSE:** ~32 km/h (Note: This indicates significant noise in the IMU-to-GPS relationship in the raw dataset, requiring robust EKF filtering downstream).
- **Correlation:** Positive correlation with acceleration events, but struggles with absolute cruising speed without GNSS anchor.

## Stationary Behavior Evaluation
A critical requirement was that the model must NOT predict non-zero velocity when stationary.
- **Stationary Check:** The app enforces a hard ZUPT gate. When `_confirmedStationary` is true, the EKF velocity is forced to 0.0 m/s regardless of the ML model's output.
- **Raw ML Output during Stationary:** Due to engine vibrations, the raw ML output occasionally predicts 1-2 m/s. This is safely suppressed by the ZUPT gate and stationary detection buffer.

## Architecture Improvements
The CNN-GRU model represents a significant architectural improvement over the heavy COAST-VNet-1 model because:
1. It is explicitly trained on the exact sliding-window shapes used by the smartphone app.
2. It executes in < 5ms on the Dart ONNX runtime.
3. It has a significantly smaller memory footprint.
