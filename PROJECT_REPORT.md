# WayFinder — AI-Enhanced GNSS-Denied Navigation System
**Comprehensive Project Report**

---

## 1. Executive Summary
WayFinder is an advanced, smartphone-based navigation system designed to solve the **Smart India Hackathon (SIH)** challenge: achieving **< 10% position drift over a 30-second GNSS-denied segment** using only smartphone sensors (IMU). 

Traditional Pedestrian Dead Reckoning (PDR) or simple accelerometer integration for vehicles fails catastrophically due to gravity leakage, tilt, and sensor noise (often exceeding 50% drift). WayFinder solves this by combining a **Lightweight 1D CNN-GRU Neural Network** to estimate forward speed from raw IMU data, and an **Extended Kalman Filter (EKF)** with Non-Holonomic Constraints (NHC) to fuse the AI speed with compass heading, completely eliminating lateral drift. 

The final system was previously reported to achieve a **0.2% mean drift** and **100% pass rate** on the test dataset — **this number has since been found to be measured incorrectly (see §4) and is not currently reliable**; the system runs fully on-device in a Flutter mobile app, but its true GNSS-denied drift performance needs re-evaluation with the corrected metric.

---

## 2. Dataset and Preprocessing
The system was trained on the open-source **IO-VNBD** (Inertial Odometry Vehicle Navigation Benchmark Dataset).
- **Size**: 71 synchronised driving sequences (1,238.6 km, ~27 hours).
- **Inputs**: Smartphone accelerometer and gyroscope at 10 Hz (S-file).
- **Ground Truth**: Vehicle ECU speed and GPS trajectories at 10 Hz (V-file).
- **Split Strategy**: The dataset was split at the *trajectory level* (not sample level) to prevent temporal leakage. Sequences from Driver E were used for training (42 seqs) and validation (20 seqs), while different drivers/vehicles (9 seqs) were held out for true generalisation testing.

---

## 3. The Core Architecture

### 3.1 IMU Alignment (Phase 4)
Smartphones can be mounted in any orientation inside a vehicle. Before feeding data into the neural network, the app calculates a dynamic rotation matrix:
1. **Down Vector ($Z_{veh}$)**: Extracted by averaging the gravity vector while stationary.
2. **Forward Vector ($X_{veh}$)**: Extracted using the smartphone compass and GPS heading while moving.
3. **Right Vector ($Y_{veh}$)**: Computed via the cross product $Z \times X$.

All raw IMU data is rotated into this vehicle frame in real-time.

### 3.2 AI Speed Estimator: CNN-GRU (Phases 5 & 6)
Accelerometers measure acceleration ($m/s^2$). Integrating this once yields speed, and integrating again yields position. Small sensor noise or tilt errors ($0.1 m/s^2$) compound exponentially, causing extreme drift. 

Instead of mathematically integrating the accelerometer, WayFinder uses an **Artificial Neural Network** to learn the complex, non-linear mapping between vehicle vibrations/IMU patterns and true forward speed.

**Model Architecture (`LightweightCNNGRU`)**:
- **Input**: A 5-second sliding window of 6-axis IMU data (50 samples @ 10 Hz).
- **Conv1D Layer 1**: Extracts local spatial/vibration features (16 filters, kernel size 3) → ReLU → MaxPool(2).
- **Conv1D Layer 2**: Extracts higher-level features (32 filters, kernel size 3) → ReLU → MaxPool(2).
- **GRU Layer**: A Gated Recurrent Unit (32 hidden units) processes the sequence of CNN features to capture temporal dynamics (acceleration and braking profiles).
- **Fully Connected (Dense) Layer**: Outputs a single float: predicted forward speed ($m/s$).
- **Size**: The model is incredibly lightweight (**~174 KB**) and runs in pure Dart code on the mobile app, requiring zero external heavy machine learning libraries (no TFLite needed).

### 3.3 Extended Kalman Filter (EKF) Fusion (Phase 7)
The AI model predicts forward speed, and the compass predicts heading, but these predictions contain noise. The EKF optimally fuses these measurements.

**EKF State Vector**: 
`[x, y, heading, speed, heading_bias, speed_bias]`

**Key EKF Constraints:**
1. **Non-Holonomic Constraint (NHC)**: Intended to enforce $v_{lateral} = 0$. In the shipped state model, velocity is parameterised purely as `speed` + `heading` (there is no separate lateral-velocity state), so the *correctly*-derived NHC innovation is identically zero for every heading and speed — it is a mathematical no-op, not a source of drift reduction. A sign/pairing bug in the original formula (`ekf_navigation.dart` / `ekf_fusion.py`, `updateNhc`/`update_nhc`) made it compute a spurious non-zero value (`v·cos(2·heading)`) instead, which was actively injecting incorrect corrections into the heading/speed state. **This has been fixed** (2026-09-15): the formula and its Jacobian now match, and the update is a verified no-op. The claim below that NHC "reduced heading drift by >90%" was not correct and has been removed — meaningful lateral-drift constraints would require a state model with an independent body-frame lateral-velocity term.
2. **Zero Velocity Update (ZUPT)**: When the vehicle is stopped at a traffic light, the AI speed and IMU variance drop near zero. The EKF applies a $v = 0$ measurement, allowing it to estimate and subtract sensor bias in real-time.

---

## 4. Evaluation and Results (Phase 8)

**⚠️ Correction (2026-09-15):** The table below (and the underlying `phase8_full_evaluation.json`) was produced with a metric bug in `run_ekf_fusion()` (`src/navigation/ekf_fusion.py`): `drift_pct` was computed as `final_error_m / total_dist_m` using the position error at the **end of the entire trip** and the **entire trip's distance** — not the error/distance during the 30-second simulated GNSS outage being tested. Since GNSS is reacquired after the 30s outage and keeps correcting the filter for the rest of the (often 1+ hour) trip, the end-of-trip error converges back toward zero regardless of how bad the dead-reckoning was during the outage. A synthetic test (AI speed deliberately wrong by 50% for the full 30s outage) reproduced this: the old metric reported **0.0% drift**, while the corrected outage-only metric (`denied_drift_pct`, added in the same fix) correctly reported **~47% drift**. The code has been fixed to compute and report `denied_drift_pct` (error accumulated *during* the outage window ÷ distance travelled *during* that window) as the primary GNSS-denied metric.
The numbers in the table below are therefore **not reliable** and must be regenerated by re-running `python src/navigation/evaluate_system.py` against the real IO-VNBD dataset (this repo's local `data/IO-VNBD-master/` is empty, so it could not be re-run here). The table is left in place for historical reference only — treat every number in the "WayFinder (AI + EKF) Drift" column as unverified until regenerated.

The system was evaluated against a **Classical INS** (Inertial Navigation System) baseline using raw mathematical integration.

**SIH Target**: < 10% drift over a 30-second GNSS-denied segment.

| Test Sequence | Distance | Classical INS Drift | **WayFinder (AI + EKF) Drift — UNVERIFIED, see correction above** | Status |
|---------------|----------|---------------------|--------------------------------|--------|
| S1 (38.0 km)  | 38.0 km  | 9.2%                | 0.0%                       | UNVERIFIED   |
| S2 (75.5 km)  | 75.5 km  | 53.8%               | 0.0%                       | UNVERIFIED   |
| S3a (25.9 km) | 25.9 km  | 6.8%                | 0.1%                       | UNVERIFIED   |
| S3b (3.8 km)  | 3.8 km   | 7.6%                | 1.6%                       | UNVERIFIED   |
| S3c (44.2 km) | 44.2 km  | 66.6%               | 0.0%                       | UNVERIFIED   |
| S4 (88.4 km)  | 88.4 km  | 15.3%               | 0.1%                       | UNVERIFIED   |
| V-Vfa01       | 18.8 km  | 111.5%              | 0.0%                       | UNVERIFIED   |
| V-Vfa02       | 163.9 km | 89.0%               | 0.0%                       | UNVERIFIED   |
| Y1 (60.7 km)  | 60.7 km  | 6.2%                | 0.1%                       | UNVERIFIED   |
| **MEAN**      |          | **40.7%**           | 0.2%                       | **UNVERIFIED** |

Additionally, `V-Vfa01`/`V-Vfa02` are from `Vf (Driver E)`, the same underlying driver as the training set (only the vehicle/session label differs) — the train/val/test split has been corrected in `src/preprocessing/pipeline.py` to move `Vf` into validation, so re-running the pipeline will also change which sequences appear as "test" results.

**Conclusion**: Pending re-evaluation with the corrected metric and split, no drift conclusion can currently be drawn. The model's own validation RMSE (~32 km/h, see `CNN_GRU_DIAGNOSTIC_REPORT.md`) suggests the true GNSS-denied drift is likely far higher than the numbers previously reported here.

---

## 5. Mobile Application Implementation
A production-ready Flutter application was developed to run the entire system on-device.

- **Dual-Mode**: Supports both `Vehicle Mode` (AI + EKF) and `Walking Mode` (Pedestrian Dead Reckoning via step detection).
- **Pure-Dart Inference**: The trained PyTorch CNN-GRU model was exported to a custom JSON weight format. A custom matrix-multiplication engine was written in Dart (`speed_estimator.dart`) to run the inference natively at 10 Hz without any heavy native C++ dependencies.
- **State Machine**: 
  - `GNSS Navigation`: High-accuracy GPS is used to anchor the EKF state.
  - `Dead Reckoning`: If GPS accuracy degrades or drops out for > 3 seconds, the app gracefully switches to DR mode, rendering the marker using the EKF's AI-driven state.
  - `GNSS Reacquired`: Smoothly transitions back to GPS ground-truth when signals are restored.
- **UI/UX**: Google Maps aesthetic with dark mode CartoDB tiles, animated trajectory paths, real-time telemetry (heading, speed, uncertainty radius), and OSRM-powered route planning.

---
*Report generated for the WayFinder SIH submission.*
