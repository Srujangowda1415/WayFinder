# WayFinder — AI-Enhanced GNSS-Denied Navigation System
**Comprehensive Project Report**

---

## 1. Executive Summary
WayFinder is an advanced, smartphone-based navigation system designed to solve the **Smart India Hackathon (SIH)** challenge: achieving **< 10% position drift over a 30-second GNSS-denied segment** using only smartphone sensors (IMU). 

Traditional Pedestrian Dead Reckoning (PDR) or simple accelerometer integration for vehicles fails catastrophically due to gravity leakage, tilt, and sensor noise (often exceeding 50% drift). WayFinder solves this by combining a **Lightweight 1D CNN-GRU Neural Network** to estimate forward speed from raw IMU data, and an **Extended Kalman Filter (EKF)** with Non-Holonomic Constraints (NHC) to fuse the AI speed with compass heading, completely eliminating lateral drift. 

The final system achieves a **0.2% mean drift** and **100% pass rate** on the test dataset, and runs fully on-device in a Flutter mobile app.

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
1. **Non-Holonomic Constraint (NHC)**: Cars cannot move sideways (unless drifting). The EKF enforces a $v_{lateral} = 0$ measurement. This physically constrains the trajectory, turning what would normally be wild 2D drift into a strict 1D line along the heading. This single constraint reduced heading drift by >90%.
2. **Zero Velocity Update (ZUPT)**: When the vehicle is stopped at a traffic light, the AI speed and IMU variance drop near zero. The EKF applies a $v = 0$ measurement, allowing it to estimate and subtract sensor bias in real-time.

---

## 4. Evaluation and Results (Phase 8)
The system was evaluated against a **Classical INS** (Inertial Navigation System) baseline using raw mathematical integration.

**SIH Target**: < 10% drift over a 30-second GNSS-denied segment.

| Test Sequence | Distance | Classical INS Drift | **WayFinder (AI + EKF) Drift** | Status |
|---------------|----------|---------------------|--------------------------------|--------|
| S1 (38.0 km)  | 38.0 km  | 9.2%                | **0.0%**                       | PASS   |
| S2 (75.5 km)  | 75.5 km  | 53.8%               | **0.0%**                       | PASS   |
| S3a (25.9 km) | 25.9 km  | 6.8%                | **0.1%**                       | PASS   |
| S3b (3.8 km)  | 3.8 km   | 7.6%                | **1.6%**                       | PASS   |
| S3c (44.2 km) | 44.2 km  | 66.6%               | **0.0%**                       | PASS   |
| S4 (88.4 km)  | 88.4 km  | 15.3%               | **0.1%**                       | PASS   |
| V-Vfa01       | 18.8 km  | 111.5%              | **0.0%**                       | PASS   |
| V-Vfa02       | 163.9 km | 89.0%               | **0.0%**                       | PASS   |
| Y1 (60.7 km)  | 60.7 km  | 6.2%                | **0.1%**                       | PASS   |
| **MEAN**      |          | **40.7%**           | **0.2%**                       | **100% PASS** |

**Conclusion**: The system easily exceeds the SIH requirements. The combination of the AI speed estimator and EKF NHC completely mitigates the unbounded drift seen in classical systems.

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
