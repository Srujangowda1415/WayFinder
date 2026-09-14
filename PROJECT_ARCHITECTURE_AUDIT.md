# Phase 1: Full Project Architecture Audit

## 1. Overview
The project is a Flutter-based mobile application (`wayfinder_app`) with a Python backend/training pipeline (`src/ai_models`, `src/preprocessing`). It implements an Intelligent Dead Reckoning (IDR) system for GNSS-denied navigation.

## 2. Current Architecture & Data Flow
- **Sensors:** Accelerometer, Gyroscope, and Magnetometer data is captured at ~10Hz via `sensors_plus` in `NavigationService`.
- **GNSS Mode:** When GPS is active, speed and heading are taken directly from the location stream. A transformation matrix (`_alignR`) is estimated to convert raw phone coordinates to the vehicle frame.
- **Dead Reckoning (Drive Mode):**
  - **Motion Detection:** A 20-sample circular buffer (`_motionAccelMags`) determines if the vehicle is stationary based on variance and mean.
  - **AI Velocity Estimation:** Currently handled by a CNN-GRU model (`SpeedEstimatorModel` / `phase6_speed_model.onnx`). The model takes a 50-sample window of aligned IMU data and predicts speed and vibration.
  - **ZUPT:** If stationary, speed is forced to 0.
  - **EKF Fusion:** `EKFNavigation` fuses the AI velocity (or ZUPT) and constraints (NHC) to update state (`x`, `y`, `heading`, `speed`).
  - **Map Matching:** `MapMatcher` snaps the raw EKF coordinates to a route geometry if the vehicle is moving.

## 3. Component Breakdown

### AI/ML Models
- `COAST-VNet-1/`: Obsolete pretrained model directory. Contains `coast_vnet_1.onnx` and evaluation scripts (`evaluate_coast_iovnbd.py`). **(To be removed)**
- `src/ai_models/models.py`: Contains PyTorch implementations of `MotionClassifier` and `SpeedEstimator` (CNN-GRU).
- `src/ai_models/train_phase6.py`: Training script for the CNN-GRU speed estimator.
- `mobile_app/wayfinder_app/assets/phase6_speed_model.onnx`: ONNX export of the current CNN-GRU model.

### Inference & Integration
- `mobile_app/wayfinder_app/lib/core/speed_estimator.dart`: Dart ONNX Runtime wrapper that runs the CNN-GRU model inference. Uses normalization parameters from `normalization_stats.json`.
- `mobile_app/wayfinder_app/lib/core/navigation_service.dart`: The core fusion loop (`_fuse()`). Feeds data to the AI model and EKF.

### Preprocessing
- `src/preprocessing/pipeline.py`: Generates the sliding windows and computes `normalization_stats.json`. 
- **CRITICAL FLAW:** Normalization in the training pipeline expects vehicle-aligned data (gravity in Z-axis). The Dart app currently feeds raw phone data if `_alignDone` is false, causing catastrophic out-of-distribution hallucinations (~41 km/h when stationary).

### Core Navigation Logic
- **EKF (`ekf_navigation.dart`):** Constant-velocity Extended Kalman Filter. Integrates heading and speed.
- **NHC:** Non-Holonomic Constraint implemented as a zero-lateral-velocity measurement update in EKF.
- **ZUPT:** Zero Velocity Update applied when `_confirmedStationary` is true.
- **Map Matching (`map_matcher.dart`):** Uses cross-track distance to snap position to polyline segments.

## 4. Potential Sources of Unwanted Movement
1. **AI Hallucinations:** The CNN-GRU model outputs 40+ km/h when stationary due to normalization/alignment mismatch.
2. **EKF Prediction Loop:** If the AI outputs a false speed, `predict()` integrates it into position.
3. **Map Matching:** Snaps drifting EKF positions to roads, artificially accelerating the marker along routes.

## 5. Review Required & Obsolete Files
- **Obsolete:** `COAST-VNet-1/` folder, `dataset_inspection.py`, `evaluate_coast_iovnbd.py`.
- **Review Required:** The current CNN-GRU weights in `results/metrics/phase6_speed_model.pth`. The model has a validation RMSE of ~32 km/h and needs to be completely retrained with a robust dataset split and correct preprocessing assumptions.
