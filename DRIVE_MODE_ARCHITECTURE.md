# Drive Mode Architecture (Current)

## 1. Overview
The current Drive Mode pipeline estimates the vehicle's position during GNSS-denied periods (Dead Reckoning) using a combination of IMU sensors, an AI Speed Estimator (CNN-GRU), an Extended Kalman Filter (EKF), and a Map Matcher.

## 2. Sensor Collection & Preprocessing
1. **Accelerometer & Gyroscope:** Sampled at ~10Hz.
2. **Alignment:** Uses a dynamic alignment matrix `_alignR` to convert raw phone coordinates to the vehicle frame. 
   - *Issue:* `_alignR` defaults to the Identity matrix until 200 GNSS-active samples are collected.
3. **Normalization:** Uses pre-computed means and std-devs from the training dataset.
   - *Issue:* Training data was already vehicle-aligned (gravity on Z-axis). When the app passes raw phone data (e.g. portrait mode where gravity is on Y-axis) before alignment is complete, the normalization step produces out-of-distribution values (e.g., Z-axis -13.8 sigma).

## 3. Motion State & Speed Estimation
1. **Stationary Detection:** `isStationary` is evaluated in `_fuse()` based on `_alignAccelBuf`.
   - *Issue:* `_alignAccelBuf` is ONLY populated when GNSS is active. In DR mode, this buffer is stale, causing the system to forever assume the state prior to GNSS loss.
2. **Speed Estimation:** The CNN-GRU model predicts speed based on a 50-sample window.
   - *Issue:* Due to the normalization mismatch (Issue #2), the model hallucinates high speeds (~41 km/h) even when completely stationary.
3. **Stale Speed Decay:** If the ML model is warming up, speed decays by a factor of `0.98` per 100ms.
   - *Issue:* This decay is too slow, causing a "coasting" effect for 30+ seconds after stopping.

## 4. EKF Fusion & Constraints
1. **Fusion Loop:** `_fuse()` runs every 100ms.
2. **Prediction:** `predict(dt, yawRate)` integrates the current velocity state (`_x[3]`).
   - *Issue:* `predict()` is called *before* `updateSpeed()`. It integrates the previous (potentially stale) speed state, moving the position erroneously.
3. **NHC:** `updateNhc()` constrains lateral and vertical velocity.
4. **Velocity Update:** `updateSpeed()` updates the EKF with the new AI-predicted speed.

## 5. Map Matching & Rendering
1. **Map Matching:** `MapMatcher.snapToRoute()` snaps the EKF position to the nearest point on the route segment.
   - *Issue:* This runs unconditionally in DR mode if a route exists. Even tiny position drifts (caused by sensor noise or stale speed) are snapped to the road, creating artificial forward movement along the route.
2. **Marker Update:** The map marker is updated using an EMA (Exponential Moving Average) filter for smoothness.
