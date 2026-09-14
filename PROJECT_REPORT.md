# WayFinder — AI-Enhanced GNSS-Denied Navigation System
**Comprehensive Project Report**

> **📋 See `FAILURE_ANALYSIS_AND_NEXT_STEPS.md`** for the current root-cause
> analysis (learning-curve, capacity, and EKF-ceiling experiments) and hard
> recommendation: **CHANGE DATASET**. This report's §4 conclusion (that
> further model/EKF tuning was the path forward) is superseded by that
> document's evidence that the system is data-limited, not capacity- or
> formulation-limited.

---

## 1. Executive Summary
WayFinder is an advanced, smartphone-based navigation system designed to solve the **Smart India Hackathon (SIH)** challenge: achieving **< 10% position drift over a 30-second GNSS-denied segment** using only smartphone sensors (IMU). 

Traditional Pedestrian Dead Reckoning (PDR) or simple accelerometer integration for vehicles fails catastrophically due to gravity leakage, tilt, and sensor noise (often exceeding 50% drift). WayFinder attempts to solve this by combining a **Lightweight 1D CNN-GRU Neural Network** to estimate forward speed from raw IMU data, and an **Extended Kalman Filter (EKF)** to fuse the AI speed with gyroscope heading.

**Corrected result (2026-09-15):** the system was previously reported to achieve a **0.2% mean drift** and **100% pass rate**. That number came from a chain of bugs — a whole-trip-instead-of-outage-only drift metric, an NHC formula bug, an alignment sign-ambiguity bug that corrupted ~40% of training data, a normalization/alignment mismatch, an inference bug that scrambled speed/variance/vibration outputs together, and an EKF harness bug that leaked ground-truth GPS speed into the "GNSS-denied" simulation (see §3.2, §3.3, §4 for each). All are now fixed, and the CNN-GRU speed estimator was reworked (wider architecture, fixed loss weighting, inverse-frequency sample weighting to correct a training-data speed-bin imbalance). The evaluation methodology was also hardened: drift is now averaged over 5 outage windows per test sequence instead of one fixed window, since a single window turned out to be highly sensitive to luck. Re-running everything end-to-end against the real IO-VNBD dataset gives a **true mean GNSS-denied drift of 103.7%**, with **0 of 7** unseen-driver test sequences meeting the SIH <10% target. The system does not currently meet the SIH requirement, and the gap is larger than earlier single-window measurements (taken mid-debugging) suggested. See §4 for the full results, the debugging history, and why further progress likely needs more/better training data rather than more code fixes.

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

**Model Architecture (`LightweightCNNGRU`)** — updated 2026-09-15:
- **Input**: A 5-second sliding window of 6-axis IMU data (50 samples @ 10 Hz).
- **Conv1D Layer 1**: 24 filters, kernel size 5 (was 16/k3) → ReLU → MaxPool(2).
- **Conv1D Layer 2**: 48 filters, kernel size 5 (was 32/k3) → ReLU → MaxPool(2) → Dropout(0.2).
- **GRU Layer**: 48 hidden units (was 32); takes the last timestep's output (tried mean-pooling over the window — reverted, see changelog below) → Dropout(0.2).
- **Fully Connected (Dense) Layer**: Outputs `[speed, log_variance, vibration_score]`.
- **Size**: **~82 KB** (still comfortably lightweight) — validated to run in pure Dart on the mobile app.

**2026-09-15 training-side changelog** (validation RMSE: 32.5 km/h → 24.4 km/h): widened channels/kernel size; fixed a loss-weighting bug where an auxiliary "vibration" head was weighted 5x the primary speed task (cut to 0.05x); clamped log-variance to stop the GNLL loss from minimizing itself by inflating predicted uncertainty instead of improving the point estimate; trained longer (60 epochs, early-stopped) instead of a fixed 10; tried mean-pooling the GRU output over the window instead of taking only the last timestep — this smeared earlier (higher-speed) dynamics into predictions for windows ending in a stop/slowdown and made low-speed bias measurably worse, so it was reverted. The single biggest lever was **inverse-frequency sample weighting**: training windows were heavily imbalanced toward cruising speed (47% in 10-20 m/s, <10% near-stationary), which was driving a "regression to the mean" bias — low speeds over-predicted, high speeds under-predicted, worst exactly where it matters most for drift (a falsely-inflated speed during a real stop integrates straight into position error). Weighting samples by inverse speed-bin frequency dropped the stationary-hallucination rate (predicting >2 m/s while truly stationary) from 17% to 6%.

### 3.3 Extended Kalman Filter (EKF) Fusion (Phase 7)
The AI model predicts forward speed, and the compass predicts heading, but these predictions contain noise. The EKF optimally fuses these measurements.

**EKF State Vector**: 
`[x, y, heading, speed, heading_bias, speed_bias]`

**Key EKF Constraints:**
1. **Non-Holonomic Constraint (NHC)**: Intended to enforce $v_{lateral} = 0$. In the shipped state model, velocity is parameterised purely as `speed` + `heading` (there is no separate lateral-velocity state), so the *correctly*-derived NHC innovation is identically zero for every heading and speed — it is a mathematical no-op, not a source of drift reduction. A sign/pairing bug in the original formula (`ekf_navigation.dart` / `ekf_fusion.py`, `updateNhc`/`update_nhc`) made it compute a spurious non-zero value (`v·cos(2·heading)`) instead, which was actively injecting incorrect corrections into the heading/speed state. **This has been fixed** (2026-09-15): the formula and its Jacobian now match, and the update is a verified no-op. The claim below that NHC "reduced heading drift by >90%" was not correct and has been removed — meaningful lateral-drift constraints would require a state model with an independent body-frame lateral-velocity term.
2. **Zero Velocity Update (ZUPT)**: When the vehicle is stopped at a traffic light, the AI speed and IMU variance drop near zero. The EKF applies a $v = 0$ measurement, allowing it to estimate and subtract sensor bias in real-time.

---

## 4. Evaluation and Results (Phase 8)

**⚠️ Correction (2026-09-15, updated same day after CNN-GRU rework):** The numbers originally reported here were produced with a metric bug in `run_ekf_fusion()`: `drift_pct` used the position error/distance over the **entire trip**, not the 30-second outage being tested, so it converged near zero regardless of dead-reckoning quality. That bug is fixed (`denied_drift_pct`, isolating outage-only error/distance).

While validating the fix, three more bugs surfaced and were fixed:
- **`estimate_alignment_matrix()` had a sign-ambiguous "forward axis"** (`src/preprocessing/alignment.py`): 32 of 42 training sequences fell back to a PCA eigenvector with no defined sign, and ~half of those came out backwards — i.e. the same physical event (accelerating) had opposite-signed features across roughly half of training data. Fixed by resolving the sign against actual speed change direction; verified 0/42 sign-flips post-fix (was 17/42).
- **Normalization stats were computed from raw phone-frame IMU data, but the model is trained/run on vehicle-frame (aligned) data** (`src/preprocessing/pipeline.py`) — inconsistent with what the model actually sees. Fixed to compute stats from aligned data.
- **`predict_speed_sequence()` (`src/ai_models/inference.py`) silently scrambled predictions**: the model outputs `(batch, 3)` = `[speed, log_var, vibration]`, but a bare `.flatten()` interleaved all three columns before truncating against the output index list — so the "AI speed" sequence fed into every evaluation run was actually a mix of speed, log-variance, and vibration values, not pure speed. Fixed to select column 0 explicitly.
- **`EKF_INS.predict()` leaked ground-truth GPS speed into the "GNSS-denied" simulation**: it took a `forward_speed` argument and blended 10% of it into the state on every call — and `run_ekf_fusion()` was passing the true GPS speed for that argument on every tick, including inside the simulated outage window, before the speed measurement update even ran. This is the same "predict() before speed is validated" bug the mobile app already found and fixed (`DRIVE_MODE_FIX_REPORT.md` RC-2), never back-ported to this Python harness. Fixed: `predict()` no longer takes an external speed argument, and the loop now updates speed (from AI or ZUPT) *before* calling predict(), matching `ekf_navigation.dart`.

The CNN-GRU itself was also reworked (see §3.2 changelog): wider CNN channels, a fixed loss (the vibration auxiliary head was weighted 5x the primary speed task — cut to 0.05x; log-variance is now clamped to stop the GNLL loss from "cheating" by inflating uncertainty), more epochs with early stopping, and **inverse-frequency sample weighting** during training, since training windows were heavily imbalanced (47% at 10-20 m/s cruising speed, <10% near-stationary), causing systematic "regression to the mean": low speeds over-predicted, high speeds under-predicted. This dropped the stationary-hallucination rate (predicting >2 m/s while truly stationary) from 17% to 6% and roughly halved the low-speed bias. Validation RMSE improved from 9.03 m/s (32.5 km/h) to roughly 6.5–7.4 m/s (23–27 km/h) depending on training run (see next paragraph on why that range matters).

**Two more things surfaced while trying to validate whether the CNN-GRU rework actually helped, and they matter more than any single number in the table below:**
1. **Training-run variance is large relative to the differences between architecture changes.** Re-running the *identical* config (same code, same data) produced val RMSE anywhere from 6.5–7.4 m/s and downstream mean outage drift anywhere from 53.8% to 64.2% on a single fixed test window — a bigger swing than most of the deliberate architecture changes tried (wider channels, mean-pooling vs. last-timestep, added magnitude features). Training now uses a fixed seed (`torch.manual_seed(42)`) to remove one axis of this noise, but a single seed is still one sample, not a distribution.
2. **A single fixed 30-second test window per sequence is not a reliable way to compare models.** `evaluate_system.py` now averages the outage-drift metric over 5 windows (at 15/30/45/60/75% into each trip) per sequence instead of one fixed window at 25%. This is a more statistically honest measurement — and it's *worse* than any single-window snapshot had suggested (a single window can get lucky). **This is the number that should be trusted going forward.**

**The table below is the 5-window-averaged result, re-run end-to-end against the real IO-VNBD dataset** (previously only Git LFS pointer stubs in this environment) — not a projection.

The system was evaluated against a **Classical INS** baseline using raw mathematical integration. Two EKF conditions are shown: **full GNSS** (GPS available the whole trip — mostly confirms the filter tracks GPS, not a dead-reckoning quality measure) and **30s GNSS-denied** (the actual SIH scenario, averaged over 5 windows per sequence).

**SIH Target**: < 10% drift over a 30-second GNSS-denied segment.

| Test Sequence | Distance | Classical INS Drift (whole trip, no GNSS) | EKF, full GNSS | **EKF, 30s GNSS-denied, 5-window avg (the SIH metric)** | SIH Status |
|---------------|----------|---------------------|-----------------|--------------------------------|--------|
| S1  | 38.0 km  | 9.2%   | 0.0%  | **19.1%**  | FAIL |
| S2  | 75.5 km  | 53.8%  | 0.0%  | **162.3%** | FAIL |
| S3a | 25.9 km  | 6.8%   | 0.0%  | **14.8%**  | FAIL |
| S3b | 3.8 km   | 7.6%   | 0.0%  | **45.8%**  | FAIL |
| S3c | 44.2 km  | 66.6%  | 0.0%  | **42.8%**  | FAIL |
| S4  | 88.4 km  | 15.3%  | 0.0%  | **226.2%** | FAIL |
| Y1  | 60.7 km  | 6.2%   | 0.1%  | **215.3%** | FAIL |
| **MEAN** |      | **23.7%** | **0.01%** | **103.7%** | **0/7 PASS** |

(Test set is `S (Driver A)` and `Y (Driver D)` only — 7 sequences — per the split-leakage fix in §2.)

**Conclusion**: The system does **not** meet the SIH <10% GNSS-denied drift target, and — once measured without a single-window's luck baked in — the gap is larger than earlier single-window snapshots suggested (mean 103.7% vs. the 42.6–64.2% range seen across different single-window/single-seed measurements during this debugging pass). All of the concrete bugs found along the way (alignment sign flip, normalization mismatch, scrambled inference output, EKF ground-truth leakage, training-data class imbalance) were real and worth fixing — each was individually verified — and the CNN-GRU's own validation accuracy did genuinely improve (32.5 → ~24 km/h RMSE). But two further architecture experiments (mean-pooling the GRU output, adding accel/gyro-magnitude input channels) both looked plausible and both made the real drift metric *worse* despite looking neutral-to-positive on validation-set bin bias — while simply re-running the same config with a different random seed swung the result by more than either of those changes did. That combination (large training variance + a model that performs wildly inconsistently across different segments of the same trip: S1/S3a near 15-19%, S2/S4/Y1 over 150%) points to the CNN-GRU having hit a real accuracy ceiling on this training set and model size, not a remaining code bug or an easy architecture tweak. Closing the SIH gap from here most likely needs meaningfully more/more-diverse training data (more drivers, more road/vehicle types) or a different sensing approach entirely (e.g. fusing in wheel-speed-like signals), rather than continued tuning of this same architecture against this same dataset.

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
