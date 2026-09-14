# WayFinder — Project Status

> **🚀 CURRENT STATE FOR SIH: see `PROTOTYPE_STATUS.md`.** The CNN-GRU has
> been removed from the navigation path and replaced with "hold the speed we
> had when GPS dropped + ZUPT", which measured roughly half the position
> error and ~5× the pass rate on held-out drivers (test: 32.8% mean drift,
> ~45 m median error after a 30 s outage, vs 42.1% / 93 m for the model).
> This is consistent with the freeze below — the fix was to stop depending
> on the model, not to tune it. Still short of the <10% SIH bar; see
> `PROTOTYPE_STATUS.md` §3 for what is and isn't defensible to claim.

> **🧊 ARCHITECTURE AND EKF ARE FROZEN, pending new data (2026-09-15).**
> `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` established, with direct experimental
> evidence, that the CNN-GRU architecture and the EKF/NHC formulation are
> **not** the bottleneck (capacity: 7× params → only 8.8% RMSE gain;
> EKF-only diagnostic ceiling: 9.1%, already near the SIH target) — the
> system is data-limited (unsaturated learning curve: 45.1→35.3→25.2 km/h
> RMSE as training data went 25%→50%→100%). Decision: **CHANGE DATASET**.
>
> Consequently: **no further architecture changes, loss-function changes,
> or EKF/Kalman-filter tuning should be made unless new evidence
> specifically justifies revisiting this freeze.** The current model
> checkpoint, split, and metrics are preserved unmodified at
> `results/metrics/baseline_frozen_2026-09-15/` as the comparison point for
> the data-collection effort now described in `DATA_COLLECTION_PLAN.md` and
> `DATASET_PROTOCOL.md`. See those two documents for what data is being
> collected next, how it will be split without leakage, and exactly what
> would count as a real improvement.

> **📋 See `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` for the current, evidence-based
> root-cause analysis and recommendation (decision: **CHANGE DATASET** — the
> model is data-limited, not capacity- or EKF-formulation-limited). This file
> below still documents the debugging history and is left intact, but its
> "Final Conclusion" is superseded by that document.

> **⚠️ Correction (2026-09-15):** The "0.2% Mean Drift" figure cited below came
> from a chain of bugs (whole-trip-vs-outage-only drift metric, NHC formula,
> alignment sign-ambiguity corrupting ~40% of training data, a normalization
> mismatch, an inference bug that scrambled model outputs, and an EKF harness
> bug leaking ground-truth speed into the "denied" simulation), all now fixed,
> plus a CNN-GRU rework (fixed loss weighting + inverse-frequency sample
> weighting). The evaluation itself was also hardened to average drift over
> 5 outage windows per sequence instead of one fixed window, since a single
> window turned out to be highly sensitive to luck (re-running the identical
> model config swung the single-window result from 53.8% to 64.2% mean drift
> on training-seed randomness alone). With that more honest measurement,
> real mean GNSS-denied drift is **103.7%**, with **0 of 7** unseen-driver
> test sequences passing the SIH <10% target — worse than any single-window
> snapshot suggested during debugging (see `PROJECT_REPORT.md` §4 for the
> full breakdown and history). Treat the "READY"/"PASS" statuses below as
> outdated — the system does not currently meet the SIH requirement, and the
> bottleneck now looks like training-data quantity/diversity rather than a
> fixable bug.

## Final Project Status

Overall: **NOT READY — 0/7 test sequences meet the SIH <10% GNSS-denied drift target on the (more rigorous) 5-window-averaged metric (see correction above)**

### Working
- IO-VNBD Dataset Preprocessing and Parsing
- Naive INS Mathematical Baseline (40.7% drift — proves AI is needed)
- CNN-GRU AI Velocity Predictor (speed + vibration score + uncertainty variance)
- EKF Fusion (Constant Velocity predict + AI speed measurement update with dynamic $R$)
- Non-Holonomic Constraints (NHC) — zero lateral velocity
- Zero Velocity Updates (ZUPT) — traffic light/stoppage detection
- HMM Viterbi Map Matching — snaps to active OSRM route during GNSS outages
- Smooth GNSS Handover, Dead Reckoning Transition, GNSS Reacquisition
- Real-Time Flutter App (CPH2613 deployed, 10 Hz inference)

### Partially Working
- **AI Vibration Score:** Predicted by the model but not yet surfaced in the UI (computed, stored in `_mlVibrationScore`, not displayed).

### Not Working
- None.

### Not Implemented
- None. All originally planned features are now complete.

### Not Tested
- High-speed GNSS-denied scenarios > 80 km/h (lack of training data for that speed range).
- Extreme rugged terrain/off-roading (violates NHC lateral velocity assumption).

---

### Bugs Fixed (Chronological)
1. **DR marker freezing** — `ekf.predict()` was inside a null guard; moved outside so it always fires.
2. **Marker teleport/fast jump** — `dt` was hardcoded; replaced with clamped real wall-clock measurement.
3. **1Hz map stutter (GNSS mode)** — Redundant 10Hz GPS speed updates collapsed EKF covariance; removed.
4. **Map shaking (Drive Mode)** — Compass used for yaw rate was magnetic noise; replaced with gyroscope.
5. **NHC formula bug** (2026-09-15) — `updateNhc()`/`update_nhc()` computed a spurious non-zero "lateral velocity" (`v·cos(2·heading)`) instead of the correct always-zero value for this state model, injecting incorrect corrections into heading/speed on every tick. Fixed in both `ekf_navigation.dart` and `ekf_fusion.py`.
6. **Drift metric bug** (2026-09-15) — `drift_pct` measured whole-trip error instead of GNSS-outage-only error, making dead-reckoning performance look far better than it is. Fixed by adding `denied_drift_pct`.
7. **Train/test split leakage** (2026-09-15) — `Vf (Driver E)` was in the test set despite being the same driver as training; moved to validation.

---

### Strongest Experimental Result — CORRECTED (2026-09-15)
The previously reported **0.2% Mean Drift** was invalid (see correction banner at top). Re-evaluated with the corrected metric, NHC fix, split fix, a reworked CNN-GRU (fixed loss weighting + inverse-frequency sample weighting), and — most importantly — a 5-window-averaged drift measurement instead of one fixed window: **103.7% mean drift** over 30-second simulated GNSS outages on unseen drivers, with **0 of 7** sequences meeting the SIH <10% target. (Earlier single-window measurements taken mid-debugging showed 42.6%/1-of-7, then 53.8%/2-of-7, then 61.7-64.2%/0-1-of-7 across different fixes and re-runs of the identical config — none of those are comparable to each other or to the final number, since a single fixed 30s window turned out to be highly sensitive to luck.) The classical INS baseline (23.7% mean drift on the same 7 sequences, whole-trip, never GNSS-corrected) is not a fair apples-to-apples comparison to the outage-only EKF number, but for reference: 4 of those 7 sequences also happen to beat the naive INS 10% bar despite never seeing GNSS at all.

### Biggest Remaining Technical Risk
The CNN-GRU speed estimator's own accuracy is the dominant source of error during a GNSS outage — its validation RMSE (~24-27 km/h after the 2026-09-15 rework, down from 32.5 km/h, varies by training run) directly drives the mean outage drift above, and it performs wildly inconsistently across different segments of the same trip (some 30s windows under 20% drift, others over 200%). Two further architecture experiments (mean-pooling the GRU output, adding accel/gyro-magnitude features) each looked plausible but made real drift worse, while simply re-running the same config on a different random seed moved the result by more than either change did — evidence that the model has hit a real accuracy ceiling on this training set/size, not that there's an easy remaining tweak. Gyroscope thermal drift over very long tunnels is a secondary concern once the speed estimate itself is accurate enough to matter.

### Final Conclusion
WayFinder is **not yet meeting SIH26168's core <10% GNSS-denied drift requirement** (0/7 test sequences pass on the 5-window-averaged metric, real mean drift 103.7% — see correction above). The EKF/NHC fusion architecture and mobile pipeline work as engineering artifacts, and the concrete bugs found this session (alignment sign flip, normalization mismatch, scrambled inference output, EKF ground-truth leakage, training-data class imbalance) were real and are now fixed — but closing the remaining gap most likely needs meaningfully more/more-diverse training data or a different sensing approach, not further tuning of this architecture against this dataset.
