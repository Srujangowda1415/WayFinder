# WayFinder — Failure Analysis & Next-Direction Recommendation
**2026-09-15. This document supersedes the "Conclusion" sections of `PROJECT_REPORT.md` and `PROJECT_STATUS.md` — those still carry the debugging history and are left intact, but this is the current, evidence-based read of where the project stands and what to do next.**

This analysis follows a bug-fixing pass (see `PROJECT_REPORT.md` §4 for the full list) that found and fixed five real, independently-verified issues: an alignment sign-ambiguity bug corrupting ~40% of training data, a normalization/alignment mismatch, an inference bug that scrambled model outputs, an EKF harness bug leaking ground-truth speed into the "GNSS-denied" simulation, and a training-data class imbalance. All are fixed. This document does **not** revisit those fixes or propose more tuning — it investigates, with new evidence, *why the system still doesn't meet the SIH target* and what's actually worth doing about it.

No numbers in this document are estimated or invented. Every figure is either read directly from `results/metrics/*.json` or produced by an experiment described in §4, each run against the real IO-VNBD dataset with the existing (already-fixed) code paths, unmodified.

---

## 1. Current Baseline

| Item | Value |
|---|---|
| Architecture | `LightweightCNNGRU`: Conv1d(6→24,k5)→ReLU→MaxPool2 → Conv1d(24→48,k5)→ReLU→MaxPool2→Dropout(0.2) → GRU(48→48,1 layer)→last timestep→Dropout(0.2) → Linear(48→3) `[speed, log_var, vibration]`. 20,811 parameters, ~83 KB (`src/ai_models/models.py`). |
| Training data | 42 sequences, **311.4 km, 6.75 hours**, from a **single driver/vehicle family** ("Driver E", sessions `Vta`+`Vtb`) |
| Validation data | 22 sequences, 591.1 km, 9.52 hours, same driver ("Driver E", sessions `Vw`+`Vf`), different routes |
| Test data | 7 sequences, 336.6 km, 10.71 hours, **two different, never-seen-in-training drivers/vehicles** ("Driver A" = `S`, "Driver D" = `Y`) |
| Split methodology | Trajectory-level, split by driver (no sequence ever spans train/val/test) — see `src/preprocessing/pipeline.py` |
| CNN-GRU raw RMSE (validation, same driver as training) | **7.36 m/s = 26.5 km/h** (`results/metrics/phase6_speed_estimator.json`) |
| CNN-GRU raw RMSE (test, unseen drivers) | **8.47 m/s = 30.5 km/h**, measured directly against `gps_velocity_ms` for this analysis (§4, Experiment 2) — not previously reported anywhere |
| EKF/downstream RMSE (outage-window only, real AI speed, 5-window avg) | **59.5 m** (`results/metrics/phase8_full_evaluation.json` → `mean_outage_rmse_m`) |
| EKF/downstream drift (the SIH metric, 5-window avg) | **103.7% mean, 0/7 sequences pass** |
| SIH target | < 10% drift over a 30-second GNSS-denied segment |
| Biggest remaining error source | CNN-GRU speed accuracy, not the EKF (see §4, Experiment 3 — the EKF's own ceiling is ~9.1%, already near the SIH target) |
| Conclusively ruled out as bugs | Alignment sign ambiguity, normalization/alignment mismatch, scrambled inference output, EKF ground-truth leakage, training-data class imbalance, evaluation-window cherry-picking (all fixed and independently verified — see `PROJECT_REPORT.md` §4) |

---

## 2. Bottleneck Determination

Four candidate explanations were directly tested with evidence (§4). Result, in order of contribution:

1. **Insufficient/non-diverse training data — dominant, confirmed.** The learning curve (Experiment 1) shows validation RMSE falling steeply and *without saturating* as training data increases (45.1 → 35.3 → 25.2 km/h for 25%/50%/100% of the current training set). A doubling of data (50%→100%) bought a ~29% relative RMSE improvement. There is no evidence of diminishing returns at the current data volume — the opposite: the curve is still steep. Separately, **100% of training data comes from one driver/vehicle family**, while the test set is two different, never-seen drivers/vehicles; this shows up as a real (if secondary) domain-shift signature: +15% RMSE and a directional bias flip (val model under-predicts on average, test over-predicts) between validation and test (Experiment 2).
2. **Model capacity — ruled out as the dominant lever.** A ~7x parameter increase (20.8K → 142.5K params, same data) bought only an 8.8% relative RMSE improvement (Experiment 3) — an order of magnitude less leverage than the data-scaling result for a much bigger capacity investment. The model is not meaningfully capacity-starved at its current size.
3. **EKF formulation / measurement model — ruled out as the dominant contributor.** With ground-truth speed substituted for AI speed in an isolated, diagnostic-only run (never used as a system metric), the EKF/NHC formulation's own ceiling drift is **9.1% mean** — already at the SIH target (Experiment 4). This means essentially all of the 103.7% mean drift is attributable to AI speed error being integrated by an EKF that is, on its own, adequate. Kalman-filter tuning, NHC formulation changes, or measurement-noise retuning have very little room left to help.
4. **Sensor observability** is not directly disprovable with these experiments, but the data-scaling curve's lack of saturation is evidence *against* "the sensors fundamentally don't contain enough information" being the binding constraint *yet* — if that were true, more data would stop helping past some point, and it hasn't, up to all currently available data.

**Conclusion: the dominant, confirmed bottleneck is training data — its quantity and, secondarily, its diversity (single driver/vehicle family) — not model architecture, not EKF formulation, and not (as far as current evidence shows) fundamental sensor observability.**

---

## 3. Candidate Directions

| # | Direction | Why it could help | Evidence | Data/code needed | Risk | Expected improvement | Worth it under SIH constraints? |
|---|---|---|---|---|---|---|---|
| A | **More/diverse training drives** | Directly targets the confirmed bottleneck | Learning curve unsaturated (§4 Exp.1); domain-shift signature vs. unseen drivers (§4 Exp.2) | New driving data — ideally multiple vehicles/drivers, varied road types; no code changes to the pipeline itself | Low (uses existing, working pipeline) | **High** — likely the largest lever available | **Yes — highest priority** |
| B | Different input representation/features | Could give the model more direct signal per sample | Tested (accel/gyro magnitude channels): made real drift *worse* despite plausible validation signal (documented in `PROJECT_REPORT.md` §3.2 changelog) | Feature engineering + retraining per attempt | Medium (easy to overfit small data with more features) | Low-to-negative, based on the one variant tested | No — deprioritize; revisit only after (A) |
| C | Pure temporal model (GRU/LSTM/TCN, no CNN front-end) | Might capture longer-range temporal structure the current 5s window misses | Not tested this pass | Architecture change + retraining | Medium | Unknown, plausibly small given (3) shows capacity isn't binding | Not now — same data-limited regime would likely apply |
| D | CNN + temporal attention/Transformer | More expressive temporal modeling | Not tested; capacity experiment (§4 Exp.3) suggests raw capacity has low remaining leverage, and attention models are typically *more* data-hungry, not less | Nontrivial architecture change, more compute, larger/more complex export to Dart | Medium-high (mobile deployment complexity, data-hungry) | Low, likely negative ROI given data-limited regime | No |
| E | Direct velocity/displacement formulation change | Different target parameterization might be easier to learn | Not tested | Retraining, target redefinition, EKF integration changes | Medium | Unknown | Not now — lower priority than (A) |
| F | Different EKF measurement model | Could reduce downstream drift independent of AI accuracy | **Directly tested and ruled out** — EKF ceiling is already ~9.1%, near target (§4 Exp.4) | N/A | N/A | Near-zero remaining headroom | No |
| G | Alternative sensor-fusion formulation (e.g. full body-frame velocity states, real NHC) | Could add genuine value once speed accuracy is good enough for NHC to have something to constrain | Architecturally, the current state model has no independent lateral-velocity DOF (see `PROJECT_REPORT.md` §3.3) — a real NHC would need this | Nontrivial EKF redesign | Medium | Low today (AI speed error dominates by ~10x over any EKF-formulation gain), possibly meaningful *after* (A) | Not now — revisit after AI speed accuracy improves |
| H | Fundamentally different sensing (e.g. wheel-speed pickup, external odometry hardware) | Bypasses the accelerometer-to-speed inference problem entirely | Not tested; would trivially solve the observability question but is out of scope for a smartphone-only SIH solution | New hardware integration — likely violates the "smartphone-only" SIH constraint | High (scope/feasibility) | High technically, but likely disqualifying for this competition's constraints | No — conflicts with SIH's smartphone-only framing |

---

## 4. High-Value Experiments Run

Five candidate experiments were considered; four were run (the fifth — temporal window length — was dropped because the first four already produced a consistent, decisive picture; see note at the end of this section).

### Experiment 1 — Data-scaling / learning curve
- **Exact change**: Trained the unchanged `SpeedEstimator` architecture on 25%, 50%, and 100% of the 42 training sequences (deterministic, seeded subset selection; 25% ⊂ 50% ⊂ 100%), all other hyperparameters identical to the production training script. Validation set (22 sequences, same driver, held-out routes) held fixed and unchanged across all three runs.
- **Metric**: Validation RMSE (m/s).
- **Threshold**: If RMSE at 100% has clearly flattened relative to 50% (e.g. <10% further relative improvement), that's evidence against "more data would help." If it's still improving by a similar or larger margin than the 25%→50% step, that's evidence for it.
- **Result**: 45.1 km/h → 35.3 km/h → 25.2 km/h. The 50%→100% step (10.1 km/h absolute improvement) was *larger* than the 25%→50% step (9.8 km/h). **No saturation.**
- **Conclusion drawn**: Strong evidence the model is data-limited, not capacity- or architecture-limited, at the current data volume.

### Experiment 2 — Domain-shift / generalization-gap check
- **Exact change**: None to the pipeline. Measured the already-trained model's raw speed predictions (via the existing `predict_speed_sequence()`) directly against `gps_velocity_ms` ground truth, separately for the validation set (same driver as training) and the test set (two drivers never seen in training).
- **Metric**: Raw RMSE (m/s), and mean signed bias.
- **Threshold**: A test/val RMSE ratio near 1.0 (say, <10% relative gap) would suggest the model generalizes across drivers fine, pointing away from "driver diversity" as a lever. A large gap (or a bias-sign flip) would support it.
- **Result**: Val RMSE 7.38 m/s (bias −3.46 m/s) vs. test RMSE 8.47 m/s (bias +2.50 m/s) — a **+15% relative RMSE gap and a sign flip in bias**.
- **Conclusion drawn**: A real but *secondary* generalization gap exists. It does not explain most of the error (raw RMSE is already poor even on same-driver data), but it does support prioritizing driver/vehicle *diversity*, not just *volume*, when acquiring more data.

### Experiment 3 — Model-capacity ceiling check
- **Exact change**: Trained a ~7x larger variant of the same architecture (conv channels 24/48→64/128, GRU hidden 48→128; 142,531 vs. 20,811 params) on the full, unchanged training set, with the same loss, optimizer, and early stopping.
- **Metric**: Validation RMSE (m/s).
- **Threshold**: A large capacity increase producing a proportionally large RMSE improvement (e.g. >20% relative) would indicate capacity is a live constraint worth pursuing. A small improvement would rule it out as the dominant lever.
- **Result**: 26.5 km/h → 24.2 km/h, an **8.8% relative improvement** for a 7x parameter increase.
- **Conclusion drawn**: Capacity has real but low leverage compared to data (Experiment 1's 2x-data-for-29%-improvement). Not the dominant bottleneck; not worth pursuing before addressing data.

### Experiment 4 — EKF-formulation ceiling check (diagnostic only — never a reported system metric, never used in production)
- **Exact change**: In an isolated, standalone diagnostic script (not touching `evaluate_system.py`, `ekf_fusion.py`, or any production file), ground-truth GPS speed was substituted for the AI speed input during the same 5 outage windows × 7 test sequences already used for the real evaluation. This measures the EKF/NHC formulation's own best-case behavior, explicitly to separate "EKF error" from "AI speed error." **This substitution is never present in the actual pipeline, was not committed to any file, and is not the system's reported performance.**
- **Metric**: Mean outage drift %, same definition as the real SIH metric.
- **Threshold**: If ceiling drift with perfect speed is still high (e.g. >50%), the EKF formulation itself has a real problem worth fixing. If it's low (near or under the SIH 10% target), the EKF is not the bottleneck.
- **Result**: **9.1% mean** (S1: 1.1%, S2: 18.6%, S3a: 5.0%, S3b: 1.2%, S3c: 2.1%, S4: 16.8%, Y1: 19.2%) vs. 103.7% with the real AI speed.
- **Conclusion drawn**: The EKF/NHC formulation is not the bottleneck — it's already close to the SIH target given accurate speed input. Essentially all of the remaining gap (94.6 percentage points) is attributable to AI speed error.

### (Not run) Experiment 5 — Temporal window length
Considered (5s vs. 10s window) to test whether more temporal context helps, but dropped: Experiments 1, 3, and 4 already converged on a consistent, well-triangulated conclusion (data-limited, not capacity- or formulation-limited), and a window-length change is itself a form of architecture tuning that the evidence already argues against prioritizing. Flagged here for transparency rather than silently omitted, per the instruction not to claim experiments that weren't run.

---

## 5. Decision

# **CHANGE DATASET**

**Not** CONTINUE CURRENT ARCHITECTURE. **Not** CHANGE MODEL. **Not** CHANGE SENSOR/ESTIMATION APPROACH.

**Why, from the measured evidence:**
- The learning curve (Exp. 1) shows the model is still steeply data-limited at 100% of currently available training data (6.75 hours, one driver/vehicle family) — the single strongest signal in this analysis.
- Model capacity (Exp. 3) has an order of magnitude less leverage than data volume for a much larger investment (7x params → 8.8% gain vs. 2x data → 29% gain). Changing the model architecture without more data is very unlikely to close the gap.
- The EKF/estimation formulation (Exp. 4) is not the bottleneck — its own ceiling is already near the SIH target. There is essentially no headroom left in Kalman-filter tuning, NHC redesign, or measurement-noise retuning to chase.
- A genuinely different sensing approach (wheel-speed pickup, external hardware) would likely work, but conflicts with the SIH constraint of a smartphone-only solution, and there's no evidence yet that the smartphone sensors are fundamentally insufficient — the data just hasn't been given enough of a chance to prove that one way or the other.

**What "change dataset" means concretely**: acquire and incorporate meaningfully more driving data — ideally spanning multiple drivers and vehicles (the current training set is a single driver/vehicle family; the test-set generalization gap in Exp. 2, while secondary, points the same direction) — before investing further in architecture changes, EKF changes, or additional hyperparameter tuning. Once more/more-diverse data is available, re-run the same experiment battery (learning curve especially) to check whether the model has moved into a capacity- or formulation-limited regime — at that point, but not before, revisiting directions B, C, D, E, or G from §3 would be justified by evidence rather than guesswork.

---

## 6. Constraints Honored

- No ground truth was leaked into any inference, filtering, normalization, or evaluation step used for the system's reported metrics. Experiment 4's ground-truth substitution is clearly marked as an isolated, non-production diagnostic and is reported only as a ceiling analysis, never as a system-performance number.
- No evaluation methodology was changed to improve a metric — the 5-window-averaged drift methodology (from the prior session) is unchanged and used as-is for all reporting here.
- Training/validation improvements (Experiments 1 and 3, both raw model RMSE) are kept explicitly separate from the downstream EKF/system metric (drift %) throughout this document.
- All figures were re-measured directly from `results/metrics/*.json` or freshly run experiments against genuinely held-out data (validation for Experiments 1/3, test for Experiment 2 and the real system metric, both untouched by any training).
- No results were fabricated; Experiment 5 is explicitly marked as not run rather than omitted or implied.
- All previously verified fixes (alignment sign, normalization, inference output, EKF leakage/ordering, class-imbalance weighting) are preserved unchanged; nothing in this analysis reverts them.
