# WayFinder — Data Collection Plan

Companion to `DATASET_PROTOCOL.md` (the split/leakage rules) and `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` (the evidence this plan is built from). This document answers: *given the evidence, what specific data should be collected next, in what order, and how will we know if it's working?*

No data has been collected yet as part of this plan. This is the plan, not the collection.

---

## 1. Data Requirements — sized from evidence, not guesses

### 1.1 Total hours

**Current**: 6.75 hours (train), single driver/vehicle family.

The learning curve in `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` §4 Exp. 1 (25%/50%/100% of current data → 45.1/35.3/25.2 km/h validation RMSE) shows no saturation. Fitting a simple power-law trend to those three points (`RMSE ≈ 57.0 · hours^-0.42`) **purely as an illustrative planning aid — this is not a guarantee, per the instruction not to extrapolate the curve as if it were certain**:

| Training hours | Fitted trend (illustrative only) |
|---|---|
| 6.75 (current) | 25.6 km/h (actual: 25.2 km/h) |
| 13.5 (2×) | ~19.1 km/h |
| 27 (4×) | ~14.3 km/h |
| 54 (8×) | ~10.7 km/h |

This trend says nothing about the end-to-end drift %, which must be *measured*, not inferred from RMSE — the whole point of `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` Experiment 4 was that RMSE and downstream drift are related but not simply proportional (the EKF's own ceiling and window-to-window variance both matter). **Target: 2× current hours (~13.5h) for Stage 1, re-measure everything, and let the measured result — not the fitted line — decide whether to continue to 4×.**

### 1.2 Independent drivers and vehicles

**Current**: training data comes from **one driver/vehicle family** (100% of train). Test data comes from **two drivers never seen in training at all**, and the domain-shift experiment (Exp. 2) measured a real, secondary gap from this (val RMSE 7.38 → test RMSE 8.47 m/s, +15%, plus a sign flip in bias — the model's regression-to-the-mean calibration is somewhat specific to the one driver/vehicle it was trained on).

**Target for Stage 1: at least 3 additional distinct driver/vehicle identities added to train** (up from the current 1), with **at least 2 of them being different vehicles**, not just different people in the same car — vehicle mass/suspension/tire type plausibly affects the vibration-to-speed mapping the model has to learn, and this is currently completely unvaried in training. This is a floor, not a target ceiling: more distinct identities is better, but 1→4 identities is the single highest-value step available (going from zero within-training-set driver variation to some).

### 1.3 Road types, speeds, and driving regimes — prioritized by observed failure mode

The bias-by-speed-bin analysis from the CNN-GRU rework (`PROJECT_REPORT.md` §3.2) found the model's error is not uniform across speed — it's worst exactly where **training data is thinnest**:

| Speed range | Share of current training windows | Observed problem |
|---|---|---|
| 0–0.5 m/s (near-stationary) | 9.5% | Highest hallucination rate (predicting motion while stopped) |
| 0.5–10 m/s (city/urban) | ~24% combined | Largest positive bias (over-prediction) |
| 10–20 m/s (cruising) | 47.5% (majority) | Best-represented, least biased |
| 20+ m/s (highway) | 18.9% | Largest negative bias (under-prediction) |

**This directly sets collection priorities:**
- **P0 — Stop-and-go / urban traffic**: deliberately over-represent low-speed and full-stop driving (traffic lights, stop signs, congestion) relative to what a "typical drive" would naturally contain. This is the single most failure-linked gap.
- **P0 — Highway/high-speed cruising** (>60 km/h sustained): the other underrepresented tail.
- **P1 — Frequent, clear acceleration/deceleration events**: separately from speed *level*, the alignment estimator (`src/preprocessing/alignment.py`) only gets a physically-grounded "forward axis" reading when a drive contains enough clear accel/brake events (>0.15 m/s² sustained, per the current threshold) — only 10/42 current training sequences triggered this; the other 32 fell back to a (now sign-corrected, but still less precise) PCA estimate. Drives with clear stop-sign/traffic-light/merge acceleration events directly improve alignment quality, not just speed-label coverage. City driving with frequent junctions serves both this and the P0 stop-and-go goal simultaneously.
- **P1 — Curved roads / turns / roundabouts**: moderate priority. The EKF-ceiling experiment (Exp. 4) showed the *current* formulation is already adequate given accurate speed, including on the existing mostly-straight-road-dominated test set — but this hasn't been stress-tested on turn-heavy driving, and the (currently-disabled/no-op, see `PROJECT_REPORT.md` §3.3) NHC redesign in `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` §3 direction G would need turn data to ever be evaluated meaningfully if revisited later.
- **P2 — Road surface variety** (highway asphalt vs. rougher local roads): plausible secondary factor (vibration signature depends on surface), no direct evidence yet either way — worth some coverage, not a dedicated collection priority.
- **P2 (opportunistic only) — Weather**: the sensors in use (accelerometer, gyroscope, GPS) are not vision-based, so **lighting is irrelevant** to sensor readings and should not be a scheduling factor at all. Weather (wet roads) could plausibly affect tire/road vibration and GPS multipath slightly; capture opportunistically if it happens during other planned drives, but do not schedule dedicated sessions for it — no evidence yet justifies the priority.

---

## 2. Dataset Split (see `DATASET_PROTOCOL.md` for the full rules)

Applied to this plan specifically:
- New driver/vehicle identities are assigned to **train** or **val** at planning time (§3's matrix, `split` column) — decided now, before collection, not after seeing results.
- **The current test set (`S`, `Y`) is not touched by this plan.** No new identities are added to test in Stage 1. This keeps the one number that matters (end-to-end drift on frozen test drivers) comparable across stages.
- Cross-driver evaluation (checking a stage's model against validation drivers) is unlimited and encouraged during development.

---

## 3. Collection Matrix (Stage 1)

Only Stage 1 is specified in detail — deeper stages get refined once Stage 1's results are in, per the instruction to prioritize information gain over pre-committing to a large speculative plan.

| Scenario | Driver | Vehicle | Duration | Target speed range | Split | Priority | Purpose |
|---|---|---|---|---|---:|---|---|
| Urban stop-and-go, traffic lights/signs | New driver #2 | New vehicle #2 | 45–60 min | 0–30 km/h, frequent full stops | train | **P0** | Fix near-stationary hallucination (9.5%→target higher share); driver+vehicle diversity |
| Highway/arterial cruising | New driver #2 | New vehicle #2 | 45–60 min | 60–100 km/h sustained | train | **P0** | Fix high-speed under-representation (18.9%→target higher share) |
| Mixed urban with junctions/merges | New driver #3 | New vehicle #3 (different from #2 if feasible) | 45–60 min | 0–50 km/h, frequent accel/brake events | train | **P0** | More mask-based (non-PCA-fallback) alignment sequences; 2nd distinct vehicle |
| Rural/mixed roads | New driver #4 | Vehicle #2 or #3 (reuse) | 30–45 min | 20–80 km/h | train | P1 | Additional driver diversity without requiring a 4th vehicle |
| Curved roads / roundabouts | New driver #2 or #3 (reuse) | Vehicle #2 or #3 (reuse) | 20–30 min | 20–50 km/h, sustained turning | val | P1 | Turn-heavy data for future EKF/NHC work; kept in val (not train) so it can stress-test generalization to turning without inflating train's straight-road majority |
| Repeat session, different route, same driver/vehicle as an existing train identity | Existing new driver #2 | Existing new vehicle #2 | 30 min | Mixed | val | P1 | Same-driver/vehicle held-out-route validation, mirroring how `Vw`/`Vf` validate against `Vta`/`Vtb` today |
| Opportunistic wet-road session (only if it happens naturally) | Any of the above | Any of the above | N/A | N/A | train or val (per driver) | P2 | Secondary surface-variety coverage; not scheduled specially |

**Explicitly not in Stage 1**: any new test-set drives (test stays frozen at `S`, `Y`); large volumes of additional cruising-speed data resembling the current training set's already-dominant 10–20 m/s regime (redundant, low information gain per the bias analysis).

---

## 4. Staged Milestones

| Stage | Data state | When to retrain | Metrics to measure | Continue if... | Diminishing returns if... |
|---|---|---|---|---|---|
| **Stage 0** (current, frozen) | 6.75h, 1 driver/vehicle | — (done) | All of §7 below, values in `results/metrics/baseline_frozen_2026-09-15/` | — | — |
| **Stage 1** | ~13.5h (≈2×), +3 driver/vehicle identities per §3's matrix | After all Stage-1 drives are ingested and pass diagnostics (§6) | Full §7 metric set, on frozen test set | Val RMSE improves by a margin comparable to the 25%→50%→100% steps (roughly 20–30% relative) **and** end-to-end drift improves and isn't driven by one sequence | Val RMSE improves <10% relative, or end-to-end drift is flat/worse despite RMSE improving (would itself be an important, reportable finding — see `DATASET_PROTOCOL.md` §5.5) |
| **Stage 2** | ~27h (≈4×), broaden driver/vehicle count further per Stage-1 findings | After Stage-2 data ingested | Same, plus compare Stage 1→2 step size to Stage 0→1 step size | Stage 1→2 step is still a similar order of magnitude to Stage 0→1 | Stage 1→2 step is much smaller than Stage 0→1 — signals the curve is bending; time to revisit `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` §3's other directions (feature representation, EKF/NHC redesign, or a harder look at sensor observability) instead of collecting more of the same |
| **Stage 3** | Larger/diverse dataset, scope set by Stage-2 outcome | Only if Stage 2 justified it | Same | — | If Stage 3 is reached and still shows returns, keep going; if not, stop expanding data and treat the ceiling as evidence about sensor observability, not a data problem anymore |

**No stage is entered on a schedule** — each one starts only after the previous stage's data has actually been collected, ingested, and diagnosed clean (§6). This plan does not commit to a calendar; it commits to a decision rule at each step.

---

## 5. Data-Ingestion Pipeline — what exists, what was built

### 5.1 What the repository already has

- **Recording**: `mobile_app/wayfinder_app/lib/core/navigation_service.dart` already has a working session recorder (`startRecording()`/`stopRecording()`, ~10 Hz) writing `drive_<timestamp>.csv` with columns `timestamp_iso8601, ax_ms2, ay_ms2, az_ms2, gx_rads, gy_rads, gz_rads, gps_lat_deg, gps_lon_deg, gps_speed_ms, gps_heading_deg`, wired to a Record button in `home_screen.dart`. **This did not need to be built.**
- **Preprocessing/normalization/sequencing/training/evaluation**: `src/preprocessing/{loader,alignment,pipeline}.py`, `src/ai_models/{dataset,models,train_phase6,inference}.py`, `src/navigation/{ekf_fusion,evaluate_system}.py` — all already fixed and verified (`PROJECT_REPORT.md` §4) and require **no changes** to accept more data of the same shape.
- **What was missing**: nothing bridges the recorder's raw CSV format to the training pipeline's expected input, and nothing reports on new data quality before it's used.

### 5.2 What this plan adds (§ "minimal reusable tooling", implemented alongside this document)

- **`src/preprocessing/new_drive_ingest.py`**: converts one or more app-recorded `drive_*.csv` files into the exact on-disk V-file/S-file CSV format `discover_sequences()`/`load_sequence()` already parse (see `loader.py`'s `V_COLS`/`S_COLS`), written under `data/wayfinder-drives/Synchronised V abd S datasets/Categorised IOVNB Dataset/<driver_id> (<vehicle_id>)/<drive_id>/`. This means **zero changes to `loader.py`, `pipeline.py`, `dataset.py`, or `inference.py`** — new drives become indistinguishable, at the code level, from IO-VNBD sequences. It also writes/updates `data/wayfinder-drives/registry.json` per `DATASET_PROTOCOL.md` §4, and refuses to ingest a drive without an explicit `--split` assignment.
- **`src/preprocessing/dataset_diagnostics.py`**: the report described in §6 below, runnable against any combination of IO-VNBD + new-drive sequences, before any retraining happens.

### 5.3 Recommended (not implemented) recorder improvement

The recorder currently does **not** log GPS accuracy (`Position.accuracy`, already read in `_onGps` for other purposes) or magnetometer — both trivial additions matching the existing row-write pattern in `navigation_service.dart`. GPS accuracy specifically matters for new data because, unlike IO-VNBD's vehicle ECU speed, **the only ground-truth speed available for app-recorded drives is phone GPS speed**, which is noisier at low speed. Without logged accuracy, low-quality fixes can't be filtered out before being used as training labels. This is flagged here as a recommendation with exact file/line guidance, not implemented — it's a mobile app change outside the scope of "Python-side ingestion/diagnostic tooling," and should be a deliberate decision, not a side effect of this task.

**Consequence for Stage 1 in the meantime**: `dataset_diagnostics.py` (§6) flags GPS-speed volatility as a proxy for accuracy issues (large frame-to-frame jumps inconsistent with plausible vehicle acceleration) even without a logged accuracy field, and any new drive whose ground-truth speed looks unreliable by that proxy check should be reviewed manually before use.

---

## 6. Dataset Diagnostics (`src/preprocessing/dataset_diagnostics.py`)

Run **before** any retraining on new data. Reports, for any given set of sequences (and, where a split manifest is supplied, broken out per split):

- Total hours, total drives, driver count, vehicle count
- Speed distribution (histogram over the same bins used in the bias analysis: 0–0.5, 0.5–2, 2–5, 5–10, 10–20, 20+ m/s)
- Acceleration-magnitude distribution
- Yaw-rate (turning) distribution, as a proxy for steering/turning variety
- Target (speed-label) distribution, same as the speed distribution — called out separately since this is exactly what the class-imbalance bug was about
- Missing data (NaN counts per relevant column)
- V-file/S-file synchronization issues (length mismatches, as already found and handled for `Vta01b`/`Vtb10`)
- Duplicate/near-duplicate segment detection (sequences with suspiciously similar start/end GPS position and duration)
- Train/val/test contamination check: asserts no driver/vehicle identity appears in more than one split
- Distribution differences between splits (train vs. val vs. test speed/acceleration distributions), so a "more diverse" claim is checked, not assumed

This directly implements the requirement that it should be "impossible to accidentally claim 'more data' when the added data is essentially redundant" — the report is the thing that gets checked before that claim is made, every stage.

---

## 7. Metrics Tracked at Every Stage

Per `DATASET_PROTOCOL.md` §6, every stage's report carries all of:

1. CNN-GRU speed-estimation RMSE (validation)
2. CNN-GRU speed-estimation RMSE (held-out drivers, i.e. the test split)
3. CNN-GRU speed-estimation bias (signed mean error), both splits
4. EKF-only diagnostic ceiling drift (re-measured only if the test sequences or EKF change — expected to stay ~9.1% since the EKF is frozen)
5. End-to-end system drift (5-window-averaged), mean and per-sequence, on the frozen test set
6. Performance specifically on unseen drives within validation (cross-driver check)
7. Worst-case error: max and 90th-percentile drift across all evaluated windows, not just the mean

Frozen baseline (Stage 0) values for all of these are recorded in `results/metrics/baseline_frozen_2026-09-15/README.md`.

**A stage's result is only reported as an improvement if it satisfies every criterion in `DATASET_PROTOCOL.md` §5** — no exceptions for "the metric looks good this time."
