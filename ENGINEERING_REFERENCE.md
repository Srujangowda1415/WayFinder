# WayFinder — Engineering Reference

**Start here.** This is the current, accurate picture of what the system is, how it actually works, what it measurably does, and what is still broken or unverified. Written 2026-09-15 as a working reference for continued development.

> **On the other 16 markdown files in this repo:** most predate the debugging work and contain claims that turned out to be false (the "0.2% drift / 100% pass" headline came from a chain of bugs). They carry correction banners now, but treat any number in them as suspect unless it also appears here. §7 says which docs are still worth reading.

---

## 1. What the system is

A smartphone navigation app that keeps tracking the vehicle when GNSS drops — the SIH brief is **< 10% position drift over a 30-second GNSS-denied segment**, phone sensors only.

The repo has two halves:
- **`mobile_app/wayfinder_app/`** — the Flutter app. This is the product.
- **`src/`** — a Python research pipeline (IO-VNBD dataset loading, CNN-GRU training, offline EKF evaluation). This is *not* what runs on the phone; it informs it.

The two halves implement the same EKF independently (`src/navigation/ekf_fusion.py` and `lib/core/ekf_navigation.dart`). **They have drifted apart before** — a bug fixed in one was absent in the other for a long time. Treat the Dart one as the product and keep the Python one in sync deliberately.

---

## 2. How dead reckoning actually works now

### Drive mode, GNSS healthy
GPS anchors position and speed into the EKF each fix. Roads for a ~2.5 km box are prefetched from Overpass and cached **in memory**. The EKF heading propagates from the gyro; it is *not* corrected against GPS course.

### Drive mode, GNSS lost
Entered via the `📡 GNSS off` toggle, a >3 s GPS timeout, or (since this pass) a fix degrading past 25 m accuracy. On entry `_captureGnssAnchor()` snapshots the current speed. Then, every 100 ms:

1. **Stationary?** IMU-only detector (accel range < 0.5 m/s² over 2 s, mean near gravity, gyro < 0.1 rad/s, with 1.0 s hysteresis) → ZUPT, speed forced to 0.
2. **Moving?** Speed = the value held from GNSS-loss. Not integrated from the accelerometer, not predicted by a model.
3. **Road aiding** (if a road matched within 40 m and bearing agrees within 45°): heading ← road bearing, position snapped laterally onto the road.
4. **Heading**: gyro yaw rate, plus a compass complementary pull — **skipped entirely when a road matched**, because the two fight each other.
5. Predict → NHC → EMA smoothing onto the map marker.

### Walk mode
Step detection (fixed 0.75 m stride along compass heading) moves the marker, and GNSS re-anchors it when healthy (EMA 0.2, hard snap beyond 25 m). PDR carries position when GNSS drops.

### What is *not* in the path
- **The CNN-GRU.** Removed. Still in the repo (`lib/core/speed_estimator.dart`, `assets/model_weights.json`) but never loaded.
- **Route-based map matching.** `map_matcher.dart` exists but is hard-disabled at `navigation_service.dart` (`if (false && ...)`).
- **The phone→vehicle alignment matrix.** `_alignR` is still computed but nothing reads it.

---

## 3. Measured performance — read the configuration column carefully

There is **no single "current system drift" number**, because the shipped configuration changed after the last full Python evaluation. Do not quote a number without its configuration.

| Measurement | Configuration | Result |
|---|---|---|
| CNN-GRU RMSE, validation | same-driver held-out routes | 7.36 m/s (26.5 km/h) |
| CNN-GRU RMSE, unseen drivers | test split | 8.47 m/s (30.5 km/h) |
| **System drift, `phase8_full_evaluation.json`** | **CNN-GRU in the loop — NOT what ships** | 103.7% mean, 0/7 pass |
| System drift, hold-speed + ZUPT | Python harness, **used the car's ECU yaw sensor** | 32.8% mean, 45 m median |
| **DR replay, hold-speed** | **real Dart EKF, phone sensors only** | 173 m @600 s, 340 m @1200 s |
| **DR replay, + OSM road aiding** | **real Dart EKF, phone sensors only** | **48 m @600 s, 5 m @1200 s** |
| EKF-only ceiling | ground-truth speed substituted (diagnostic) | 9.1% mean |

Two things this table encodes:

- **The EKF is not the bottleneck.** Given accurate speed it reaches 9.1%, already near the SIH target. Kalman tuning has almost no headroom left.
- **The 32.8% figure is optimistic** and should not be quoted as the app's performance. It came from a harness using `vdf['yaw_rate_rads']` — the vehicle's ECU yaw sensor, which a phone does not have.

**The honest summary for a judge:** over a 30 s outage the system keeps the vehicle localised to tens of metres, holds position correctly when stopped, and re-anchors cleanly when GPS returns — versus a classical INS baseline that diverges by kilometres. It does not currently meet the < 10% bar.

---

## 4. Open issues

### P0 — these are what stand between the system and the target

**4.1 Heading is the dominant error, and it is unvalidated on real hardware.**
Replaying a recorded drive through the real Dart EKF: held speed stayed within 1.2 m/s of truth while heading drifted ~25°, and 428 m of travel at 25° is ~174 m cross-track — essentially the entire position error.

In IO-VNBD the phone gyro simply does not capture vehicle yaw: correlation **+0.04** against truth, magnitude **7× too small**, while the car's ECU sensor scores **−0.997**. No axis worked, and a gravity-projected (mounting-independent) yaw rate did not help either.

**Whether *your* phone is better is unknown and untested.** Modern phone gyros are generally good; this looks like a dataset logging limitation. This is the single highest-value thing to measure next. `tool/dr_replay.dart` plus a recorded drive answers it in ~10 minutes.

Related: `_vehYawRate = _gz` uses the **raw phone Z-gyro**, which is only the vehicle's yaw axis when the phone lies flat. `_alignR` exists and could rotate the gyro into the vehicle frame, but nothing uses it — and it never did, even before the ML removal.

**4.2 OSM matching locks onto the wrong parallel road.**
Greedy nearest-segment matching, even with bearing gating, snapped at **0 m distance while being 288 m wrong** in one test window. Bearing gating rejects crossing roads but not parallel ones (dual carriageways, service roads). This is why OSM helped in 3 of 7 windows rather than all.

The fix is proper HMM/Viterbi matching with transition continuity — candidate roads scored on emission (distance) *and* transition (route continuity from the previous match), rather than nearest-wins each tick. Stickiness to the previously matched way would capture most of the benefit cheaply.

### P1 — likely to bite in real use

**4.3 OSM cache is memory-only and network-dependent.** Lost on app restart; if Overpass is unreachable or you drive beyond the cached ~2.5 km box before a refresh, aiding silently disables and DR falls back to the (broken, per 4.1) gyro/compass heading. Persisting tiles to disk would make this robust.

**4.4 Stationary detection is untuned for your vehicle.** The ZUPT gate needs accel range < 0.5 m/s² over 2 s. A rough idling engine could exceed that, so the marker would creep at traffic lights — the single most visible demo failure. Watch `road aid` / `ZUPT state` in `DRIVE_TRACE` logs on a real drive.

**4.5 Held speed goes stale.** It is exact at outage onset and degrades as the vehicle accelerates or brakes. This is the fundamental limit of the current speed approach; accelerometer integration was measured *worse* (bias compounds quadratically into position over 30 s).

**4.6 Route-based map matching is still disabled.** When a destination is set, snapping to the known OSRM route has no wrong-parallel-road ambiguity and would be more reliable than free OSM matching. Deliberately left off to avoid re-introducing an old "fake motion while stationary" bug, but it is low-hanging fruit — gate it on `!_confirmedStationary` as the road matcher already is.

### P2 — correctness and maintenance debt

**4.7 Dead code from the ML removal.** `_rotateVec` (unused), `_alignR` / `_alignSpeedBuf` / `_drAnchorLat` / `_drAnchorLon` / `_drAnchorHeadingRad` / `_lastCompassRad` (all write-only). 75 analyzer warnings, 0 errors.

**4.8 Latent bug in dead code:** `_alignSpeedBuf.add(_gpsLat)` appends a *latitude* to a buffer named for speeds. Harmless only because nothing reads it.

**4.9 The CNN-GRU can't simply be switched back on.** Re-enabling needs *both* the `load()` call restored *and* `assets/model_weights.json` + `norm_stats.json` declared under `assets:` in `pubspec.yaml` — the missing declaration is why it never worked in the first place.

**4.10 Map snapping reuses `updateGnss()`** with the GNSS measurement noise (25 m²). Semantically it is a map constraint, not a satellite fix, and probably wants its own variance.

**4.11 No tests.** Zero Flutter tests; `src/evaluation/test_inference.py` is a diagnostic script, not a suite. The EKF and road matcher are pure functions and would be cheap to cover.

**4.12 Release APK is signed with the debug keystore.** Fine for demos, not for distribution.

### P3 — longer horizon

**4.13 The ML path is data-bound, not architecture-bound.** Learning curve is steeply unsaturated (45.1 → 35.3 → 25.2 km/h for 25/50/100% of data); 7× model capacity bought only 8.8%. Training data is 6.75 h from a **single driver/vehicle**. See `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` and `DATA_COLLECTION_PLAN.md`. The architecture and EKF are frozen pending data.

**4.14 Drift-metric artifact.** 8 of 35 evaluation windows have the vehicle essentially parked (0.5–1.1 m in 30 s), where "10% drift" means 5 cm accuracy. Drift is reported over moving windows, with stationary behaviour reported separately in metres. Disclosed, not silently applied — and worth raising yourself if a judge asks about methodology.

---

## 5. Things that were broken and are now fixed

Useful context, because several were invisible and could regress:

- The CNN-GRU **never ran on the phone** — its weights were never bundled, so `predictSpeed()` always returned null and DR speed decayed to zero within ~5 s. The marker froze.
- `_vehYawRate` was nested inside `if (_speedModel.isLoaded)`, so **yaw rate was permanently 0** and gyro heading propagation was dead.
- The drive recorder wrote to a **different package's directory** — every recording failed silently. Confirmed on-device: not one file existed.
- Alignment had a **sign-ambiguous forward axis**; ~40% of training data had forward pointing backwards.
- `predict_speed_sequence()` `.flatten()`'d a `(batch,3)` output, **interleaving speed/variance/vibration** into what was meant to be a pure speed array.
- `EKF_INS.predict()` **leaked ground-truth GPS speed** into the "GNSS-denied" simulation.
- Walk mode **never re-anchored to GPS** after the first fix.
- The drift metric measured **whole-trip** error rather than outage error.

---

## 6. Working on it

```bash
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools

# app
cd mobile_app/wayfinder_app
flutter analyze lib                  # expect 0 errors, ~75 warnings
flutter build apk --release && adb install -r build/app/outputs/flutter-apk/app-release.apk

# regenerate the launcher icon
python3 tool/make_icon.py

# replay a drive through the REAL Dart EKF (no driving needed)
dart run tool/dr_replay.dart <replay.csv> <outage_start_s> 30
OSM_JSON=<overpass.json> FIX_YAW=1 dart run tool/dr_replay.dart ...

# python pipeline
python3 src/preprocessing/pipeline.py
python3 src/ai_models/train_phase6.py
python3 src/navigation/evaluate_system.py
python3 src/preprocessing/dataset_diagnostics.py --data-root data/IO-VNBD-master
```

**The replay harness is the most useful tool here.** It runs the shipped `EKFNavigation` headless against recorded drives and reproduces the DR branch exactly. It is what found the heading problem, and it contradicted the Python numbers that had quietly been using a sensor the phone does not have. Prefer it over the Python harness when asking "what will the app do?".

Live debugging: `adb logcat | grep DRIVE_TRACE` prints held velocity, ZUPT state, road-aid match/snap distance and marker delta every tick.

---

## 7. Which documents to trust

| Document | Status |
|---|---|
| **ENGINEERING_REFERENCE.md** (this) | Current. Start here. |
| `PROTOTYPE_STATUS.md` | Current. On-device state, demo script, what is defensible to claim. |
| `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` | Current. Why the ML path is data-bound, with the experiments. |
| `DATA_COLLECTION_PLAN.md` / `DATASET_PROTOCOL.md` | Current. What data to collect and how to split it without leakage. |
| `PROJECT_REPORT.md` | Corrected but legacy. §4 has the real numbers; earlier sections are historical. |
| `PROJECT_STATUS.md` | Corrected banner, body outdated. |
| `PROJECT_TEST_AND_RESULTS_REPORT.md` | **Unreliable.** Self-contradictory; claims features that do not exist (e.g. "HMM Viterbi map matching" — `map_matcher.dart` is a plain nearest-segment projection, and disabled). |
| `PROJECT_FINAL_AUDIT.md`, `PROJECT_ARCHITECTURE_AUDIT.md`, `DRIVE_MODE_*.md`, `DR_AI_INTEGRATION.md`, `MODEL_EVALUATION.md`, `CNN_GRU_DIAGNOSTIC_REPORT.md`, `CLEANUP_REPORT.md` | Historical. Describe states the code has since left. `CNN_GRU_DIAGNOSTIC_REPORT.md` is still technically sound on the model itself. |

Consolidating or deleting the historical set would be a real improvement — their main cost is that a reader cannot tell which describes reality.

---

## 8. If you only do three things next

1. **Record a 10-minute drive and replay it.** Answers whether your phone's gyro tracks yaw (§4.1) — the question everything else depends on. The recorder works now; `tool/dr_replay.dart` takes it from there.
2. **Make road matching sticky** (§4.2). Prefer the previously matched way; only switch when a candidate is clearly better. Cheapest large win available.
3. **Persist the OSM cache to disk** (§4.3), so aiding survives an app restart and works outside the prefetched box.
