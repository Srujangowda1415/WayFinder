# WayFinder — Engineering Reference

**Start here.** This is the current, accurate picture of what the system is, how it actually works, what it measurably does, and what is still broken or unverified. Written 2026-09-15 as a working reference for continued development.

> **Update, same day:** the root cause behind §4.1 (straight line through turns
> with GNSS off, real hardware) has been fixed and ported into the app — this
> section previously described the gap between `tool/dr_replay.dart`'s
> validated fixes and what actually shipped. See the end of §2, §4.1, and §5
> for what changed: gravity-projected yaw rate, magnetometer health/hard-iron
> calibration, GPS-course heading sync, route map matching re-enabled, and dead
> alignment code removed.

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
GPS anchors position and speed into the EKF each fix. Roads for a ~2.5 km box are prefetched from Overpass and cached **in memory**. The EKF heading now also syncs to GPS course-over-ground (`p.heading`) whenever speed > 3 m/s — previously heading propagated from the gyro only and was never corrected against GPS course, so every outage started from whatever the gyro had already drifted to.

### Drive mode, GNSS lost
Entered via the `📡 GNSS off` toggle, a >3 s GPS timeout, or a fix degrading past 25 m accuracy. On entry `_captureGnssAnchor()` snapshots the current speed. Then, every 100 ms:

1. **Stationary?** IMU-only detector (accel range < 0.5 m/s² over 2 s, mean near gravity, gyro < 0.1 rad/s, with 1.0 s hysteresis) → ZUPT, speed forced to 0.
2. **Moving?** Speed = the value held from GNSS-loss. Not integrated from the accelerometer, not predicted by a model.
3. **Road aiding** (if a road matched within 40 m and bearing agrees within 45°): heading ← road bearing, position snapped laterally onto the road.
4. **Heading**: gravity-projected (mounting-independent) gyro yaw rate, plus a compass complementary pull — skipped when a road matched (the two fight each other) **and now also skipped whenever the magnetometer isn't `compassValid`** (unavailable, stalled, or not yet hard-iron calibrated). Previously an unconditional pull toward `_compassRad` meant a magnetometer that never fired even once left the compass frozen at 0 rad while still being trusted — a constant ~0.5 rad/s correction dragging heading back to north regardless of real turns. This was the confirmed root cause of the reported "straight line through turns with GNSS off" bug.
5. Predict → NHC → EMA smoothing onto the map marker. Route-based map matching (`map_matcher.dart`) also runs here now when a route is set (see below).

### Walk mode
Step detection (fixed 0.75 m stride along compass heading) moves the marker, and GNSS re-anchors it when healthy (EMA 0.2, hard snap beyond 25 m). PDR carries position when GNSS drops.

### What changed in this pass
- **Yaw rate is gravity-projected**, not the raw phone Z-gyro. `_vehYawRate` now projects the full gyro vector onto the measured gravity direction (`_onAccel`), so vehicle yaw is captured correctly regardless of phone mounting angle/tilt — previously only a phone lying flat with Z pointing up worked. Validated in `tool/dr_replay.dart` (`FIX_YAW`) before being ported; see also `test/ekf_navigation_test.dart`.
- **Magnetometer health + hard-iron calibration.** `magHealthy` requires an event within the last 2 s (previously `_hasMag` latched true forever on the first-ever event). `compassValid` additionally requires each axis to have shown enough spread to trust an online-calibrated hard-iron offset. Only a `compassValid` compass is used for heading correction.
- **GPS course-over-ground syncs EKF heading** while GNSS is healthy and moving (`_onGps`), and `heading_bias`'s process/initial noise was loosened (`ekf_navigation.dart`) so it can actually be learned from these updates instead of staying frozen near 0.
- **Route-based map matching re-enabled** — the `if (false && ...)` hard-disable is gone; the existing GNSS-off/route-exists/confirmed-moving gate (§4.6, now resolved) already prevented it from inventing motion.
- **Dead code removed:** the phone→vehicle alignment matrix (`_alignR`, `_rotateVec`, `_estimateAlignment`, `_alignAccelBuf`/`_alignSpeedBuf`) was computed but never applied to anything — gravity-projected yaw supersedes what it was meant to do. The write-only DR-anchor fields (`_drAnchorLat/Lon/HeadingRad`) are also gone.
- **UI reads fused heading.** The map's heading-cone arrow and the "Head" stat now read `NavigationService.displayHeadingDeg` (the EKF state that actually drives position) instead of raw, possibly-stale compass output.
- **Map:** `TileLayer.userAgentPackageName` corrected to the real application id (`com.wayfinder.wayfinder_app` — same mismatch class of bug as the drive-recorder path-fix below), added `RichAttributionWidget` (OSM/CARTO attribution, previously missing) and `errorTileCallback` so a failed tile can't blank the map.

### What is *still not* in the path
- **The CNN-GRU.** Removed. Still in the repo (`lib/core/speed_estimator.dart`, `assets/model_weights.json`) but never loaded — measured worse than hold-speed+ZUPT (§3), left as-is deliberately.

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

**4.1 Heading is the dominant error — root cause found and fixed, confirmed on a real-hardware report.**
A real on-device test (GNSS off, DR enabled) reproduced exactly the symptom this section predicted: the vehicle went straight through turns, and the report specifically named the magnetometer as not working. Reading the actual pipeline turned up two concrete, confirmed bugs (not a sensor-quality unknown):
1. `_vehYawRate = _gz` used the raw phone Z-gyro, which is only the vehicle's yaw axis when the phone lies flat. **Fixed** — it's now gravity-projected (see §2), mounting-independent, and covered by `test/ekf_navigation_test.dart`.
2. `_hasMag` latched `true` forever on the first-ever magnetometer event and was never re-checked, so a magnetometer that stalled or never fired left `_compassRad` frozen at its initial value (0 rad) — while the DR path *unconditionally* pulled heading toward it every 100 ms whenever no road was matched. A constant ~0.5 rad/s pull toward a frozen "north" reading exactly reproduces "goes straight regardless of real turns." **Fixed** — heading correction now requires `compassValid` (§2).

The IO-VNBD finding below is a separate, still-open question — it does not explain the reported bug (that dataset issue would make DR *bad*, not specifically *straight*), but it's worth keeping distinct:

In IO-VNBD the phone gyro simply does not capture vehicle yaw: correlation **+0.04** against truth, magnitude **7× too small**, while the car's ECU sensor scores **−0.997**. No axis worked, and a gravity-projected yaw rate did not help either — this looks like a dataset logging limitation specific to that recording rig, not a general phone-gyro problem, but **whether *your* phone is better is still worth confirming** with `tool/dr_replay.dart` against a recording from the actual test device.

**4.2 OSM matching locks onto the wrong parallel road.**
Greedy nearest-segment matching, even with bearing gating, snapped at **0 m distance while being 288 m wrong** in one test window. Bearing gating rejects crossing roads but not parallel ones (dual carriageways, service roads). This is why OSM helped in 3 of 7 windows rather than all.

The fix is proper HMM/Viterbi matching with transition continuity — candidate roads scored on emission (distance) *and* transition (route continuity from the previous match), rather than nearest-wins each tick. Stickiness to the previously matched way would capture most of the benefit cheaply.

### P1 — likely to bite in real use

**4.3 OSM cache is memory-only and network-dependent.** Lost on app restart; if Overpass is unreachable or you drive beyond the cached ~2.5 km box before a refresh, aiding silently disables and DR falls back to the (broken, per 4.1) gyro/compass heading. Persisting tiles to disk would make this robust.

**4.4 Stationary detection is untuned for your vehicle.** The ZUPT gate needs accel range < 0.5 m/s² over 2 s. A rough idling engine could exceed that, so the marker would creep at traffic lights — the single most visible demo failure. Watch `road aid` / `ZUPT state` in `DRIVE_TRACE` logs on a real drive.

**4.5 Held speed goes stale.** It is exact at outage onset and degrades as the vehicle accelerates or brakes. This is the fundamental limit of the current speed approach; accelerometer integration was measured *worse* (bias compounds quadratically into position over 30 s).

**4.6 Route-based map matching — fixed, re-enabled.** The `if (false && ...)` hard-disable is gone. The gate was already correct (`!_confirmedStationary` etc.) and needed no changes.

### P2 — correctness and maintenance debt

**4.7 Dead code from the ML removal — fixed, removed.** `_rotateVec`, `_alignR`, `_alignAccelBuf`/`_alignSpeedBuf`, `_estimateAlignment`, and the write-only `_drAnchorLat`/`_drAnchorLon`/`_drAnchorHeadingRad`/`_lastCompassRad` are all gone (gravity-projected yaw supersedes what the alignment matrix was meant to do). 64 analyzer warnings remain (pre-existing style/lint items, e.g. non-camelCase matrix variable names, `avoid_print` in `DRIVE_TRACE` logging), 0 errors.

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
- `_vehYawRate` used the **raw phone Z-gyro** instead of a gravity-projected (mounting-independent) yaw rate — broke on any phone mount that wasn't flat. Fixed and unit-tested (§4.1, §2).
- The magnetometer had **no availability/staleness check**, so a stalled or never-fired sensor left the compass frozen while a DR-path correction unconditionally pulled heading toward it — the confirmed cause of a real-hardware "goes straight through turns" report. Fixed with `magHealthy`/`compassValid` gating and online hard-iron calibration (§4.1, §2).
- GPS course-over-ground was **never fed into the EKF** while GNSS was healthy, so heading (and `heading_bias`) entered every outage uncorrected. Fixed (§2).
- Route-based map matching was hard-disabled (`if (false && ...)`) despite the safety gate already being correct. Re-enabled (§4.6).
- The map's tile `userAgentPackageName` didn't match the app's real Android application id (the same class of bug as the drive-recorder path fix above).

---

## 6. Working on it

```bash
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools

# app
cd mobile_app/wayfinder_app
flutter analyze lib                  # expect 0 errors, ~64 warnings/infos (pre-existing style debt)
flutter test                         # EKF/gravity-projected-yaw unit tests (test/ekf_navigation_test.dart)
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
| `PROJECT_TEST_AND_RESULTS_REPORT.md` | **Unreliable.** Self-contradictory; claims features that do not exist (e.g. "HMM Viterbi map matching" — `map_matcher.dart` is a plain nearest-segment projection; it is enabled again as of this pass, but is still not HMM/Viterbi). |
| `PROJECT_FINAL_AUDIT.md`, `PROJECT_ARCHITECTURE_AUDIT.md`, `DRIVE_MODE_*.md`, `DR_AI_INTEGRATION.md`, `MODEL_EVALUATION.md`, `CNN_GRU_DIAGNOSTIC_REPORT.md`, `CLEANUP_REPORT.md` | Historical. Describe states the code has since left. `CNN_GRU_DIAGNOSTIC_REPORT.md` is still technically sound on the model itself. |

Consolidating or deleting the historical set would be a real improvement — their main cost is that a reader cannot tell which describes reality.

---

## 8. If you only do three things next

1. **Drive-test the fix.** GNSS ON → OFF → straight → turn → stop → turn → ON → OFF, watching `adb logcat | grep DRIVE_TRACE` (now also prints `compassValid`/`magHealthy` and live heading) and the map. Confirms §4.1's fix holds on your actual device/mount, not just in replay.
2. **Make road matching sticky** (§4.2). Prefer the previously matched way; only switch when a candidate is clearly better. Cheapest large win available.
3. **Persist the OSM cache to disk** (§4.3), so aiding survives an app restart and works outside the prefetched box.
