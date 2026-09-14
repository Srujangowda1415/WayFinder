# Frozen Baseline — 2026-09-15

This directory is an immutable snapshot of the model checkpoint and evaluation
results at the point the failure analysis (`FAILURE_ANALYSIS_AND_NEXT_STEPS.md`)
concluded the system is data-limited. **Never overwrite these files.** All
future data-collection stages compare against these numbers, not against
whatever happens to be in the live `results/metrics/` directory at the time.

## Frozen numbers (source of truth for comparison)

- Training data: 42 sequences, 311.4 km, 6.75 hours, single driver/vehicle family (`Vta`+`Vtb`, "Driver E")
- CNN-GRU validation RMSE: **7.365 m/s (26.51 km/h)** — `phase6_speed_estimator.json`
- CNN-GRU raw RMSE on genuinely unseen test drivers: **8.47 m/s (30.5 km/h)** — measured in `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` §4 Exp. 2, not stored as a file here (reproducible via `DATASET_PROTOCOL.md`'s diagnostic procedure)
- EKF-only diagnostic ceiling (ground-truth speed substituted, isolated diagnostic, never a system metric): **9.1% mean drift** — `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` §4 Exp. 4
- End-to-end system drift (5-window-averaged, the real SIH metric): **103.7% mean, 0/7 pass** — `phase8_full_evaluation.json`

## Frozen artifacts in this directory

- `phase6_speed_model.pth` — the exact trained weights behind the numbers above
- `phase6_speed_estimator.json` — CNN-GRU validation metrics at freeze time
- `phase8_full_evaluation.json` — full per-sequence system evaluation at freeze time
- `normalization_stats.json` — normalization stats the frozen model expects (needed to run it correctly if reloaded later)
- `split_manifest.json` — the exact train/val/test split used to produce these numbers

## How to compare a new result against this baseline

1. Never mix these frozen files with data from a new collection stage.
2. Regenerate `results/metrics/normalization_stats.json` / `split_manifest.json` fresh for the new, larger dataset (via `pipeline.py`) — do not reuse the frozen ones for new training.
3. Report new results in the same units/definitions used here: CNN-GRU RMSE in m/s (validation *and* held-out-driver test, kept separate), EKF-only diagnostic ceiling (if re-measured), and end-to-end 5-window-averaged system drift %.
4. A new result only counts as an improvement over this baseline if it satisfies every criterion in `DATASET_PROTOCOL.md` §5 ("What counts as a valid improvement").
