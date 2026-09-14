# WayFinder — Dataset & Evaluation Protocol

**Written before any new data collection**, per `FAILURE_ANALYSIS_AND_NEXT_STEPS.md`'s recommendation to test whether more/diverse driving data closes the SIH gap. This document is the contract for how new data gets split, evaluated, and compared — it exists specifically so that "we added more data" claims can't be made or half-checked after the fact in whatever way makes the number look best.

The current architecture and EKF are **frozen** (see `PROJECT_STATUS.md`). Nothing in this document changes them.

---

## 1. Why a protocol, not just "add data and retrain"

Two lessons from the last debugging/analysis pass make this necessary:

1. **A single fixed evaluation window can lie.** The 5-window-averaged drift metric (already in production, see `PROJECT_REPORT.md` §4) exists because a single window swung results across a huge range. New data must not reopen this door — e.g. by evaluating on a conveniently-easy new drive, or by re-introducing a single-window comparison "just this once."
2. **The current train/val/test split is already driver-based, not random.** That's correct and must be preserved and *extended* consistently as new drivers/vehicles are added — never regress to frame-level or session-level random splitting, which would leak temporal autocorrelation between train and eval.

---

## 2. Unit of splitting: driver × vehicle, never a frame or a window

A **drive** is one recorded session (one V-file+S-file pair, or one converted app recording) — a few minutes to a few hours of continuous driving by one driver in one vehicle.

A **driver/vehicle identity** is the combination of who was driving and what vehicle they were in (a phone recording is tagged with both, via the registry described in §4). Two drives share an identity only if both driver and vehicle match.

**Rule: every drive from the same driver/vehicle identity goes entirely into one split (train, val, or test) — never split across two.** This is the existing rule in `src/preprocessing/pipeline.py`'s `trajectory_split()`; it does not change. What changes is that the driver/vehicle registry now needs to cover new-drive identities too (§4).

Frame-level or window-level random splitting is never acceptable for this project: adjacent 5-second windows within one drive are highly autocorrelated (same road, same speed regime seconds apart), so a random split at that level silently leaks information between "train" and "test" and produces optimistic numbers that don't reflect real generalization. This was never done here and must not be introduced.

---

## 3. Splits

| Split | Purpose | Rule |
|---|---|---|
| **Train** | Model fitting | Only identities explicitly assigned `train` in the registry (§4). Currently: `Vta`, `Vtb` (Driver E). |
| **Validation** | Model selection, early stopping, hyperparameter choices, architecture comparisons (like the experiments in `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` §4) | Only identities assigned `val`. Currently: `Vw`, `Vf` (Driver E — same driver as train, different routes/sessions). Used freely and repeatedly during development. |
| **Cross-driver evaluation** | An *intermediate* check during a data-collection stage: "does this look better on a driver we've already used for eval before, once more data / more drivers are added to train?" | Any identity assigned `val` or a *previously-used, already-reported* test identity. Can be re-run as often as needed. |
| **Cross-vehicle evaluation** | Same idea, but isolating vehicle rather than driver, once ≥2 vehicles exist in the training pool | Only meaningful once the registry has ≥2 distinct vehicle IDs in train. Compares performance on a vehicle seen in training vs. a genuinely new vehicle. |
| **Final test set** | The one number that gets reported as "the system's real-world performance" | Only identities assigned `test`. **Touched exactly once per data-collection stage** — see §5. Currently: `S` (Driver A), `Y` (Driver D). |

**The final test set is not a resource to iterate against.** It gets evaluated once a model is finalized for a given data-collection stage (§5 of `DATA_COLLECTION_PLAN.md`), the result is recorded, and it is not looked at again until the next stage's data has been added and a new model finalized. Every architecture/hyperparameter/feature decision during a stage is made using the *validation* split, exactly as `FAILURE_ANALYSIS_AND_NEXT_STEPS.md`'s experiments did.

### Growing the splits as new data arrives

- New driver/vehicle identities are assigned to train, val, or test **before their data is even fully collected**, based on the collection plan (`DATA_COLLECTION_PLAN.md`'s collection matrix specifies this per planned drive). Assignment is not deferred until after looking at how a driver's data performs — that would be a leak in spirit even if not in mechanism.
- Prefer growing **train and val** with new identities in most stages; only add to **test** deliberately and rarely, since a test set that keeps growing mid-project makes "the test set" a moving target and complicates comparison to the frozen baseline. If new test identities are added, the old ones are also always kept, and the stage's report shows both the old-test-set and new-full-test-set numbers explicitly.
- Reassigning an identity between splits after it has already been used (e.g. moving a val driver to train) requires the same due-diligence this project already did once for `Vf` — check it doesn't create leakage (`PROJECT_REPORT.md` §2), document why in `DATASET_PROTOCOL.md`'s changelog (§7), and never do it merely because the number looked better one way.

---

## 4. Registry: how identities and drives are tracked

New drives (from the mobile app's recorder, or any future source) need explicit metadata that the raw sensor CSV alone doesn't carry: who was driving, what vehicle, what scenario, and which split it's assigned to. This is `data/wayfinder-drives/registry.json`, created and maintained by the ingestion tool (`src/preprocessing/new_drive_ingest.py`, see `DATA_COLLECTION_PLAN.md` §5 for its interface). Each entry:

```json
{
  "drive_id": "wd-2026-10-03-001",
  "source_csv": "drive_1717000000.csv",
  "driver_id": "driver_03",
  "vehicle_id": "vehicle_02",
  "split": "train",
  "scenario_tags": ["highway", "cruising", "clear_weather"],
  "collected_at": "2026-10-03T14:22:00+05:30",
  "duration_s": 1834.2,
  "distance_km": 41.2,
  "notes": "Free text"
}
```

`split` is set at collection-planning time (per §3's rule), not after the fact. The ingestion tool refuses to silently default a drive to `train` — every drive must have an explicit `split` value, or ingestion fails loudly, so a drive can never end up in the training pool "by accident" because nobody assigned it.

---

## 5. What counts as a valid improvement

A result from a new data-collection stage is only reported as a real improvement over the frozen baseline (`results/metrics/baseline_frozen_2026-09-15/`) if **all** of the following hold. This list exists to operationalize `PROJECT_REPORT.md`'s repeated lesson that plausible-looking numbers can be produced by bugs, luck, or leakage.

1. **No ground truth leaked into inference, filtering, normalization, or the reported evaluation.** (Diagnostic-only ceiling checks like `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` Experiment 4 are fine — but they are never reported as the system's performance, only as an isolated ceiling.)
2. **The evaluation drive(s) were genuinely unseen** — the identity was never in train or val for the model being evaluated, for any stage, ever.
3. **The evaluation pipeline is unchanged** from the frozen baseline's methodology (5-window-averaged outage drift, same window fractions, same `run_ekf_fusion`/`evaluate_system.py` logic) unless a change to that methodology is itself justified by new evidence and documented as such — never changed merely because it moves the metric.
4. **The improvement survives cross-drive evaluation** — i.e. it isn't just one lucky test drive; report the full per-sequence breakdown (as `phase8_full_evaluation.json` already does), not only the mean.
5. **The end-to-end metric improves**, not just the CNN-GRU's own validation RMSE. A better raw model that doesn't move the downstream drift number is not a reportable win (see `DATA_COLLECTION_PLAN.md` §7 for the full metric set that must be tracked together).
6. **Raw model metrics and system metrics are reported separately, never conflated.** "Validation RMSE improved by X%" and "system drift improved by Y%" are two different sentences, always.

If a result fails any of these, it gets reported honestly as "did not establish an improvement" — exactly the standard already applied to the mean-pooling and magnitude-feature experiments in `PROJECT_REPORT.md` §3.2's changelog, both of which looked promising and were reverted once measured properly.

---

## 6. Metric definitions (for consistency across stages)

All defined precisely in existing code — repeated here so a future stage's report can't quietly redefine them:

- **CNN-GRU RMSE (validation)**: RMSE of `predict_speed_sequence()` output vs. `gps_velocity_ms`, on the validation split, matching `results/metrics/phase6_speed_estimator.json`'s definition.
- **CNN-GRU RMSE (unseen-driver)**: same, computed on the test split, as done for `FAILURE_ANALYSIS_AND_NEXT_STEPS.md` Experiment 2. Report alongside validation RMSE, never as a substitute for it.
- **CNN-GRU bias**: mean signed error (`pred - truth`), not just RMSE — a model can have acceptable RMSE with a large systematic bias (as the class-imbalance bug demonstrated).
- **EKF-only diagnostic ceiling**: `run_ekf_fusion` with ground-truth speed substituted for AI speed, exactly as Experiment 4. Diagnostic only (see §5.1).
- **End-to-end system drift**: `evaluate_system.py`'s 5-window-averaged `denied_drift_pct`, mean and per-sequence, against the SIH <10% target.
- **Worst-case error**: in addition to the mean drift, report the max and the 90th-percentile drift across all evaluated windows (not just sequences) — a system that's excellent on typical windows but catastrophic on a tail of them is not SIH-ready even with a good mean. (Not previously tracked; added starting with the next stage's report — see `DATA_COLLECTION_PLAN.md` §7.)

---

## 7. Changelog

- **2026-09-15**: Protocol established. Current splits: train = `Vta`, `Vtb` (Driver E); val = `Vw`, `Vf` (Driver E); test = `S` (Driver A), `Y` (Driver D). Baseline frozen at `results/metrics/baseline_frozen_2026-09-15/`.
