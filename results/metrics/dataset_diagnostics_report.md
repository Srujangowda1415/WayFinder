# Dataset Diagnostics Report

## Warnings (sequences excluded or flagged)
- None.

## Overview
- Sequences: 71
- Total duration: 26.97 hours
- Total distance: 1238.5 km
- Distinct driver identities: 6 — ['S (Driver A)', 'Vf (Driver E)', 'Vta (Driver E)', 'Vtb (Driver E)', 'Vw (Driver E)', 'Y (Driver D)']
- Distinct vehicle identities: 1 — ['n/a (IO-VNBD; not separately identified)']

## Per-split breakdown
| Split | Sequences | Hours | km | Drivers | Vehicles |
|---|---:|---:|---:|---|---|
| train | 42 | 6.75 | 311.6 | Vta (Driver E), Vtb (Driver E) | n/a (IO-VNBD; not separately identified) |
| val | 22 | 9.52 | 590.4 | Vf (Driver E), Vw (Driver E) | n/a (IO-VNBD; not separately identified) |
| test | 7 | 10.71 | 336.5 | S (Driver A), Y (Driver D) | n/a (IO-VNBD; not separately identified) |

## Speed distribution (all sequences combined)
| Bin (m/s) | Count | Fraction |
|---|---:|---:|
| 0-0.5 | 111953 | 11.6% |
| 0.5-2 | 35552 | 3.7% |
| 2-5 | 58373 | 6.1% |
| 5-10 | 194438 | 20.2% |
| 10-20 | 333163 | 34.5% |
| 20+ | 231288 | 24.0% |

## Distribution differences between splits
  test  : speed mean=8.80 m/s, std=6.30 m/s, n_samples=379124
  val   : speed mean=17.25 m/s, std=9.84 m/s, n_samples=342588
  train : speed mean=12.82 m/s, std=7.66 m/s, n_samples=243055
  KS test train vs val: statistic=0.299, p=0.00e+00 (DIFFERENT distributions)
  KS test train vs test: statistic=0.276, p=0.00e+00 (DIFFERENT distributions)
  KS test val vs test: statistic=0.483, p=0.00e+00 (DIFFERENT distributions)

## Synchronization issues (V-file/S-file row-count mismatch)
- V-Vfa01: V=11535 rows, S=11486 rows (handled by truncation in loader.load_sequence — informational only)
- V-Vfa02: V=67755 rows, S=67523 rows (handled by truncation in loader.load_sequence — informational only)
- Vta01b: V=953 rows, S=954 rows (handled by truncation in loader.load_sequence — informational only)
- Vtb10: V=195 rows, S=196 rows (handled by truncation in loader.load_sequence — informational only)
- Vw01: V=20475 rows, S=20476 rows (handled by truncation in loader.load_sequence — informational only)
- Vw02: V=52712 rows, S=52713 rows (handled by truncation in loader.load_sequence — informational only)
- Vw04: V=126527 rows, S=126526 rows (handled by truncation in loader.load_sequence — informational only)
- Vw09: V=553 rows, S=552 rows (handled by truncation in loader.load_sequence — informational only)
- Vw15: V=1391 rows, S=1380 rows (handled by truncation in loader.load_sequence — informational only)

## Missing data
- None found in latitude/velocity/accel_x.

## Train/val/test contamination check
- ✅ No driver/vehicle identity appears in more than one split.

## Duplicate / near-duplicate segment detection
- ⚠️  Vw07 and Vw08 have similar start location, duration, and distance — check they aren't the same drive ingested twice, or trivially redundant.

## Per-sequence detail
| Sequence | Split | Driver | Vehicle | Duration (min) | Distance (km) | Mean speed (m/s) | Mean |fwd accel| (m/s²) | Mean |yaw rate| (rad/s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| S1 | test | S (Driver A) | n/a (IO-VNBD; not separately identified) | 86.2 | 38.0 | 7.33 | 0.79 | 0.069 |
| S2 | test | S (Driver A) | n/a (IO-VNBD; not separately identified) | 156.5 | 75.5 | 8.04 | 0.82 | 0.066 |
| S3a | test | S (Driver A) | n/a (IO-VNBD; not separately identified) | 41.0 | 25.9 | 10.54 | 0.65 | 0.058 |
| S3b | test | S (Driver A) | n/a (IO-VNBD; not separately identified) | 11.4 | 3.8 | 5.53 | 0.79 | 0.095 |
| S3c | test | S (Driver A) | n/a (IO-VNBD; not separately identified) | 62.0 | 44.1 | 11.75 | 0.67 | 0.060 |
| S4 | test | S (Driver A) | n/a (IO-VNBD; not separately identified) | 157.7 | 88.4 | 9.35 | 0.75 | 0.049 |
| V-Vfa01 | val | Vf (Driver E) | n/a (IO-VNBD; not separately identified) | 19.1 | 18.8 | 16.34 | 1.36 | 0.037 |
| V-Vfa02 | val | Vf (Driver E) | n/a (IO-VNBD; not separately identified) | 112.5 | 163.1 | 24.15 | 1.50 | 0.016 |
| Vta01a | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 42.8 | 40.7 | 15.86 | 1.98 | 0.044 |
| Vta01b | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 1.6 | 1.3 | 13.21 | 1.55 | 0.038 |
| Vta02 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 18.3 | 11.0 | 10.05 | 1.74 | 0.040 |
| Vta03 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 1.1 | 0.4 | 5.83 | 0.78 | 0.098 |
| Vta04 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 3.0 | 2.0 | 11.14 | 1.67 | 0.040 |
| Vta05 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 0.5 | 0.4 | 11.80 | 1.91 | 0.068 |
| Vta06 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 2.3 | 2.6 | 18.85 | 2.43 | 0.017 |
| Vta07 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 1.4 | 1.5 | 17.94 | 1.91 | 0.075 |
| Vta08 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 6.1 | 3.4 | 9.26 | 1.20 | 0.025 |
| Vta09 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 0.3 | 0.3 | 19.93 | 1.31 | 0.006 |
| Vta10 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 2.5 | 3.7 | 25.00 | 1.87 | 0.030 |
| Vta11 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 0.8 | 0.7 | 14.05 | 2.08 | 0.165 |
| Vta12 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 1.0 | 1.1 | 18.42 | 1.29 | 0.019 |
| Vta13 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 0.7 | 1.0 | 24.26 | 1.63 | 0.040 |
| Vta14 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 4.8 | 5.4 | 18.83 | 1.83 | 0.023 |
| Vta15 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 1.4 | 1.7 | 19.88 | 1.98 | 0.023 |
| Vta16 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 18.9 | 13.7 | 12.04 | 1.30 | 0.044 |
| Vta17 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 7.5 | 4.1 | 9.08 | 1.67 | 0.050 |
| Vta19 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 0.5 | 0.2 | 8.08 | 1.28 | 0.036 |
| Vta20 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 5.4 | 0.4 | 1.17 | 0.51 | 0.012 |
| Vta21 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 3.5 | 2.8 | 13.27 | 1.90 | 0.050 |
| Vta22 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 2.6 | 1.7 | 10.61 | 1.64 | 0.064 |
| Vta23 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 1.8 | 1.1 | 9.81 | 2.37 | 0.065 |
| Vta24 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 1.9 | 0.7 | 5.93 | 0.97 | 0.065 |
| Vta25 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 1.1 | 0.2 | 2.46 | 0.93 | 0.088 |
| Vta26 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 3.2 | 1.0 | 5.17 | 1.41 | 0.028 |
| Vta27 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 4.2 | 3.1 | 12.40 | 1.87 | 0.037 |
| Vta28 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 7.0 | 3.9 | 9.33 | 1.96 | 0.059 |
| Vta29 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 39.5 | 26.1 | 10.99 | 1.68 | 0.055 |
| Vta30 | train | Vta (Driver E) | n/a (IO-VNBD; not separately identified) | 28.6 | 11.7 | 6.85 | 1.04 | 0.035 |
| Vtb01 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 54.1 | 41.9 | 12.80 | 1.40 | 0.043 |
| Vtb02 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 9.5 | 4.3 | 7.55 | 1.37 | 0.049 |
| Vtb03 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 13.7 | 0.7 | 0.82 | 0.45 | 0.019 |
| Vtb04 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 0.9 | 0.2 | 4.34 | 2.11 | 0.050 |
| Vtb05 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 107.3 | 111.4 | 17.32 | 1.23 | 0.022 |
| Vtb06 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 0.8 | 0.9 | 17.59 | 1.62 | 0.009 |
| Vtb07 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 0.8 | 0.7 | 15.67 | 0.90 | 0.031 |
| Vtb08 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 1.1 | 1.3 | 19.39 | 0.84 | 0.005 |
| Vtb09 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 0.8 | 1.0 | 21.41 | 1.69 | 0.005 |
| Vtb10 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 0.3 | 0.2 | 11.59 | 0.93 | 0.066 |
| Vtb11 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 0.6 | 0.7 | 19.39 | 2.35 | 0.007 |
| Vtb12 | train | Vtb (Driver E) | n/a (IO-VNBD; not separately identified) | 0.7 | 0.5 | 11.88 | 1.34 | 0.082 |
| Vw01 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 34.1 | 0.0 | 0.01 | 0.25 | 0.001 |
| Vw02 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 87.9 | 98.4 | 18.70 | 1.55 | 0.025 |
| Vw03 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 6.4 | 4.9 | 12.80 | 1.59 | 0.069 |
| Vw04 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 210.9 | 214.2 | 16.94 | 1.57 | 0.041 |
| Vw05 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 1.7 | 0.7 | 6.58 | 1.69 | 0.122 |
| Vw06 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 2.1 | 1.1 | 8.43 | 1.73 | 0.080 |
| Vw07 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 2.7 | 1.2 | 7.24 | 1.61 | 0.116 |
| Vw08 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 2.5 | 1.1 | 6.94 | 1.85 | 0.104 |
| Vw09 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 0.9 | 0.4 | 7.63 | 1.37 | 0.129 |
| Vw10 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 1.1 | 0.7 | 11.25 | 2.01 | 0.057 |
| Vw11 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 8.2 | 5.8 | 11.89 | 1.42 | 0.044 |
| Vw12 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 1.5 | 2.3 | 25.11 | 0.98 | 0.006 |
| Vw13 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 0.5 | 0.8 | 27.38 | 2.94 | 0.006 |
| Vw14a | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 5.2 | 7.9 | 25.24 | 1.13 | 0.014 |
| Vw14b | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 32.6 | 41.1 | 21.02 | 1.27 | 0.010 |
| Vw14c | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 26.4 | 17.1 | 10.80 | 1.05 | 0.023 |
| Vw15 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 2.3 | 0.0 | 0.02 | 0.21 | 0.001 |
| Vw16a | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 9.8 | 8.5 | 14.42 | 0.98 | 0.021 |
| Vw16b | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 1.9 | 1.9 | 16.79 | 1.18 | 0.019 |
| Vw17 | val | Vw (Driver E) | n/a (IO-VNBD; not separately identified) | 0.5 | 0.5 | 16.40 | 1.60 | 0.013 |
| Y1 | test | Y (Driver D) | n/a (IO-VNBD; not separately identified) | 127.8 | 60.7 | 8.33 | 0.45 | 0.060 |