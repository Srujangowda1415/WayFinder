# DATASET.md — IO-VNBD Inspection Report
**WayFinder IDR Navigation System**
**Phase 1 — Dataset Understanding**
**Date:** 2026-09-11
**Status: PASS**

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Name | IO-VNBD — Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning |
| Source | https://github.com/onyekpeu/IO-VNBD |
| Format | CSV (latin-1 encoding) |
| Total synchronised sequences | 71 |
| Total V-file rows | 965,061 |
| Total S-file rows | 964,771 |
| Total synchronised distance | 1,238.6 km |
| Nominal sample rate | 10 Hz |
| Measured mean sample rate | 9.99 Hz |
| Measured median sample rate | 10.00 Hz |
| Missing values | 0 (clean dataset) |

## Recording Setup

- **Vehicle:** Research vehicle (Ford Fiesta) with ECU/OBD sensors
- **Smartphone:** Android smartphone mounted in vehicle
- **Locations:** United Kingdom, Nigeria, France
- **Scenarios:** Traffic, roundabouts, hard braking, motorways, country roads
- **Drivers:** A (S), B (M), D (Y), E (Vw/Vta/Vtb/Vf)
- **Total driving time:** ~40 hours vehicle, ~58 hours smartphone

## File Structure

```
IO-VNBD-master/
├── README.md
├── Synchronised V abd S datasets/
│   └── Categorised IOVNB Dataset/
│       ├── S (Driver A)/     → S1, S2, S3a, S3b, S3c, S4
│       ├── Vf (Driver E)/    → Vfa01, Vfa02
│       ├── Vta (Driver E)/   → Vta01a–Vta30
│       ├── Vtb (Driver E)/   → Vtb01–Vtb12
│       ├── Vw (Driver E)/    → Vw01–Vw17 (with sub-sequences)
│       └── Y (Driver D)/     → Y1
└── Unsynchronised V and S Dataset/
    └── (additional sequences, not synchronised)
```

**File naming:**
- `V-*.csv` = Vehicle ECU data (29 columns)
- `S-*.csv` = Smartphone sensor data (24 columns)
- Each sequence directory contains one V-file and one S-file with matching timestamps

## V-File Columns (Vehicle ECU — 29 columns)

| # | Column Name | Units | Role |
|---|-------------|-------|------|
| 0 | gps_satellites | count | GNSS quality |
| 1 | time_of_day_s | seconds | **TIMESTAMP** |
| 2 | latitude_deg | degrees | **GROUND TRUTH POSITION** |
| 3 | longitude_deg | degrees | **GROUND TRUTH POSITION** |
| 4 | gps_velocity_kmh | km/h | **GROUND TRUTH SPEED** |
| 5 | gps_heading_deg | degrees | GROUND TRUTH HEADING |
| 6 | height_km | km | GPS altitude |
| 7 | vertical_vel_kmh | km/h | Vertical speed |
| 8 | sample_period_s | seconds | ~0.1 → 10 Hz |
| 9 | steering_angle_deg | degrees | Vehicle input |
| 10 | wheel_speed_fl_rads | rad/s | **ODOMETRY FL** |
| 11 | wheel_speed_fr_rads | rad/s | **ODOMETRY FR** |
| 12 | wheel_speed_rl_rads | rad/s | **ODOMETRY RL** |
| 13 | wheel_speed_rr_rads | rad/s | **ODOMETRY RR** |
| 14 | yaw_rate_degs | deg/s | **ANGULAR VELOCITY** |
| 15 | vehicle_speed_kmh | km/h | ECU speed |
| 16 | long_accel_g | g | **LONGITUDINAL ACCEL** |
| 17 | lat_accel_g | g | **LATERAL ACCEL** |
| 18 | handbrake | 0/1 | Motion state |
| 19 | gear_requested | 1-5 | Motion state |
| 20 | gear | 1-5 | Motion state |
| 21 | engine_speed_rpm | rpm | Motion state |
| 22 | coolant_temp_c | °C | Engine state |
| 23 | clutch_position | 0/1 | Motion state |
| 24 | brake_pressure_psi | psi | Braking |
| 25 | brake_position | 0/1 | Braking |
| 26 | battery_voltage_v | V | System |
| 27 | air_temp_c | °C | Environment |
| 28 | accelerator_pedal | 0/1 | Motion state |

## S-File Columns (Smartphone — 24 columns)

| # | Column Name | Units | Role |
|---|-------------|-------|------|
| 0 | gps_latitude_deg | degrees | GPS position |
| 1 | gps_longitude_deg | degrees | GPS position |
| 2 | gps_altitude_m | m | GPS altitude |
| 3 | gps_speed_kmh | km/h | **GPS SPEED (Ground Truth)** |
| 4 | gps_accuracy_m | m | **GNSS QUALITY INDICATOR** |
| 5 | gps_orientation_deg | degrees | GPS heading |
| 6 | gps_satellites_in_range | count | GNSS quality |
| 7 | time_since_start_ms | ms | **TIMESTAMP** |
| 8 | date_str | string | Datetime |
| 9 | accel_x_ms2 | m/s² | **ACCELEROMETER X** |
| 10 | accel_y_ms2 | m/s² | **ACCELEROMETER Y** |
| 11 | accel_z_ms2 | m/s² | **ACCELEROMETER Z** |
| 12 | gravity_x_ms2 | m/s² | Gravity component X |
| 13 | gravity_y_ms2 | m/s² | Gravity component Y |
| 14 | gravity_z_ms2 | m/s² | Gravity component Z |
| 15 | gyro_x_rads | rad/s | **GYROSCOPE X** |
| 16 | gyro_y_rads | rad/s | **GYROSCOPE Y** |
| 17 | gyro_z_rads | rad/s | **GYROSCOPE Z** |
| 18 | mag_x_uT | µT | **MAGNETOMETER X** |
| 19 | mag_y_uT | µT | **MAGNETOMETER Y** |
| 20 | mag_z_uT | µT | **MAGNETOMETER Z** |
| 21 | orient_azimuth_deg | degrees | Fused orientation |
| 22 | orient_pitch_deg | degrees | Fused orientation |
| 23 | orient_roll_deg | degrees | Fused orientation |

## Confirmed Sensors

| Sensor | Source | Available | Units |
|--------|--------|-----------|-------|
| Accelerometer (3-axis) | S-file | ✅ YES | m/s² |
| Gyroscope (3-axis) | S-file | ✅ YES | rad/s |
| Magnetometer (3-axis) | S-file | ✅ YES | µT |
| GNSS position | Both files | ✅ YES | degrees |
| GNSS speed | Both files | ✅ YES | km/h |
| GNSS accuracy | S-file | ✅ YES | m |
| Wheel odometry (4 wheels) | V-file | ✅ YES | rad/s |
| Yaw rate (ECU) | V-file | ✅ YES | deg/s |
| ECU acceleration | V-file | ✅ YES | g |
| Orientation (fused) | S-file | ✅ YES | degrees |

## Coordinate Frame

- **GNSS:** WGS-84 geographic coordinates (lat/lon degrees)
- **Smartphone IMU:** Device body frame (X, Y, Z)
- **Vehicle ECU:** Vehicle body frame (longitudinal/lateral)
- **Note:** Phone-to-vehicle alignment required (Phase 4)

## Ground Truth

| Field | File | Value |
|-------|------|-------|
| Position | V-file | latitude_deg, longitude_deg |
| Speed | V-file | gps_velocity_kmh |
| Heading | V-file | gps_heading_deg |
| Altitude | V-file | height_km |
| Quality | S-file | gps_accuracy_m, gps_satellites_in_range |

## Proposed Train/Val/Test Split

To avoid data leakage, sequences are split at the trajectory level:

| Split | Driver/Category | Sequences | Rationale |
|-------|----------------|-----------|-----------|
| Train | Vta (Driver E), Vtb (Driver E) | ~42 seq | Large set, same driver |
| Validation | Vw (Driver E), Vf (Driver E) | ~22 seq | Same driver, different routes/sessions |
| Test | S (Driver A), Y (Driver D) | ~7 seq | Different drivers — true generalisation |

`Vf (Driver E)` was previously listed under Test, but it is the same driver as
Train/Val (only the vehicle/session label differs), so evaluating on it never
tested cross-driver generalisation — it has been moved to Validation.

**No temporal leakage:** Adjacent time samples within one trajectory are never split across train/test.

## Notes and Caveats

1. **GPS in V-files is the primary ground truth** — higher quality than smartphone GPS
2. **gps_accuracy_m in S-file** can be used for GNSS outage simulation (> threshold = denied)
3. **Wheel speeds are in rad/s** — vehicle wheel radius needed to convert to m/s
4. **ECU accelerations in g** — multiply by 9.81 for m/s²
5. **Smartphone IMU includes gravity** — gravity subtraction needed for pure linear acceleration
6. **orient_* fields** are Android OS fused orientation (Kalman-filtered) — useful for alignment check
7. **All 71 synchronised sequences have 0 missing values**
8. **Typical sequence: 51,746 rows ≈ 5,175 seconds ≈ 86 minutes**

## Phase 1 Exit Criteria — VERIFIED

- [x] Dataset structure documented
- [x] Units verified from actual data
- [x] Ground truth identified (lat/lon, GPS velocity)
- [x] At least one complete trajectory visualised (S1: 51,746 rows, 520 km)
- [x] Data loader works reproducibly (0 missing values, consistent 10 Hz)
- [x] All sensor types confirmed (accel, gyro, mag, GNSS, odometry)

**PHASE 1: PASS**
