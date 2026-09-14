#!/usr/bin/env python3
"""
dataset_inspection.py — Phase 1: IO-VNBD Dataset Inspection
WayFinder / IDR Navigation System
Date: 2026-09-11

Discovers and documents:
  - Dataset structure
  - Sensor columns and units
  - Sampling rates
  - Missing values
  - Trajectory statistics
  - Ground truth format
"""

import os
import sys
import glob
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("/Users/srujangowda/Desktop/WayFinder/data/IO-VNBD-master")
OUTPUT_DIR = Path("/Users/srujangowda/Desktop/WayFinder/results")
PLOTS_DIR  = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── COLUMN DEFINITIONS (discovered empirically) ─────────────────────────────

# V-file (Vehicle ECU) columns — 29 columns
V_COLS = [
    "gps_satellites",          # No. GPS satellites
    "time_of_day_s",           # Time since start of day (seconds)
    "latitude_deg",            # Latitude (degrees)
    "longitude_deg",           # Longitude (degrees)
    "gps_velocity_kmh",        # GPS velocity (km/hr)
    "gps_heading_deg",         # GPS heading (degrees)
    "height_km",               # GPS height (km)
    "vertical_vel_kmh",        # Vertical velocity (km/hr)
    "sample_period_s",         # Sample period (seconds) — ~0.1s → 10 Hz
    "steering_angle_deg",      # Steering angle (degrees)
    "wheel_speed_fl_rads",     # Wheel speed front-left (rad/s)
    "wheel_speed_fr_rads",     # Wheel speed front-right (rad/s)
    "wheel_speed_rl_rads",     # Wheel speed rear-left (rad/s)
    "wheel_speed_rr_rads",     # Wheel speed rear-right (rad/s)
    "yaw_rate_degs",           # Yaw rate (deg/s)
    "vehicle_speed_kmh",       # Indicated vehicle speed (km/hr)
    "long_accel_g",            # Longitudinal acceleration (g)
    "lat_accel_g",             # Lateral acceleration (g)
    "handbrake",               # Handbrake (0/1)
    "gear_requested",          # Gear requested (1-5)
    "gear",                    # Gear actual (1-5)
    "engine_speed_rpm",        # Engine speed (rpm)
    "coolant_temp_c",          # Coolant temperature (°C)
    "clutch_position",         # Clutch (0/1)
    "brake_pressure_psi",      # Brake pressure (psi)
    "brake_position",          # Brake position (0/1)
    "battery_voltage_v",       # Battery voltage (V)
    "air_temp_c",              # Air temperature (°C)
    "accelerator_pedal",       # Accelerator pedal (0/1)
]

# S-file (Smartphone) columns — 24 columns
S_COLS = [
    "gps_latitude_deg",        # GPS latitude (degrees)
    "gps_longitude_deg",       # GPS longitude (degrees)
    "gps_altitude_m",          # GPS altitude (m)
    "gps_speed_kmh",           # GPS speed (km/h)
    "gps_accuracy_m",          # GPS accuracy (m)
    "gps_orientation_deg",     # GPS orientation (degrees)
    "gps_satellites_in_range", # GPS satellites in range
    "time_since_start_ms",     # Time since start (ms)
    "date_str",                # Date string
    "accel_x_ms2",             # Accelerometer X (m/s²)
    "accel_y_ms2",             # Accelerometer Y (m/s²)
    "accel_z_ms2",             # Accelerometer Z (m/s²)
    "gravity_x_ms2",           # Gravity X (m/s²)
    "gravity_y_ms2",           # Gravity Y (m/s²)
    "gravity_z_ms2",           # Gravity Z (m/s²)
    "gyro_x_rads",             # Gyroscope X (rad/s)
    "gyro_y_rads",             # Gyroscope Y (rad/s)
    "gyro_z_rads",             # Gyroscope Z (rad/s)
    "mag_x_uT",                # Magnetic field X (µT)
    "mag_y_uT",                # Magnetic field Y (µT)
    "mag_z_uT",                # Magnetic field Z (µT)
    "orient_azimuth_deg",      # Orientation azimuth (degrees)
    "orient_pitch_deg",        # Orientation pitch (degrees)
    "orient_roll_deg",         # Orientation roll (degrees)
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def load_v_file(path):
    """Load a V (vehicle ECU) CSV file."""
    try:
        df = pd.read_csv(path, header=0, names=V_COLS, encoding='latin-1')
        return df
    except Exception as e:
        print(f"  WARN: Could not load V-file {path}: {e}")
        return None

def load_s_file(path):
    """Load an S (smartphone) CSV file."""
    try:
        df = pd.read_csv(path, header=0, names=S_COLS, encoding='latin-1')
        # Convert time_since_start_ms to seconds
        df['time_s'] = df['time_since_start_ms'] / 1000.0
        return df
    except Exception as e:
        print(f"  WARN: Could not load S-file {path}: {e}")
        return None

def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres."""
    R = 6371000.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def compute_trajectory_stats(vdf):
    """Compute trajectory statistics from a V-file DataFrame."""
    stats = {}
    if vdf is None or len(vdf) < 2:
        return stats
    
    # Duration
    stats['n_samples'] = len(vdf)
    stats['duration_s'] = vdf['time_of_day_s'].iloc[-1] - vdf['time_of_day_s'].iloc[0]
    
    # Sample rate
    dt = vdf['time_of_day_s'].diff().dropna()
    stats['sample_period_mean'] = float(dt.mean())
    stats['sample_rate_hz'] = 1.0 / stats['sample_period_mean'] if stats['sample_period_mean'] > 0 else 0
    
    # Distance via GPS
    lats = vdf['latitude_deg'].values
    lons = vdf['longitude_deg'].values
    dists = [haversine_m(lats[i], lons[i], lats[i+1], lons[i+1]) for i in range(len(lats)-1)]
    stats['total_distance_m'] = float(np.sum(dists))
    stats['total_distance_km'] = stats['total_distance_m'] / 1000.0
    
    # Speed
    stats['mean_speed_kmh'] = float(vdf['gps_velocity_kmh'].mean())
    stats['max_speed_kmh'] = float(vdf['gps_velocity_kmh'].max())
    
    # Missing values
    stats['missing_lat'] = int(vdf['latitude_deg'].isna().sum())
    stats['missing_lon'] = int(vdf['longitude_deg'].isna().sum())
    
    return stats

# ─── DISCOVER ALL SEQUENCES ───────────────────────────────────────────────────

print("=" * 70)
print("  IO-VNBD Dataset Inspection — WayFinder IDR System")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Walk the synchronised categorised directory (primary dataset for training)
SYNC_CATEG = DATA_ROOT / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset"

sequences = []

for driver_dir in sorted(SYNC_CATEG.iterdir()):
    if not driver_dir.is_dir():
        continue
    driver_name = driver_dir.name
    
    for seq_dir in sorted(driver_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq_name = seq_dir.name
        
        # Find V and S files
        v_files = list(seq_dir.glob("V-*.csv")) + list(seq_dir.glob("V-*.CSV"))
        s_files = list(seq_dir.glob("S-*.csv")) + list(seq_dir.glob("S-*.CSV"))
        
        if not v_files:
            v_files = [f for f in seq_dir.glob("*.csv") if f.name[0] in 'Vv' and 'S-' not in f.name]
        if not s_files:
            s_files = [f for f in seq_dir.glob("*.csv") if f.name.startswith('S-')]
        
        vf = str(v_files[0]) if v_files else None
        sf = str(s_files[0]) if s_files else None
        
        sequences.append({
            'driver': driver_name,
            'sequence': seq_name,
            'v_file': vf,
            's_file': sf,
        })

print(f"\nTotal synchronised sequences discovered: {len(sequences)}")
print(f"Drivers/categories: {sorted(set(s['driver'] for s in sequences))}")

# ─── PER-SEQUENCE STATISTICS ──────────────────────────────────────────────────

print("\n--- Computing per-sequence statistics ---")

all_stats = []
v_row_counts = []
s_row_counts = []
v_sample_rates = []

for seq in sequences:
    entry = {**seq}
    
    if seq['v_file']:
        vdf = load_v_file(seq['v_file'])
        if vdf is not None and len(vdf) > 10:
            vstats = compute_trajectory_stats(vdf)
            entry.update({f'v_{k}': v for k, v in vstats.items()})
            v_row_counts.append(vstats.get('n_samples', 0))
            if vstats.get('sample_rate_hz', 0) > 0:
                v_sample_rates.append(vstats['sample_rate_hz'])
    
    if seq['s_file']:
        sdf = load_s_file(seq['s_file'])
        if sdf is not None and len(sdf) > 10:
            s_row_counts.append(len(sdf))
    
    all_stats.append(entry)

# ─── SUMMARY STATISTICS ───────────────────────────────────────────────────────

total_v_rows = sum(v_row_counts)
total_s_rows = sum(s_row_counts)
mean_sr = float(np.mean(v_sample_rates)) if v_sample_rates else 0
median_sr = float(np.median(v_sample_rates)) if v_sample_rates else 0

print(f"\n=== DATASET SUMMARY ===")
print(f"Synchronised sequences:        {len(sequences)}")
print(f"V-file (ECU) total rows:       {total_v_rows:,}")
print(f"S-file (smartphone) total rows:{total_s_rows:,}")
print(f"V-file mean sample rate:       {mean_sr:.2f} Hz")
print(f"V-file median sample rate:     {median_sr:.2f} Hz")
print(f"Mean seq length (V, rows):     {np.mean(v_row_counts):.0f}")
print(f"Min / Max seq length (V):      {min(v_row_counts)} / {max(v_row_counts)}")

# Total distance
total_dist_km = sum(s.get('v_total_distance_km', 0) for s in all_stats if isinstance(s.get('v_total_distance_km'), float))
print(f"Total distance (synchronised): {total_dist_km:.1f} km")

# ─── COLUMN INSPECTION REPORT ─────────────────────────────────────────────────

print("\n=== V-FILE COLUMNS (Vehicle ECU, 29 columns) ===")
print(f"{'#':>3}  {'Short Name':<30}  {'Description'}")
print("-" * 70)
raw_vcols = [
    ("gps_satellites",      "No. GPS Satellites"),
    ("time_of_day_s",       "Time since start of day (seconds)"),
    ("latitude_deg",        "Latitude (degrees) — GROUND TRUTH"),
    ("longitude_deg",       "Longitude (degrees) — GROUND TRUTH"),
    ("gps_velocity_kmh",    "GPS velocity (km/hr) — GROUND TRUTH SPEED"),
    ("gps_heading_deg",     "GPS heading (degrees)"),
    ("height_km",           "GPS height (km)"),
    ("vertical_vel_kmh",    "Vertical velocity (km/hr)"),
    ("sample_period_s",     "Sample period (s) ≈ 0.1 → 10 Hz"),
    ("steering_angle_deg",  "Steering angle (degrees)"),
    ("wheel_speed_fl_rads", "Wheel speed FL (rad/s) — ODOMETRY"),
    ("wheel_speed_fr_rads", "Wheel speed FR (rad/s) — ODOMETRY"),
    ("wheel_speed_rl_rads", "Wheel speed RL (rad/s) — ODOMETRY"),
    ("wheel_speed_rr_rads", "Wheel speed RR (rad/s) — ODOMETRY"),
    ("yaw_rate_degs",       "Yaw rate (deg/s) — ANGULAR VELOCITY"),
    ("vehicle_speed_kmh",   "Indicated vehicle speed (km/hr)"),
    ("long_accel_g",        "Longitudinal acceleration (g) — IMU"),
    ("lat_accel_g",         "Lateral acceleration (g) — IMU"),
    ("handbrake",           "Handbrake (0/1)"),
    ("gear_requested",      "Gear requested (1-5)"),
    ("gear",                "Gear actual (1-5)"),
    ("engine_speed_rpm",    "Engine speed (rpm)"),
    ("coolant_temp_c",      "Coolant temperature (°C)"),
    ("clutch_position",     "Clutch (0/1)"),
    ("brake_pressure_psi",  "Brake pressure (psi)"),
    ("brake_position",      "Brake position (0/1)"),
    ("battery_voltage_v",   "Battery voltage (V)"),
    ("air_temp_c",          "Air temperature (°C)"),
    ("accelerator_pedal",   "Accelerator pedal position (0/1)"),
]
for i, (name, desc) in enumerate(raw_vcols):
    print(f"{i:>3}  {name:<30}  {desc}")

print(f"\n=== S-FILE COLUMNS (Smartphone, 24 columns) ===")
raw_scols = [
    ("gps_latitude_deg",        "GPS Latitude (degrees)"),
    ("gps_longitude_deg",       "GPS Longitude (degrees)"),
    ("gps_altitude_m",          "GPS Altitude (m)"),
    ("gps_speed_kmh",           "GPS Speed (km/h) — GROUND TRUTH SPEED"),
    ("gps_accuracy_m",          "GPS Accuracy (m) — QUALITY INDICATOR"),
    ("gps_orientation_deg",     "GPS Orientation (degrees)"),
    ("gps_satellites_in_range", "GPS Satellites in range"),
    ("time_since_start_ms",     "Time since start (ms)"),
    ("date_str",                "Date-time string"),
    ("accel_x_ms2",             "Accelerometer X (m/s²) — IMU"),
    ("accel_y_ms2",             "Accelerometer Y (m/s²) — IMU"),
    ("accel_z_ms2",             "Accelerometer Z (m/s²) — IMU"),
    ("gravity_x_ms2",           "Gravity X (m/s²)"),
    ("gravity_y_ms2",           "Gravity Y (m/s²)"),
    ("gravity_z_ms2",           "Gravity Z (m/s²)"),
    ("gyro_x_rads",             "Gyroscope X (rad/s)"),
    ("gyro_y_rads",             "Gyroscope Y (rad/s)"),
    ("gyro_z_rads",             "Gyroscope Z (rad/s)"),
    ("mag_x_uT",                "Magnetic Field X (µT) — MAGNETOMETER"),
    ("mag_y_uT",                "Magnetic Field Y (µT) — MAGNETOMETER"),
    ("mag_z_uT",                "Magnetic Field Z (µT) — MAGNETOMETER"),
    ("orient_azimuth_deg",      "Orientation Azimuth (degrees)"),
    ("orient_pitch_deg",        "Orientation Pitch (degrees)"),
    ("orient_roll_deg",         "Orientation Roll (degrees)"),
]
for i, (name, desc) in enumerate(raw_scols):
    print(f"{i:>3}  {name:<28}  {desc}")

# ─── LOAD A REPRESENTATIVE SEQUENCE FOR FULL INSPECTION ──────────────────────

print("\n--- Loading representative sequence for detailed analysis ---")

# Pick a well-formed sequence
rep_seq = None
for s in all_stats:
    if s.get('v_total_distance_km', 0) > 5 and s.get('s_file'):
        rep_seq = s
        break

if rep_seq is None:
    rep_seq = all_stats[0]

print(f"Representative sequence: {rep_seq['driver']} / {rep_seq['sequence']}")
vdf_rep = load_v_file(rep_seq['v_file'])
sdf_rep = load_s_file(rep_seq['s_file'])

print(f"V-file rows: {len(vdf_rep)}")
print(f"S-file rows: {len(sdf_rep)}")

# Missing value analysis
print("\n--- V-file missing values ---")
print(vdf_rep.isnull().sum().to_string())

print("\n--- S-file missing values ---")
print(sdf_rep.isnull().sum().to_string())

# ─── VISUALISATIONS ───────────────────────────────────────────────────────────

print("\n--- Generating plots ---")

# 1. Ground truth trajectory plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"IO-VNBD Ground Truth Trajectory\n{rep_seq['driver']} / {rep_seq['sequence']}", 
             fontsize=13, fontweight='bold')

# V-file trajectory (ECU GPS)
ax1 = axes[0]
ax1.plot(vdf_rep['longitude_deg'], vdf_rep['latitude_deg'], 
         'b-', linewidth=0.8, alpha=0.8, label='V-file (ECU GPS)')
ax1.plot(vdf_rep['longitude_deg'].iloc[0], vdf_rep['latitude_deg'].iloc[0],
         'go', markersize=10, label='Start')
ax1.plot(vdf_rep['longitude_deg'].iloc[-1], vdf_rep['latitude_deg'].iloc[-1],
         'rs', markersize=10, label='End')
ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax1.set_title('V-file ECU GPS Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)

# S-file trajectory (Smartphone GPS)
ax2 = axes[1]
valid_s = sdf_rep[sdf_rep['gps_latitude_deg'].notna() & (sdf_rep['gps_latitude_deg'] != 0)]
if len(valid_s) > 10:
    ax2.plot(valid_s['gps_longitude_deg'], valid_s['gps_latitude_deg'],
             'r-', linewidth=0.8, alpha=0.8, label='S-file (Smartphone GPS)')
    ax2.plot(valid_s['gps_longitude_deg'].iloc[0], valid_s['gps_latitude_deg'].iloc[0],
             'go', markersize=10, label='Start')
    ax2.plot(valid_s['gps_longitude_deg'].iloc[-1], valid_s['gps_latitude_deg'].iloc[-1],
             'bs', markersize=10, label='End')
ax2.set_xlabel('Longitude (°)')
ax2.set_ylabel('Latitude (°)')
ax2.set_title('S-file Smartphone GPS Trajectory')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out1 = PLOTS_DIR / "phase1_trajectory.png"
plt.savefig(out1, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out1}")

# 2. Sensor time series plot
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle(f"IO-VNBD Sensor Time Series — {rep_seq['sequence']}", 
             fontsize=13, fontweight='bold')

t_v = vdf_rep['time_of_day_s'] - vdf_rep['time_of_day_s'].iloc[0]
t_s = sdf_rep['time_s'] - sdf_rep['time_s'].iloc[0]

# Row 0: Speed
axes[0,0].plot(t_v, vdf_rep['gps_velocity_kmh'], 'b-', linewidth=0.8, label='GPS velocity')
axes[0,0].plot(t_v, vdf_rep['vehicle_speed_kmh'], 'r--', linewidth=0.8, alpha=0.7, label='ECU speed')
axes[0,0].set_ylabel('Speed (km/h)')
axes[0,0].set_title('Vehicle Speed (V-file)')
axes[0,0].legend(fontsize=8)
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(t_s, sdf_rep['gps_speed_kmh'], 'r-', linewidth=0.8)
axes[0,1].set_ylabel('Speed (km/h)')
axes[0,1].set_title('Smartphone GPS Speed (S-file)')
axes[0,1].grid(True, alpha=0.3)

# Row 1: Accelerometer
axes[1,0].plot(t_v, vdf_rep['long_accel_g'], 'g-', linewidth=0.8, label='Longitudinal (g)')
axes[1,0].plot(t_v, vdf_rep['lat_accel_g'], 'm-', linewidth=0.8, alpha=0.7, label='Lateral (g)')
axes[1,0].set_ylabel('Acceleration (g)')
axes[1,0].set_title('ECU Acceleration (V-file)')
axes[1,0].legend(fontsize=8)
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(t_s, sdf_rep['accel_x_ms2'], 'r-', lw=0.6, label='X')
axes[1,1].plot(t_s, sdf_rep['accel_y_ms2'], 'g-', lw=0.6, label='Y')
axes[1,1].plot(t_s, sdf_rep['accel_z_ms2'], 'b-', lw=0.6, label='Z')
axes[1,1].set_ylabel('Acceleration (m/s²)')
axes[1,1].set_title('Smartphone Accelerometer (S-file)')
axes[1,1].legend(fontsize=8)
axes[1,1].grid(True, alpha=0.3)

# Row 2: Gyroscope / Yaw
axes[2,0].plot(t_v, vdf_rep['yaw_rate_degs'], 'b-', linewidth=0.8)
axes[2,0].set_ylabel('Yaw Rate (deg/s)')
axes[2,0].set_title('ECU Yaw Rate (V-file)')
axes[2,0].grid(True, alpha=0.3)

axes[2,1].plot(t_s, sdf_rep['gyro_x_rads'], 'r-', lw=0.6, label='X')
axes[2,1].plot(t_s, sdf_rep['gyro_y_rads'], 'g-', lw=0.6, label='Y')
axes[2,1].plot(t_s, sdf_rep['gyro_z_rads'], 'b-', lw=0.6, label='Z')
axes[2,1].set_ylabel('Gyro (rad/s)')
axes[2,1].set_title('Smartphone Gyroscope (S-file)')
axes[2,1].legend(fontsize=8)
axes[2,1].grid(True, alpha=0.3)

# Row 3: Magnetometer / Heading
axes[3,0].plot(t_v, vdf_rep['gps_heading_deg'], 'b-', linewidth=0.8)
axes[3,0].set_ylabel('Heading (°)')
axes[3,0].set_title('ECU GPS Heading (V-file)')
axes[3,0].grid(True, alpha=0.3)
axes[3,0].set_xlabel('Time (s)')

axes[3,1].plot(t_s, sdf_rep['mag_x_uT'], 'r-', lw=0.6, label='MagX (µT)')
axes[3,1].plot(t_s, sdf_rep['mag_y_uT'], 'g-', lw=0.6, label='MagY (µT)')
axes[3,1].plot(t_s, sdf_rep['mag_z_uT'], 'b-', lw=0.6, label='MagZ (µT)')
axes[3,1].set_ylabel('Magnetic Field (µT)')
axes[3,1].set_title('Smartphone Magnetometer (S-file)')
axes[3,1].legend(fontsize=8)
axes[3,1].grid(True, alpha=0.3)
axes[3,1].set_xlabel('Time (s)')

plt.tight_layout()
out2 = PLOTS_DIR / "phase1_sensor_timeseries.png"
plt.savefig(out2, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out2}")

# 3. Wheel speed & odometry
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"IO-VNBD Wheel Speed & Odometry — {rep_seq['sequence']}", fontsize=13, fontweight='bold')

axes[0,0].plot(t_v, vdf_rep['wheel_speed_fl_rads'], 'b-', lw=0.8)
axes[0,0].set_title('Front-Left Wheel Speed (rad/s)')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_ylabel('Speed (rad/s)')

axes[0,1].plot(t_v, vdf_rep['wheel_speed_fr_rads'], 'r-', lw=0.8)
axes[0,1].set_title('Front-Right Wheel Speed (rad/s)')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(t_v, vdf_rep['wheel_speed_rl_rads'], 'g-', lw=0.8)
axes[1,0].set_title('Rear-Left Wheel Speed (rad/s)')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_ylabel('Speed (rad/s)')
axes[1,0].set_xlabel('Time (s)')

axes[1,1].plot(t_v, vdf_rep['wheel_speed_rr_rads'], 'm-', lw=0.8)
axes[1,1].set_title('Rear-Right Wheel Speed (rad/s)')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_xlabel('Time (s)')

plt.tight_layout()
out3 = PLOTS_DIR / "phase1_wheel_speed.png"
plt.savefig(out3, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out3}")

# 4. Sampling rate distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('IO-VNBD Sampling Rate Analysis', fontsize=13, fontweight='bold')

v_dts = (vdf_rep['time_of_day_s'].diff().dropna() * 1000).values  # ms
axes[0].hist(v_dts, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].axvline(100, color='red', linestyle='--', linewidth=1.5, label='100 ms (10 Hz)')
axes[0].set_xlabel('Sample Interval (ms)')
axes[0].set_ylabel('Count')
axes[0].set_title('V-file Sample Interval Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

s_dts = (sdf_rep['time_s'].diff().dropna() * 1000).values  # ms
axes[1].hist(s_dts, bins=50, color='coral', edgecolor='white', linewidth=0.5)
axes[1].axvline(100, color='blue', linestyle='--', linewidth=1.5, label='100 ms (10 Hz)')
axes[1].set_xlabel('Sample Interval (ms)')
axes[1].set_ylabel('Count')
axes[1].set_title('S-file Sample Interval Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out4 = PLOTS_DIR / "phase1_sampling_rate.png"
plt.savefig(out4, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out4}")

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  PHASE 1 INSPECTION COMPLETE")
print("=" * 70)
print(f"  Sequences:      {len(sequences)}")
print(f"  Sample rate:    {mean_sr:.2f} Hz (should be ~10 Hz)")
print(f"  Total distance: {total_dist_km:.1f} km (synchronised)")
print(f"  Ground truth:   Latitude / Longitude in V-file (ECU GPS)")
print(f"  IMU source:     S-file Accelerometer, Gyroscope, Magnetometer")
print(f"  Odometry:       V-file Wheel Speeds (4 wheels)")
print(f"  Plots saved:    {PLOTS_DIR}")
print()

# Save a JSON summary
summary = {
    'inspection_date': datetime.now().isoformat(),
    'total_sequences': len(sequences),
    'v_file_columns': V_COLS,
    'v_file_n_cols': len(V_COLS),
    's_file_columns': S_COLS,
    's_file_n_cols': len(S_COLS),
    'nominal_sample_rate_hz': 10.0,
    'measured_sample_rate_hz': mean_sr,
    'total_synchronised_distance_km': total_dist_km,
    'ground_truth_fields': ['latitude_deg', 'longitude_deg', 'gps_velocity_kmh', 'gps_heading_deg'],
    'imu_fields_smartphone': ['accel_x_ms2', 'accel_y_ms2', 'accel_z_ms2', 'gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads'],
    'magnetometer_available': True,
    'odometry_available': True,
    'encoding': 'latin-1',
    'representative_sequence': f"{rep_seq['driver']}/{rep_seq['sequence']}",
}

import json
summary_path = OUTPUT_DIR / "metrics" / "dataset_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Summary saved: {summary_path}")
print("\n  PHASE 1 STATUS: PASS")
