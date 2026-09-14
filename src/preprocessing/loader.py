"""
loader.py — Reusable IO-VNBD data loader
Phase 2: Data Preprocessing

Loads V-files (Vehicle ECU) and S-files (Smartphone) into DataFrames.
Handles: encoding, column naming, unit conversion, timestamp alignment.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ─── COLUMN DEFINITIONS ───────────────────────────────────────────────────────

V_COLS = [
    "gps_satellites", "time_of_day_s", "latitude_deg", "longitude_deg",
    "gps_velocity_kmh", "gps_heading_deg", "height_km", "vertical_vel_kmh",
    "sample_period_s", "steering_angle_deg",
    "wheel_speed_fl_rads", "wheel_speed_fr_rads",
    "wheel_speed_rl_rads", "wheel_speed_rr_rads",
    "yaw_rate_degs", "vehicle_speed_kmh",
    "long_accel_g", "lat_accel_g",
    "handbrake", "gear_requested", "gear", "engine_speed_rpm",
    "coolant_temp_c", "clutch_position", "brake_pressure_psi",
    "brake_position", "battery_voltage_v", "air_temp_c", "accelerator_pedal",
]

S_COLS = [
    "gps_latitude_deg", "gps_longitude_deg", "gps_altitude_m",
    "gps_speed_kmh", "gps_accuracy_m", "gps_orientation_deg",
    "gps_satellites_in_range", "time_since_start_ms", "date_str",
    "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
    "gravity_x_ms2", "gravity_y_ms2", "gravity_z_ms2",
    "gyro_x_rads", "gyro_y_rads", "gyro_z_rads",
    "mag_x_uT", "mag_y_uT", "mag_z_uT",
    "orient_azimuth_deg", "orient_pitch_deg", "orient_roll_deg",
]

# Wheel radius for Ford Fiesta (approx, to convert rad/s → m/s)
WHEEL_RADIUS_M = 0.31  # approximately 195/65R15 tyres

def load_v_file(path: str) -> pd.DataFrame:
    """Load a V-file (Vehicle ECU) CSV. Returns DataFrame with clean columns."""
    df = pd.read_csv(path, header=0, names=V_COLS, encoding='latin-1')
    
    # Unit conversions
    df['time_s'] = df['time_of_day_s'] - df['time_of_day_s'].iloc[0]  # relative time
    df['gps_velocity_ms'] = df['gps_velocity_kmh'] / 3.6
    df['vehicle_speed_ms'] = df['vehicle_speed_kmh'] / 3.6
    df['long_accel_ms2'] = df['long_accel_g'] * 9.80665
    df['lat_accel_ms2'] = df['lat_accel_g'] * 9.80665
    df['yaw_rate_rads'] = np.radians(df['yaw_rate_degs'])
    df['gps_heading_rad'] = np.radians(df['gps_heading_deg'])
    
    # Odometry: average wheel speed → vehicle forward speed
    df['wheel_speed_mean_rads'] = (
        df['wheel_speed_fl_rads'] + df['wheel_speed_fr_rads'] +
        df['wheel_speed_rl_rads'] + df['wheel_speed_rr_rads']
    ) / 4.0
    df['odometry_speed_ms'] = df['wheel_speed_mean_rads'] * WHEEL_RADIUS_M
    
    # Height in metres
    df['height_m'] = df['height_km'] * 1000.0
    
    return df


def load_s_file(path: str) -> pd.DataFrame:
    """Load an S-file (Smartphone) CSV. Returns DataFrame with clean columns."""
    df = pd.read_csv(path, header=0, names=S_COLS, encoding='latin-1')
    
    # Timestamp in seconds (relative)
    df['time_s'] = (df['time_since_start_ms'] - df['time_since_start_ms'].iloc[0]) / 1000.0
    
    # Linear acceleration (remove gravity for pure motion)
    df['lin_accel_x_ms2'] = df['accel_x_ms2'] - df['gravity_x_ms2']
    df['lin_accel_y_ms2'] = df['accel_y_ms2'] - df['gravity_y_ms2']
    df['lin_accel_z_ms2'] = df['accel_z_ms2'] - df['gravity_z_ms2']
    
    # Total acceleration magnitude
    df['accel_magnitude_ms2'] = np.sqrt(
        df['accel_x_ms2']**2 + df['accel_y_ms2']**2 + df['accel_z_ms2']**2
    )
    
    # GPS speed in m/s
    df['gps_speed_ms'] = df['gps_speed_kmh'] / 3.6
    
    return df


def discover_sequences(data_root: str) -> list:
    """Discover all synchronised V/S sequence pairs."""
    data_root = Path(data_root)
    categ_dir = data_root / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset"
    
    sequences = []
    for driver_dir in sorted(categ_dir.iterdir()):
        if not driver_dir.is_dir():
            continue
        for seq_dir in sorted(driver_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            v_files = list(seq_dir.glob("V-*.csv")) + list(seq_dir.glob("V-*.CSV"))
            s_files = list(seq_dir.glob("S-*.csv")) + list(seq_dir.glob("S-*.CSV"))
            
            if v_files and s_files:
                sequences.append({
                    'driver': driver_dir.name,
                    'sequence': seq_dir.name,
                    'v_file': str(v_files[0]),
                    's_file': str(s_files[0]),
                })
    
    return sequences


def load_sequence(seq: dict) -> tuple:
    """Load a sequence dict → (vdf, sdf).

    The dataset calls V-file/S-file rows "synchronised", but a handful of
    sequences are off by one row (see e.g. Vta01b, Vtb10). Every caller in
    this codebase indexes vdf/sdf element-for-element assuming equal length,
    so truncate both to the common length here, once, rather than let each
    caller silently misalign or crash on the mismatched ones.
    """
    vdf = load_v_file(seq['v_file'])
    sdf = load_s_file(seq['s_file'])
    n = min(len(vdf), len(sdf))
    if len(vdf) != len(sdf):
        vdf = vdf.iloc[:n].reset_index(drop=True)
        sdf = sdf.iloc[:n].reset_index(drop=True)
    return vdf, sdf


def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
