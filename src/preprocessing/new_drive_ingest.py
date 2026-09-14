"""
new_drive_ingest.py — bridges the mobile app's raw drive recordings into the
existing, already-verified training pipeline.

Background: `mobile_app/wayfinder_app/lib/core/navigation_service.dart`
already has a working recorder (`startRecording()`/`stopRecording()`) that
writes `drive_<timestamp>.csv` at ~10 Hz with columns:

    timestamp_iso8601, ax_ms2, ay_ms2, az_ms2, gx_rads, gy_rads, gz_rads,
    gps_lat_deg, gps_lon_deg, gps_speed_ms, gps_heading_deg

Nothing in src/ previously consumed that format — the only loader
(`src/preprocessing/loader.py`) parses the IO-VNBD dataset's specific
V-file (29-column vehicle ECU) / S-file (24-column smartphone) CSV pair.

Rather than teach every downstream consumer (pipeline.py, dataset.py,
inference.py, ekf_fusion.py, classical_ins.py — all already fixed and
verified, see PROJECT_REPORT.md §4) a second data format, this script
converts one app recording into a synthetic V-file/S-file pair in the
exact column layout `load_v_file`/`load_s_file` already parse, written
into a directory tree that mirrors IO-VNBD's own layout:

    data/wayfinder-drives/Synchronised V abd S datasets/
        Categorised IOVNB Dataset/<driver_id> (<vehicle_id>)/<drive_id>/
            V-<drive_id>.csv
            S-<drive_id>.csv

`discover_sequences()` and `load_sequence()` then pick these up with ZERO
code changes, exactly as they already do for IO-VNBD sequences. Point
`discover_sequences()` at `data/wayfinder-drives` (or merge its result
with `discover_sequences('data/IO-VNBD-master')`) to include them.

Fields the app recording cannot provide (vehicle ECU channels: wheel
speed, engine RPM, brake pressure, etc.; magnetometer; GPS accuracy) are
either filled with a documented fallback (see FIELD MAPPING below) or
zero — every such column is UNUSED by the actual training/EKF path
(verified by grepping every `vdf[...]`/`sdf[...]` access site; see
DATA_COLLECTION_PLAN.md §5.2), so this never silently feeds a fabricated
value into the model. The one exception is `odometry_speed_ms`, which is
set equal to GPS speed as an explicit, documented stand-in (there is no
real wheel encoder) — this is only used by the classical-INS *baseline*
comparison, never by the CNN-GRU or EKF path.

This script never assigns a drive to a split on its own — `--split` is
required, per DATASET_PROTOCOL.md's rule that a drive's train/val/test
assignment is decided at collection-planning time, not after the fact.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.preprocessing.loader import V_COLS, S_COLS, WHEEL_RADIUS_M, haversine_m  # noqa: E402

APP_CSV_COLUMNS = [
    'timestamp_iso8601', 'ax_ms2', 'ay_ms2', 'az_ms2',
    'gx_rads', 'gy_rads', 'gz_rads',
    'gps_lat_deg', 'gps_lon_deg', 'gps_speed_ms', 'gps_heading_deg',
]

DRIVES_ROOT = REPO_ROOT / 'data' / 'wayfinder-drives'
CATEG_DIR = DRIVES_ROOT / 'Synchronised V abd S datasets' / 'Categorised IOVNB Dataset'
REGISTRY_PATH = DRIVES_ROOT / 'registry.json'

VALID_SPLITS = ('train', 'val', 'test')


def _load_app_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in APP_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing expected recorder columns {missing}. "
            f"Expected exactly: {APP_CSV_COLUMNS} (see navigation_service.dart startRecording())."
        )
    if len(df) < 60:
        raise ValueError(
            f"{path} has only {len(df)} rows (~{len(df)/10:.0f}s at 10Hz) — "
            f"too short to be a usable drive (need at least a few window_size=50 sample windows)."
        )
    return df


def _build_v_and_s_frames(app_df: pd.DataFrame) -> tuple:
    """Map the app's unified 11-column recording onto the IO-VNBD V_COLS/
    S_COLS layouts. Field-by-field rationale is in the module docstring;
    every field left at a fallback value is one confirmed UNUSED by the
    training/EKF path (see DATA_COLLECTION_PLAN.md §5.2)."""
    # Dart's DateTime.toIso8601String() varies fractional-second precision
    # row to row (drops trailing zeros), so a single fixed strptime format
    # can fail partway through a file — format='ISO8601' handles that.
    ts = pd.to_datetime(app_df['timestamp_iso8601'], utc=True, format='ISO8601')
    t_rel_s = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()

    gps_speed_ms = app_df['gps_speed_ms'].to_numpy(dtype=float)
    gps_velocity_kmh = gps_speed_ms * 3.6
    yaw_rate_degs = np.degrees(app_df['gz_rads'].to_numpy(dtype=float))
    # All four "wheel speeds" set equal so load_v_file's existing mean-of-4
    # derivation reproduces odometry_speed_ms == gps_velocity_ms exactly —
    # a documented GPS-based stand-in, since there is no real wheel encoder.
    wheel_speed_rads = gps_speed_ms / WHEEL_RADIUS_M
    n = len(app_df)
    zeros = np.zeros(n)

    v_data = {
        'gps_satellites': zeros, 'time_of_day_s': t_rel_s,
        'latitude_deg': app_df['gps_lat_deg'].to_numpy(dtype=float),
        'longitude_deg': app_df['gps_lon_deg'].to_numpy(dtype=float),
        'gps_velocity_kmh': gps_velocity_kmh,
        'gps_heading_deg': app_df['gps_heading_deg'].to_numpy(dtype=float),
        'height_km': zeros, 'vertical_vel_kmh': zeros, 'sample_period_s': np.full(n, 0.1),
        'steering_angle_deg': zeros,
        'wheel_speed_fl_rads': wheel_speed_rads, 'wheel_speed_fr_rads': wheel_speed_rads,
        'wheel_speed_rl_rads': wheel_speed_rads, 'wheel_speed_rr_rads': wheel_speed_rads,
        'yaw_rate_degs': yaw_rate_degs, 'vehicle_speed_kmh': gps_velocity_kmh,
        'long_accel_g': zeros, 'lat_accel_g': zeros,
        'handbrake': zeros, 'gear_requested': zeros, 'gear': zeros, 'engine_speed_rpm': zeros,
        'coolant_temp_c': zeros, 'clutch_position': zeros, 'brake_pressure_psi': zeros,
        'brake_position': zeros, 'battery_voltage_v': zeros, 'air_temp_c': zeros,
        'accelerator_pedal': zeros,
    }
    assert list(v_data.keys()) == V_COLS, "V field mapping drifted from loader.py's V_COLS"
    vdf_raw = pd.DataFrame(v_data)

    date_str = ts.dt.strftime('%Y-%m-%d')
    s_data = {
        'gps_latitude_deg': app_df['gps_lat_deg'].to_numpy(dtype=float),
        'gps_longitude_deg': app_df['gps_lon_deg'].to_numpy(dtype=float),
        'gps_altitude_m': zeros, 'gps_speed_kmh': gps_velocity_kmh,
        'gps_accuracy_m': zeros,  # not logged by the recorder yet — see DATA_COLLECTION_PLAN.md §5.3
        'gps_orientation_deg': app_df['gps_heading_deg'].to_numpy(dtype=float),
        'gps_satellites_in_range': zeros,
        'time_since_start_ms': t_rel_s * 1000.0, 'date_str': date_str,
        'accel_x_ms2': app_df['ax_ms2'].to_numpy(dtype=float),
        'accel_y_ms2': app_df['ay_ms2'].to_numpy(dtype=float),
        'accel_z_ms2': app_df['az_ms2'].to_numpy(dtype=float),
        'gravity_x_ms2': zeros, 'gravity_y_ms2': zeros, 'gravity_z_ms2': zeros,  # not logged; unused downstream
        'gyro_x_rads': app_df['gx_rads'].to_numpy(dtype=float),
        'gyro_y_rads': app_df['gy_rads'].to_numpy(dtype=float),
        'gyro_z_rads': app_df['gz_rads'].to_numpy(dtype=float),
        'mag_x_uT': zeros, 'mag_y_uT': zeros, 'mag_z_uT': zeros,  # not logged; unused downstream
        'orient_azimuth_deg': zeros, 'orient_pitch_deg': zeros, 'orient_roll_deg': zeros,  # unused downstream
    }
    assert list(s_data.keys()) == S_COLS, "S field mapping drifted from loader.py's S_COLS"
    sdf_raw = pd.DataFrame(s_data)

    return vdf_raw, sdf_raw


def _write_raw_csv(df: pd.DataFrame, cols: list, path: Path):
    """Write with a (discarded) header row, matching load_v_file/load_s_file's
    `pd.read_csv(path, header=0, names=COLS)` — row 0 is skipped on read,
    so its content doesn't matter, but a real header keeps the file
    self-documenting on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=cols, columns=cols, encoding='latin-1')


def _drive_stats(vdf_raw: pd.DataFrame) -> dict:
    lat = vdf_raw['latitude_deg'].to_numpy()
    lon = vdf_raw['longitude_deg'].to_numpy()
    dist_m = float(np.sum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])))
    duration_s = float(vdf_raw['time_of_day_s'].iloc[-1] - vdf_raw['time_of_day_s'].iloc[0])
    return {'duration_s': duration_s, 'distance_km': dist_m / 1000.0}


def ingest(input_csv: Path, driver_id: str, vehicle_id: str, split: str,
           scenario_tags: list, notes: str, drive_id: str = None) -> dict:
    if split not in VALID_SPLITS:
        raise ValueError(
            f"--split must be one of {VALID_SPLITS} (got {split!r}). "
            f"A drive's split is decided at collection-planning time — see DATASET_PROTOCOL.md §3. "
            f"Refusing to guess."
        )

    app_df = _load_app_csv(input_csv)
    vdf_raw, sdf_raw = _build_v_and_s_frames(app_df)

    if drive_id is None:
        drive_id = f"wd-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{input_csv.stem}"

    seq_dir = CATEG_DIR / f"{driver_id} ({vehicle_id})" / drive_id
    v_path = seq_dir / f"V-{drive_id}.csv"
    s_path = seq_dir / f"S-{drive_id}.csv"
    _write_raw_csv(vdf_raw, V_COLS, v_path)
    _write_raw_csv(sdf_raw, S_COLS, s_path)

    stats = _drive_stats(vdf_raw)
    entry = {
        'drive_id': drive_id,
        'source_csv': str(input_csv),
        'driver_id': driver_id,
        'vehicle_id': vehicle_id,
        'split': split,
        'scenario_tags': scenario_tags or [],
        'collected_at': datetime.now(timezone.utc).isoformat(),
        'duration_s': stats['duration_s'],
        'distance_km': stats['distance_km'],
        'n_rows': int(len(app_df)),
        'v_file': str(v_path),
        's_file': str(s_path),
        'notes': notes or '',
    }

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {'drives': []}
    registry['drives'] = [d for d in registry['drives'] if d['drive_id'] != drive_id]
    registry['drives'].append(entry)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))

    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, type=Path, help="Path to a drive_<timestamp>.csv from the app recorder")
    p.add_argument('--driver-id', required=True, help="Stable identifier for who was driving, e.g. driver_02")
    p.add_argument('--vehicle-id', required=True, help="Stable identifier for the vehicle, e.g. vehicle_02")
    p.add_argument('--split', required=True, choices=VALID_SPLITS,
                   help="Decided at planning time per DATASET_PROTOCOL.md — no default, must be explicit")
    p.add_argument('--scenario', nargs='*', default=[], help="Scenario tags, e.g. --scenario urban stop_and_go")
    p.add_argument('--notes', default='', help="Free-text notes")
    p.add_argument('--drive-id', default=None, help="Override the auto-generated drive id")
    args = p.parse_args()

    entry = ingest(args.input, args.driver_id, args.vehicle_id, args.split,
                    args.scenario, args.notes, args.drive_id)

    print(f"Ingested {entry['drive_id']}")
    print(f"  driver={entry['driver_id']}  vehicle={entry['vehicle_id']}  split={entry['split']}")
    print(f"  duration={entry['duration_s']/60:.1f} min  distance={entry['distance_km']:.2f} km  rows={entry['n_rows']}")
    print(f"  V-file: {entry['v_file']}")
    print(f"  S-file: {entry['s_file']}")
    print(f"  Registry updated: {REGISTRY_PATH}")
    print(f"\n  Run dataset_diagnostics.py before using this in training.")


if __name__ == '__main__':
    main()
