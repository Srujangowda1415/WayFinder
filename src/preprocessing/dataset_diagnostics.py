"""
dataset_diagnostics.py — pre-training data-quality report.

Run this BEFORE retraining on any new data (per DATA_COLLECTION_PLAN.md §6).
Its purpose is narrow and deliberate: make it impossible to accidentally
claim "we added more/diverse data" when what was actually added is
redundant, contaminated across splits, or has quality problems that were
never checked.

Usage:
    python3 src/preprocessing/dataset_diagnostics.py \
        --data-root data/IO-VNBD-master \
        --data-root data/wayfinder-drives \
        --out results/metrics/dataset_diagnostics_report.md

Combines sequences from every --data-root given. For `data/IO-VNBD-master`,
split assignment reuses `pipeline.py`'s existing, unmodified
TRAIN_DRIVERS/VAL_DRIVERS/TEST_DRIVERS lists. For any other root (e.g.
`data/wayfinder-drives`), split assignment comes from that root's
`registry.json` (written by `new_drive_ingest.py`) — never guessed,
never defaulted.

This module does not modify pipeline.py, loader.py, dataset.py,
train_phase6.py, inference.py, or ekf_fusion.py. It is a read-only,
additive report generator that sits alongside them.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.preprocessing.loader import (  # noqa: E402
    discover_sequences, load_v_file, load_s_file, haversine_m,
)
from src.preprocessing.alignment import estimate_alignment_matrix, apply_alignment  # noqa: E402
from src.preprocessing.pipeline import TRAIN_DRIVERS, VAL_DRIVERS, TEST_DRIVERS  # noqa: E402

SPEED_BIN_EDGES = [0, 0.5, 2, 5, 10, 20, 1000]
SPEED_BIN_LABELS = ['0-0.5', '0.5-2', '2-5', '5-10', '10-20', '20+']


# ─── Discovery + split assignment ─────────────────────────────────────────────

def load_all_sequences_with_splits(data_roots: list) -> tuple:
    """
    Returns (sequences, warnings). Each sequence dict gets extra keys:
    'root', 'driver_id', 'vehicle_id', 'split'.
    """
    sequences = []
    warnings = []

    for root in data_roots:
        root = Path(root)
        is_iovnbd = 'IO-VNBD' in root.name
        seqs = discover_sequences(str(root))
        registry = {}
        if not is_iovnbd:
            registry_path = root / 'registry.json'
            if registry_path.exists():
                reg = json.loads(registry_path.read_text())
                registry = {d['drive_id']: d for d in reg.get('drives', [])}
            else:
                warnings.append(f"{root}: no registry.json found — cannot determine split/driver/vehicle for its sequences; skipping.")
                continue

        for s in seqs:
            s = dict(s)
            s['root'] = str(root)
            if is_iovnbd:
                s['driver_id'] = s['driver']
                s['vehicle_id'] = 'n/a (IO-VNBD; not separately identified)'
                if s['driver_id'] in TRAIN_DRIVERS:
                    s['split'] = 'train'
                elif s['driver_id'] in VAL_DRIVERS:
                    s['split'] = 'val'
                elif s['driver_id'] in TEST_DRIVERS:
                    s['split'] = 'test'
                else:
                    s['split'] = None
                    warnings.append(f"IO-VNBD sequence {s['sequence']} (driver {s['driver_id']!r}) is not in any of "
                                     f"pipeline.py's TRAIN/VAL/TEST_DRIVERS — excluded, not defaulted to train.")
            else:
                reg_entry = registry.get(s['sequence'])
                if reg_entry is None:
                    warnings.append(f"{root} sequence {s['sequence']} has no matching registry.json entry — "
                                     f"excluded, split cannot be verified.")
                    s['split'] = None
                    s['driver_id'] = 'unknown'
                    s['vehicle_id'] = 'unknown'
                else:
                    s['split'] = reg_entry['split']
                    s['driver_id'] = reg_entry['driver_id']
                    s['vehicle_id'] = reg_entry['vehicle_id']
            if s['split'] is not None:
                sequences.append(s)

    return sequences, warnings


# ─── Per-sequence summary ──────────────────────────────────────────────────────

def summarize_sequence(seq: dict) -> dict:
    vdf_raw = load_v_file(seq['v_file'])
    sdf_raw = load_s_file(seq['s_file'])
    n_v, n_s = len(vdf_raw), len(sdf_raw)
    n = min(n_v, n_s)

    lat = vdf_raw['latitude_deg'].to_numpy()[:n]
    lon = vdf_raw['longitude_deg'].to_numpy()[:n]
    speed = vdf_raw['gps_velocity_ms'].to_numpy()[:n]
    yaw_rate = np.abs(vdf_raw['yaw_rate_rads'].to_numpy()[:n])
    duration_s = float(vdf_raw['time_s'].iloc[n - 1] - vdf_raw['time_s'].iloc[0]) if n > 1 else 0.0
    dist_m = float(np.sum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]))) if n > 1 else 0.0

    accel_raw = sdf_raw[['accel_x_ms2', 'accel_y_ms2', 'accel_z_ms2']].to_numpy()[:n]
    gyro_raw = sdf_raw[['gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads']].to_numpy()[:n]
    try:
        R = estimate_alignment_matrix(accel_raw, speed)
        aligned_accel, _ = apply_alignment(accel_raw, gyro_raw, R)
        accel_fwd_mag = np.abs(aligned_accel[:, 0])
    except Exception:
        accel_fwd_mag = np.full(n, np.nan)

    nan_counts = {
        'latitude_deg': int(np.isnan(lat).sum()),
        'gps_velocity_ms': int(np.isnan(speed).sum()),
        'accel_x_ms2': int(np.isnan(accel_raw[:, 0]).sum()),
    }

    return {
        'sequence': seq['sequence'],
        'driver_id': seq['driver_id'],
        'vehicle_id': seq['vehicle_id'],
        'split': seq['split'],
        'n_v': n_v, 'n_s': n_s, 'sync_mismatch': n_v != n_s,
        'duration_s': duration_s,
        'distance_km': dist_m / 1000.0,
        'start_lat': float(lat[0]) if n else float('nan'),
        'start_lon': float(lon[0]) if n else float('nan'),
        'speed_mean': float(np.nanmean(speed)) if n else float('nan'),
        'speed_std': float(np.nanstd(speed)) if n else float('nan'),
        'speed_values': speed,
        'accel_fwd_mag_mean': float(np.nanmean(accel_fwd_mag)) if n else float('nan'),
        'yaw_rate_abs_mean': float(np.nanmean(yaw_rate)) if n else float('nan'),
        'nan_counts': nan_counts,
    }


# ─── Aggregate diagnostics ─────────────────────────────────────────────────────

def speed_histogram(speed_values: np.ndarray) -> dict:
    counts, _ = np.histogram(speed_values, bins=SPEED_BIN_EDGES)
    total = counts.sum()
    return {label: (int(c), float(c) / total if total else 0.0) for label, c in zip(SPEED_BIN_LABELS, counts)}


def detect_near_duplicates(summaries: list, dist_km_tol=0.3, dur_frac_tol=0.15, latlon_tol_deg=0.002) -> list:
    """Flags pairs of sequences with suspiciously similar start location,
    duration, and distance — a cheap proxy for accidentally re-ingesting
    the same drive twice, or two drives that are trivially near-identical
    (e.g. the same short commute repeated back-to-back with no other
    variation)."""
    flags = []
    for a, b in combinations(summaries, 2):
        if a['sequence'] == b['sequence']:
            continue
        close_start = (abs(a['start_lat'] - b['start_lat']) < latlon_tol_deg and
                        abs(a['start_lon'] - b['start_lon']) < latlon_tol_deg)
        if not close_start:
            continue
        dur_close = abs(a['duration_s'] - b['duration_s']) < dur_frac_tol * max(a['duration_s'], b['duration_s'], 1)
        dist_close = abs(a['distance_km'] - b['distance_km']) < dist_km_tol + dur_frac_tol * max(a['distance_km'], b['distance_km'])
        if dur_close and dist_close:
            flags.append((a['sequence'], b['sequence']))
    return flags


def check_split_contamination(summaries: list) -> list:
    driver_splits = {}
    for s in summaries:
        driver_splits.setdefault(s['driver_id'], set()).add(s['split'])
    return [(driver, splits) for driver, splits in driver_splits.items() if len(splits) > 1]


def split_distribution_diff(summaries: list) -> str:
    lines = []
    by_split = {}
    for s in summaries:
        by_split.setdefault(s['split'], []).extend(s['speed_values'].tolist())
    stats_by_split = {k: (np.mean(v), np.std(v)) for k, v in by_split.items() if v}
    for split, (mean, std) in stats_by_split.items():
        lines.append(f"  {split:6s}: speed mean={mean:.2f} m/s, std={std:.2f} m/s, n_samples={len(by_split[split])}")

    try:
        from scipy.stats import ks_2samp
        splits_present = [k for k in ('train', 'val', 'test') if k in by_split and by_split[k]]
        for a, b in combinations(splits_present, 2):
            stat, pval = ks_2samp(by_split[a], by_split[b])
            lines.append(f"  KS test {a} vs {b}: statistic={stat:.3f}, p={pval:.2e} "
                          f"({'DIFFERENT distributions' if pval < 0.01 else 'not conclusively different'})")
    except ImportError:
        lines.append("  (scipy not available — skipping KS test; mean/std above still applies)")

    return '\n'.join(lines)


# ─── Report ─────────────────────────────────────────────────────────────────────

def build_report(summaries: list, warnings: list) -> str:
    lines = ['# Dataset Diagnostics Report', '']

    lines.append('## Warnings (sequences excluded or flagged)')
    if warnings:
        for w in warnings:
            lines.append(f"- ⚠️  {w}")
    else:
        lines.append("- None.")
    lines.append('')

    total_hours = sum(s['duration_s'] for s in summaries) / 3600.0
    total_km = sum(s['distance_km'] for s in summaries)
    drivers = sorted(set(s['driver_id'] for s in summaries))
    vehicles = sorted(set(s['vehicle_id'] for s in summaries))
    lines.append('## Overview')
    lines.append(f"- Sequences: {len(summaries)}")
    lines.append(f"- Total duration: {total_hours:.2f} hours")
    lines.append(f"- Total distance: {total_km:.1f} km")
    lines.append(f"- Distinct driver identities: {len(drivers)} — {drivers}")
    lines.append(f"- Distinct vehicle identities: {len(vehicles)} — {vehicles}")
    lines.append('')

    lines.append('## Per-split breakdown')
    lines.append('| Split | Sequences | Hours | km | Drivers | Vehicles |')
    lines.append('|---|---:|---:|---:|---|---|')
    for split in ('train', 'val', 'test'):
        subset = [s for s in summaries if s['split'] == split]
        if not subset:
            continue
        hrs = sum(s['duration_s'] for s in subset) / 3600.0
        km = sum(s['distance_km'] for s in subset)
        drv = sorted(set(s['driver_id'] for s in subset))
        veh = sorted(set(s['vehicle_id'] for s in subset))
        lines.append(f"| {split} | {len(subset)} | {hrs:.2f} | {km:.1f} | {', '.join(drv)} | {', '.join(veh)} |")
    lines.append('')

    lines.append('## Speed distribution (all sequences combined)')
    all_speed = np.concatenate([s['speed_values'] for s in summaries]) if summaries else np.array([])
    hist = speed_histogram(all_speed)
    lines.append('| Bin (m/s) | Count | Fraction |')
    lines.append('|---|---:|---:|')
    for label, (count, frac) in hist.items():
        lines.append(f"| {label} | {count} | {frac:.1%} |")
    lines.append('')

    lines.append('## Distribution differences between splits')
    lines.append(split_distribution_diff(summaries))
    lines.append('')

    lines.append('## Synchronization issues (V-file/S-file row-count mismatch)')
    mismatches = [s for s in summaries if s['sync_mismatch']]
    if mismatches:
        for s in mismatches:
            lines.append(f"- {s['sequence']}: V={s['n_v']} rows, S={s['n_s']} rows "
                          f"(handled by truncation in loader.load_sequence — informational only)")
    else:
        lines.append("- None.")
    lines.append('')

    lines.append('## Missing data')
    any_nan = False
    for s in summaries:
        nz = {k: v for k, v in s['nan_counts'].items() if v > 0}
        if nz:
            any_nan = True
            lines.append(f"- {s['sequence']}: {nz}")
    if not any_nan:
        lines.append("- None found in latitude/velocity/accel_x.")
    lines.append('')

    lines.append('## Train/val/test contamination check')
    contamination = check_split_contamination(summaries)
    if contamination:
        for driver, splits in contamination:
            lines.append(f"- ❌ **LEAKAGE**: driver/vehicle identity {driver!r} appears in multiple splits: {splits}")
    else:
        lines.append("- ✅ No driver/vehicle identity appears in more than one split.")
    lines.append('')

    lines.append('## Duplicate / near-duplicate segment detection')
    dupes = detect_near_duplicates(summaries)
    if dupes:
        for a, b in dupes:
            lines.append(f"- ⚠️  {a} and {b} have similar start location, duration, and distance — check they aren't the same drive ingested twice, or trivially redundant.")
    else:
        lines.append("- None flagged.")
    lines.append('')

    lines.append('## Per-sequence detail')
    lines.append('| Sequence | Split | Driver | Vehicle | Duration (min) | Distance (km) | Mean speed (m/s) | Mean |fwd accel| (m/s²) | Mean |yaw rate| (rad/s) |')
    lines.append('|---|---|---|---|---:|---:|---:|---:|---:|')
    for s in summaries:
        lines.append(f"| {s['sequence']} | {s['split']} | {s['driver_id']} | {s['vehicle_id']} | "
                      f"{s['duration_s']/60:.1f} | {s['distance_km']:.1f} | {s['speed_mean']:.2f} | "
                      f"{s['accel_fwd_mag_mean']:.2f} | {s['yaw_rate_abs_mean']:.3f} |")

    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', action='append', required=True, dest='data_roots',
                    help="Repeatable. e.g. --data-root data/IO-VNBD-master --data-root data/wayfinder-drives")
    p.add_argument('--out', default=None, help="Write report to this path (also always printed to stdout)")
    args = p.parse_args()

    sequences, warnings = load_all_sequences_with_splits(args.data_roots)
    if not sequences:
        print("No usable sequences found across the given data roots. See warnings:")
        for w in warnings:
            print(f"  - {w}")
        sys.exit(1)

    print(f"Summarizing {len(sequences)} sequences...")
    summaries = [summarize_sequence(s) for s in sequences]
    report = build_report(summaries, warnings)

    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
        print(f"\nReport written to {args.out}")


if __name__ == '__main__':
    main()
