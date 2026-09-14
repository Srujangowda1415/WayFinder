"""
evaluate_ins.py — Phase 3: Classical INS Baseline Evaluation

Runs classical INS on test sequences and produces:
1. Ground-truth vs INS trajectory plots
2. Position error vs time
3. Position error vs distance
4. Quantitative metrics table
5. Yaw sign calibration

NOTE: We first need to determine the correct yaw rate sign convention
by empirical calibration on one training sequence.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.preprocessing.loader import discover_sequences, load_sequence
from src.navigation.classical_ins import run_classical_ins

PLOTS_DIR = REPO_ROOT / 'results' / 'plots'
METRICS_DIR = REPO_ROOT / 'results' / 'metrics'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Phase 3 — Classical INS Baseline")
print("=" * 60)

# ─── CALIBRATE YAW SIGN ON ONE TRAINING SEQUENCE ──────────────────────────────

seqs = discover_sequences(str(REPO_ROOT / 'data' / 'IO-VNBD-master'))
train_seqs = [s for s in seqs if 'Vta' in s['sequence'] or 'Vtb' in s['sequence']]
cal_seq = train_seqs[0]

print(f"\n--- Calibrating yaw sign on: {cal_seq['sequence']} ---")
vdf_cal, _ = load_sequence(cal_seq)

# Try both signs and compare final error
for sign in [+1.0, -1.0]:
    res = run_classical_ins(vdf_cal, use_odometry=True, yaw_sign=sign)
    print(f"  Yaw sign={sign:+.0f}: final_error={res['final_error_m']:.1f}m, "
          f"drift={res['drift_pct']:.1f}%, dist={res['total_dist_m']/1000:.1f}km")

# Determine best sign
res_pos = run_classical_ins(vdf_cal, use_odometry=True, yaw_sign=+1.0)
res_neg = run_classical_ins(vdf_cal, use_odometry=True, yaw_sign=-1.0)
best_sign = +1.0 if res_pos['final_error_m'] < res_neg['final_error_m'] else -1.0
print(f"\n  Best yaw sign: {best_sign:+.0f}")

# ─── EVALUATE ON ALL TEST SEQUENCES ───────────────────────────────────────────

test_seqs = [s for s in seqs if s['driver'] in ['S (Driver A)', 'Y (Driver D)', 'Vf (Driver E)']]
print(f"\n--- Evaluating on {len(test_seqs)} test sequences ---")

all_metrics = []
for seq in test_seqs:
    vdf, _ = load_sequence(seq)
    res = run_classical_ins(vdf, use_odometry=True, yaw_sign=best_sign)
    
    row = {
        'sequence': seq['sequence'],
        'driver': seq['driver'],
        'dist_km': res['total_dist_m'] / 1000.0,
        'final_error_m': res['final_error_m'],
        'mean_error_m': res['mean_error_m'],
        'max_error_m': res['max_error_m'],
        'rmse_m': res['rmse_m'],
        'drift_pct': res['drift_pct'],
    }
    all_metrics.append(row)
    print(f"  {seq['sequence']:10s}: dist={row['dist_km']:.1f}km, "
          f"final_err={row['final_error_m']:.1f}m, "
          f"drift={row['drift_pct']:.1f}%, "
          f"RMSE={row['rmse_m']:.1f}m")

# ─── AGGREGATE METRICS ────────────────────────────────────────────────────────

mean_drift = np.mean([m['drift_pct'] for m in all_metrics])
mean_final = np.mean([m['final_error_m'] for m in all_metrics])
mean_rmse  = np.mean([m['rmse_m'] for m in all_metrics])

print(f"\n  AGGREGATE (test set):")
print(f"  Mean drift: {mean_drift:.1f}%")
print(f"  Mean final position error: {mean_final:.1f} m")
print(f"  Mean RMSE: {mean_rmse:.1f} m")

# SIH Target check (< 10% drift)
sih_target = 10.0  # %
sih_pass_count = sum(1 for m in all_metrics if m['drift_pct'] < sih_target)
print(f"\n  SIH Target (<10% drift): {sih_pass_count}/{len(all_metrics)} sequences pass")

# ─── VISUALISATION — REPRESENTATIVE TEST SEQUENCE ─────────────────────────────

# Pick the S1 sequence (Driver A) for visualization
s1_seq = next((s for s in test_seqs if s['sequence'] == 'S1'), test_seqs[0])
vdf_s1, _ = load_sequence(s1_seq)
res_s1 = run_classical_ins(vdf_s1, use_odometry=True, yaw_sign=best_sign)

# Also run without odometry (ECU speed only) for comparison
res_s1_ecu = run_classical_ins(vdf_s1, use_odometry=False, yaw_sign=best_sign)

print(f"\n  Plotting: {s1_seq['sequence']}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Classical INS Baseline — {s1_seq['sequence']}\n"
             f"Distance: {res_s1['total_dist_m']/1000:.1f} km | "
             f"Final Error: {res_s1['final_error_m']:.1f} m | "
             f"Drift: {res_s1['drift_pct']:.1f}%",
             fontsize=13, fontweight='bold')

# (0,0) Trajectory: GT vs INS
ax = axes[0, 0]
ax.plot(res_s1['gt_x']/1000, res_s1['gt_y']/1000, 'b-', lw=1.2,
        label='Ground Truth (GPS)', alpha=0.9)
ax.plot(res_s1['est_x']/1000, res_s1['est_y']/1000, 'r--', lw=1.0,
        label=f'Classical INS (odometry)', alpha=0.8)
ax.plot(0, 0, 'go', ms=10, label='Start')
ax.set_xlabel('Easting (km)')
ax.set_ylabel('Northing (km)')
ax.set_title('Trajectory: GT vs INS')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# (0,1) Position error vs time
ax = axes[0, 1]
times = res_s1['times']
ax.plot(times/60, res_s1['errors_m'], 'r-', lw=0.8, alpha=0.9)
ax.axhline(res_s1['mean_error_m'], color='blue', ls='--', lw=1.5,
           label=f"Mean: {res_s1['mean_error_m']:.1f}m")
ax.set_xlabel('Time (min)')
ax.set_ylabel('Position Error (m)')
ax.set_title('Position Error vs Time')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (0,2) Error vs distance
gt_dists = np.cumsum(np.sqrt(np.diff(res_s1['gt_x'])**2 + np.diff(res_s1['gt_y'])**2))
gt_dists = np.concatenate([[0], gt_dists])
ax = axes[0, 2]
ax.plot(gt_dists/1000, res_s1['errors_m'], 'g-', lw=0.8, alpha=0.9)
ax.axhline(res_s1['total_dist_m'] * 0.10, color='red', ls='--', lw=1.5,
           label='SIH 10% target')
ax.set_xlabel('Distance Travelled (km)')
ax.set_ylabel('Position Error (m)')
ax.set_title('Position Error vs Distance')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (1,0) Speed comparison
vdf_s1['v_time_min'] = vdf_s1['time_s'] / 60.0
ax = axes[1, 0]
ax.plot(vdf_s1['v_time_min'], vdf_s1['gps_velocity_kmh'], 'b-', lw=0.8, label='GPS speed')
ax.plot(vdf_s1['v_time_min'], vdf_s1['vehicle_speed_kmh'], 'r--', lw=0.8, alpha=0.7, label='ECU speed')
odo_kmh = vdf_s1['odometry_speed_ms'] * 3.6
ax.plot(vdf_s1['v_time_min'], odo_kmh, 'g:', lw=0.8, alpha=0.7, label='Odometry speed')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Speed (km/h)')
ax.set_title('Speed Comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (1,1) Yaw rate
ax = axes[1, 1]
ax.plot(vdf_s1['v_time_min'], np.degrees(vdf_s1['yaw_rate_rads']), 'purple', lw=0.8, alpha=0.8)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Yaw Rate (deg/s)')
ax.set_title('ECU Yaw Rate')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', lw=0.5)

# (1,2) Odometry vs ECU INS comparison
ax = axes[1, 2]
ax.plot(res_s1['gt_x']/1000, res_s1['gt_y']/1000, 'b-', lw=1.2, label='GT', alpha=0.9)
ax.plot(res_s1['est_x']/1000, res_s1['est_y']/1000, 'r--', lw=0.8,
        label=f'Odometry ({res_s1["drift_pct"]:.1f}%)', alpha=0.8)
ax.plot(res_s1_ecu['est_x']/1000, res_s1_ecu['est_y']/1000, 'm:', lw=0.8,
        label=f'ECU speed ({res_s1_ecu["drift_pct"]:.1f}%)', alpha=0.8)
ax.set_xlabel('Easting (km)')
ax.set_ylabel('Northing (km)')
ax.set_title('Odometry vs ECU Speed Source')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
out = PLOTS_DIR / 'phase3_classical_ins.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out}")

# ─── METRICS SUMMARY TABLE ────────────────────────────────────────────────────

print("\n  CLASSICAL INS RESULTS TABLE:")
print(f"  {'Sequence':<12} {'Dist(km)':>8} {'FinalErr(m)':>12} {'Drift%':>8} {'RMSE(m)':>8}")
print("  " + "-" * 55)
for m in all_metrics:
    print(f"  {m['sequence']:<12} {m['dist_km']:>8.1f} {m['final_error_m']:>12.1f} "
          f"{m['drift_pct']:>8.1f} {m['rmse_m']:>8.1f}")
print(f"  {'MEAN':<12} {np.mean([m['dist_km'] for m in all_metrics]):>8.1f} "
      f"{mean_final:>12.1f} {mean_drift:>8.1f} {mean_rmse:>8.1f}")

# Save metrics
import json
metrics_out = {
    'phase': 3,
    'method': 'Classical INS (odometry + yaw rate)',
    'yaw_sign': best_sign,
    'sequences': all_metrics,
    'aggregate': {
        'mean_drift_pct': float(mean_drift),
        'mean_final_error_m': float(mean_final),
        'mean_rmse_m': float(mean_rmse),
    },
    'sih_target_10pct_pass_rate': f'{sih_pass_count}/{len(all_metrics)}',
}
with open(METRICS_DIR / 'phase3_classical_ins.json', 'w') as f:
    json.dump(metrics_out, f, indent=2)

print(f"\n  Phase 3 metrics saved.")
sih_ok = mean_drift < 10.0
print(f"\n  SIH Target (<10% drift): {'PASS' if sih_ok else 'NOT YET — AI improvement needed'}")
print(f"\n  PHASE 3 STATUS: PASS (baseline established)")
