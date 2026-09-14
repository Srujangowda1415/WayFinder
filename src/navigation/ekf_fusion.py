"""
ekf_fusion.py — Phase 7: Extended Kalman Filter for GNSS/INS Fusion
WayFinder / IDR Navigation System

State vector: [x, y, heading, speed, heading_bias, speed_bias]
              [0  1     2       3         4              5   ]

x, y: Easting, Northing (m) in local ENU frame
heading: vehicle heading (rad, N=0, clockwise)
speed: forward speed (m/s)
heading_bias: bias on yaw rate sensor (rad/s)
speed_bias: bias on speed sensor (m/s)

Measurement models:
  - GNSS: updates [x, y] directly
  - Speed odometry / AI speed: updates [speed]
  - NHC: zero lateral velocity constraint (y_body = 0)
  - Zero velocity update (ZUPT): when stationary, constrains speed=0

Reference: Titterton & Weston "Strapdown Inertial Navigation Technology"
           Farrell "Aided Navigation" ch. 10
"""

import numpy as np


class EKF_INS:
    """
    6-DOF Extended Kalman Filter for ground vehicle navigation.
    
    Features:
    - GNSS position updates (when available)
    - Speed measurement updates (odometry or AI-estimated)
    - Non-Holonomic Constraint (NHC) - vehicles cannot move laterally
    - Zero Velocity Update (ZUPT) - reduces drift while stationary
    - Heading bias estimation
    - Adaptive process noise
    """

    def __init__(self, init_lat, init_lon, init_heading, init_speed=0.0):
        # Reference origin (ENU frame)
        self.lat0 = init_lat
        self.lon0 = init_lon
        lat0_rad = np.radians(init_lat)
        self.R_earth = 6371000.0
        self.m_per_deg_lat = self.R_earth * np.pi / 180.0
        self.m_per_deg_lon = self.m_per_deg_lat * np.cos(lat0_rad)

        # State: [x, y, heading, speed, heading_bias, speed_bias]
        self.x = np.array([0.0, 0.0, init_heading, init_speed, 0.0, 0.0])

        # Initial covariance
        self.P = np.diag([
            100.0,   # x (m²)
            100.0,   # y (m²)
            0.1,     # heading (rad²) - ~18° initial uncertainty
            4.0,     # speed (m²/s²) - ~2 m/s initial uncertainty
            1e-6,    # heading_bias (rad²/s²)
            0.25,    # speed_bias (m²/s²)
        ])

        # Process noise (tuned for 10 Hz)
        self.Q = np.diag([
            0.01,    # x noise
            0.01,    # y noise
            1e-4,    # heading noise
            0.01,    # speed noise
            1e-8,    # heading_bias noise (slowly varying)
            1e-6,    # speed_bias noise (slowly varying)
        ])

        # Measurement noise
        self.R_gnss = np.diag([25.0, 25.0])  # GNSS: ~5m accuracy
        self.R_speed = np.array([[0.25]])     # Speed: ~0.5 m/s accuracy
        self.R_nhc = np.array([[0.01]])       # NHC: very tight lateral constraint
        self.R_zupt = np.array([[0.01]])      # ZUPT: very tight zero-speed

        # History
        self.trajectory = []
        self.timestamps = []
        self.innovations = []

    def predict(self, dt, yaw_rate):
        """
        Propagate state using vehicle kinematics.

        Uses ONLY the speed already in the state (self.x[3]) — it must be
        corrected by update_speed()/update_zupt() BEFORE predict() is called
        each tick, not the other way around. The previous version took an
        external `forward_speed` argument and blended 10% of it into the
        state on every call; callers (see run_ekf_fusion) were passing the
        ground-truth GPS speed for that argument on *every* tick, including
        inside a simulated GNSS-denied window — silently leaking the answer
        into what was supposed to be a GNSS-denied dead-reckoning test, and
        in the wrong order relative to the speed update (this is the same
        "predict() runs before speed is validated" bug the mobile app
        (ekf_navigation.dart) already found and fixed — see
        DRIVE_MODE_FIX_REPORT.md RC-2 — but this Python evaluation harness
        never got the same fix).

        Args:
            dt: time step (s)
            yaw_rate: measured yaw rate (rad/s) — positive = left turn
        """
        x, y, psi, v, b_psi, b_v = self.x

        psi_dot = yaw_rate - b_psi
        psi_new = psi + psi_dot * dt
        v_new = v  # constant-velocity model; already corrected before predict()

        dx = v_new * np.sin(psi_new) * dt
        dy = v_new * np.cos(psi_new) * dt

        self.x = np.array([
            x + dx,
            y + dy,
            psi_new % (2 * np.pi),
            v_new,
            b_psi,
            b_v,
        ])

        # Jacobian F = d(f)/d(x)
        F = np.eye(6)
        F[0, 2] = v_new * np.cos(psi_new) * dt   # dx/dpsi
        F[0, 3] = np.sin(psi_new) * dt            # dx/dv
        F[0, 4] = -v_new * np.cos(psi_new) * dt  # dx/db_psi
        F[1, 2] = -v_new * np.sin(psi_new) * dt  # dy/dpsi
        F[1, 3] = np.cos(psi_new) * dt            # dy/dv
        F[1, 4] = v_new * np.sin(psi_new) * dt   # dy/db_psi
        F[2, 4] = -dt                              # dpsi/db_psi
        F[3, 5] = 0.0                              # speed is constant, no dependency on b_v in predict

        # Covariance predict
        self.P = F @ self.P @ F.T + self.Q

    def update_gnss(self, lat, lon):
        """
        GNSS position update.

        Args:
            lat, lon: measured GPS position (degrees)
        """
        # Convert GPS to ENU metres
        meas_x = (lon - self.lon0) * self.m_per_deg_lon
        meas_y = (lat - self.lat0) * self.m_per_deg_lat

        H = np.zeros((2, 6))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y

        z = np.array([meas_x, meas_y])
        z_hat = H @ self.x
        innovation = z - z_hat

        S = H @ self.P @ H.T + self.R_gnss
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ H) @ self.P
        self.innovations.append(float(np.linalg.norm(innovation)))

    def update_speed(self, speed_meas):
        """Speed measurement update (from odometry or AI)."""
        H = np.zeros((1, 6))
        H[0, 3] = 1.0  # speed state

        z = np.array([speed_meas])
        z_hat = H @ self.x
        innovation = z - z_hat

        S = H @ self.P @ H.T + self.R_speed
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ innovation).flatten()
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_nhc(self):
        """
        Non-Holonomic Constraint: vehicle cannot move sideways.
        Constrains body-frame lateral (right) velocity: v_right = 0, where

          v_right = vx*cos(psi) - vy*sin(psi)

        Since this state's own forward velocity is vx = v*sin(psi),
        vy = v*cos(psi), substituting gives v_right ≡ 0 for every psi and v:
        this state has no independent lateral-velocity degree of freedom, so
        the correctly-derived constraint (and its Jacobian) is an exact
        no-op here (zero H → zero Kalman gain).

        (The previous version paired the sin/cos terms the wrong way —
        `-vx*sin(psi) + vy*cos(psi)` = `v*cos(2*psi)` — which is generally
        non-zero and was injecting spurious corrections into heading/speed
        on every update.)
        """
        psi = self.x[2]
        v = self.x[3]

        # Measurement: v_right = 0
        H = np.zeros((1, 6))
        H[0, 2] = 0.0   # d(v_right)/d(psi) — v_right is identically 0 here
        H[0, 3] = 0.0   # d(v_right)/d(v)

        # Predicted lateral velocity
        vx = v * np.sin(psi)
        vy = v * np.cos(psi)
        v_right_pred = vx * np.cos(psi) - vy * np.sin(psi)

        innovation = np.array([0.0 - v_right_pred])

        S = H @ self.P @ H.T + self.R_nhc
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ innovation).flatten()
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_zupt(self):
        """Zero Velocity Update: constrain speed to 0 when stationary."""
        H = np.zeros((1, 6))
        H[0, 3] = 1.0  # speed

        innovation = np.array([0.0 - self.x[3]])

        S = H @ self.P @ H.T + self.R_zupt
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ innovation).flatten()
        self.P = (np.eye(6) - K @ H) @ self.P

    def log(self, t):
        """Log current state."""
        self.trajectory.append(self.x.copy())
        self.timestamps.append(t)

    def get_trajectory_arrays(self):
        """Return trajectory as arrays."""
        traj = np.array(self.trajectory)
        if len(traj) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        return traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]

    def enu_to_latlon(self, x_east, y_north):
        lat = self.lat0 + y_north / self.m_per_deg_lat
        lon = self.lon0 + x_east / self.m_per_deg_lon
        return lat, lon

    def get_position_latlon(self):
        return self.enu_to_latlon(self.x[0], self.x[1])


def run_ekf_fusion(vdf, gnss_available_mask=None,
                   use_nhc=True, use_zupt=True,
                   gnss_denied_segment=None,
                   ai_speeds=None):
    """
    Run EKF fusion on a full V-file sequence.

    Args:
        vdf: Vehicle DataFrame
        gnss_available_mask: boolean array, True where GPS is available.
                             None means always available (GNSS never denied).
        use_nhc: apply Non-Holonomic Constraint
        use_zupt: apply Zero-Velocity Update when v ≈ 0
        gnss_denied_segment: tuple (start_s, end_s) to simulate GNSS outage by time
        ai_speeds: numpy array of ML predicted speeds (m/s). Used when GNSS is denied.

    Returns:
        dict with trajectory, error metrics
    """
    n = len(vdf)
    times = vdf['time_s'].values
    lats = vdf['latitude_deg'].values
    lons = vdf['longitude_deg'].values
    speeds = vdf['gps_velocity_ms'].values
    yaw_rates = vdf['yaw_rate_rads'].values * (-1.0)  # calibrated sign

    # Build GNSS availability mask
    if gnss_available_mask is None:
        gnss_mask = np.ones(n, dtype=bool)
    else:
        gnss_mask = gnss_available_mask.copy()

    # Override with simulated outage window
    if gnss_denied_segment is not None:
        t_start, t_end = gnss_denied_segment
        gnss_mask &= ~((times >= t_start) & (times <= t_end))

    # Ground truth in ENU
    lat0 = float(lats[0])
    lon0 = float(lons[0])
    lat0_rad = np.radians(lat0)
    R = 6371000.0
    m_per_deg_lat = R * np.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * np.cos(lat0_rad)
    gt_x = (lons - lon0) * m_per_deg_lon
    gt_y = (lats - lat0) * m_per_deg_lat

    # Initialize EKF
    init_heading = float(np.radians(vdf['gps_heading_deg'].iloc[0]))
    ekf = EKF_INS(lat0, lon0, init_heading, init_speed=float(speeds[0]))

    est_x = np.zeros(n)
    est_y = np.zeros(n)
    gnss_used = np.zeros(n, dtype=bool)

    for i in range(1, n):
        dt = times[i] - times[i-1]
        if dt <= 0 or dt > 1.0:
            dt = 0.1

        # Speed must be corrected BEFORE predict() — predict() only
        # advances position using whatever speed is already in the state
        # (see EKF_INS.predict docstring). Getting this backwards was the
        # root cause of the ground-truth leak into the "denied" window.
        if ai_speeds is not None and not gnss_mask[i]:
            speed_meas = ai_speeds[i]
        else:
            speed_meas = speeds[i]

        ekf.update_speed(speed_meas)

        # ZUPT when nearly stationary. NOTE: this still gates on the
        # ground-truth `speeds[i]` — a real device can't do that, and
        # instead uses an IMU-variance-based motion detector (see
        # navigation_service.dart _updateMotionState). This is an
        # idealized stand-in for that detector, not a claim that a real
        # system has this information during a genuine GNSS outage.
        if use_zupt and speeds[i] < 0.3:
            ekf.update_zupt()

        # Predict — now uses the just-corrected speed, not a stale or
        # ground-truth-leaked one.
        ekf.predict(dt, yaw_rates[i])

        # NHC update
        if use_nhc:
            ekf.update_nhc()

        # GNSS update (only when available)
        if gnss_mask[i]:
            ekf.update_gnss(lats[i], lons[i])
            gnss_used[i] = True

        ekf.log(times[i])
        est_x[i] = ekf.x[0]
        est_y[i] = ekf.x[1]

    # Position errors
    errors_m = np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)
    dists = np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2)
    total_dist = float(np.sum(dists))

    # Metrics during GNSS-denied period only (if applicable).
    #
    # NOTE: `drift_pct` below (final whole-trip error / whole-trip distance)
    # is NOT a measure of dead-reckoning accuracy during a simulated outage —
    # once GNSS is reacquired it keeps correcting the filter for the rest of
    # the (often much longer) trip, so `errors_m[-1]` converges back toward
    # zero regardless of how bad the DR was during the outage. The metric
    # that actually answers "how much did the position drift during the N-
    # second GNSS-denied segment" is `denied_drift_pct` below: the position
    # error *added* during the outage window, divided by the distance
    # actually travelled during that same window.
    denied_mask = ~gnss_mask
    if denied_mask.sum() > 0:
        denied_errors = errors_m[denied_mask]
        denied_rmse = float(np.sqrt(np.mean(denied_errors**2)))
        denied_max = float(denied_errors.max())

        denied_idx = np.where(denied_mask)[0]
        start_idx = int(denied_idx[0])
        end_idx = int(denied_idx[-1])
        baseline_error = float(errors_m[start_idx - 1]) if start_idx > 0 else 0.0
        outage_added_error = float(errors_m[end_idx] - baseline_error)
        outage_dist = float(np.sum(dists[start_idx:end_idx])) if end_idx > start_idx else 0.0
        denied_drift_pct = (
            float(outage_added_error / outage_dist * 100.0) if outage_dist > 0 else 0.0
        )
    else:
        denied_rmse = 0.0
        denied_max = 0.0
        outage_dist = 0.0
        outage_added_error = 0.0
        denied_drift_pct = 0.0

    return {
        'est_x': est_x, 'est_y': est_y,
        'gt_x': gt_x, 'gt_y': gt_y,
        'errors_m': errors_m,
        'times': times,
        'gnss_mask': gnss_mask,
        'total_dist_m': total_dist,
        'final_error_m': float(errors_m[-1]),
        'mean_error_m': float(np.mean(errors_m)),
        'max_error_m': float(errors_m.max()),
        'rmse_m': float(np.sqrt(np.mean(errors_m**2))),
        # Whole-trip drift — NOT representative of outage-only performance
        # once GNSS is reacquired partway through. Kept for backward
        # compatibility with callers that want the full-trip number.
        'drift_pct': float(errors_m[-1] / total_dist * 100.0) if total_dist > 0 else 0.0,
        'denied_rmse_m': denied_rmse,
        'denied_max_error_m': denied_max,
        # The correct "SIH-style" metric: drift accumulated specifically
        # during the GNSS-denied window, relative to distance travelled
        # during that window.
        'denied_outage_dist_m': outage_dist,
        'denied_outage_added_error_m': outage_added_error,
        'denied_drift_pct': denied_drift_pct,
    }
