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

    def predict(self, dt, yaw_rate, forward_speed):
        """
        Propagate state using vehicle kinematics.

        Args:
            dt: time step (s)
            yaw_rate: measured yaw rate (rad/s) — positive = left turn
            forward_speed: measured speed (m/s)
        """
        x, y, psi, v, b_psi, b_v = self.x

        # Bias-corrected measurements
        psi_dot = yaw_rate - b_psi
        v_corr = max(0.0, forward_speed - b_v)

        # State transition (mid-point Euler)
        psi_new = psi + psi_dot * dt
        v_new = v * 0.9 + v_corr * 0.1  # smooth

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
        F[3, 5] = -0.1                             # dv/db_v

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
        Constrains: v_lateral = -vx*sin(psi) + vy*cos(psi) = 0

        For a ground vehicle, velocity in body frame:
          v_right = -vx*sin(psi) + vy*cos(psi) ≈ 0
        """
        psi = self.x[2]
        v = self.x[3]

        # Measurement: v_right = 0
        H = np.zeros((1, 6))
        H[0, 2] = v * np.cos(psi)   # d(v_right)/d(psi)
        H[0, 3] = -np.sin(psi)       # NOTE: correct sign for lateral

        # Predicted lateral velocity
        vx = v * np.sin(psi)
        vy = v * np.cos(psi)
        v_right_pred = -vx * np.sin(psi) + vy * np.cos(psi)

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

        # Predict
        ekf.predict(dt, yaw_rates[i], speeds[i])

        # Speed update
        if ai_speeds is not None and not gnss_mask[i]:
            speed_meas = ai_speeds[i]
        else:
            speed_meas = speeds[i]
            
        ekf.update_speed(speed_meas)

        # NHC update
        if use_nhc:
            ekf.update_nhc()

        # ZUPT when nearly stationary
        if use_zupt and speeds[i] < 0.3:
            ekf.update_zupt()

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

    # Metrics during GNSS-denied period only (if applicable)
    denied_mask = ~gnss_mask
    if denied_mask.sum() > 0:
        denied_errors = errors_m[denied_mask]
        denied_rmse = float(np.sqrt(np.mean(denied_errors**2)))
        denied_max = float(denied_errors.max())
    else:
        denied_rmse = 0.0
        denied_max = 0.0

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
        'drift_pct': float(errors_m[-1] / total_dist * 100.0) if total_dist > 0 else 0.0,
        'denied_rmse_m': denied_rmse,
        'denied_max_error_m': denied_max,
    }
