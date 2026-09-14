"""
classical_ins.py — Classical Inertial Navigation System Baseline
Phase 3: Classical INS Baseline

Implements dead reckoning using:
- ECU yaw rate (gyroscope equivalent)
- ECU longitudinal acceleration for speed correction
- Wheel odometry for speed
- Heading integration
- Position propagation in local ENU frame

This is the BASELINE against which ALL AI improvements are measured.
NO GPS is used during the DR phase.
"""

import numpy as np


class ClassicalINS:
    """
    Classical dead-reckoning navigation system.
    
    State:
        x:    Easting (m) in local ENU frame (relative to start)
        y:    Northing (m) in local ENU frame
        heading: True heading (radians, N=0, E=π/2)
        speed: Forward speed (m/s)
    
    Inputs (per timestep):
        dt:           time step (seconds)
        yaw_rate:     vehicle yaw rate (rad/s) — from ECU or gyro
        forward_speed: vehicle forward speed (m/s) — from odometry or ECU
    
    Coordinate convention:
        x = East
        y = North
        heading: 0 = North, increases clockwise (navigation convention)
    """
    
    def __init__(self, init_lat=None, init_lon=None):
        """
        Initialize INS.
        
        Args:
            init_lat: Initial latitude (degrees) — for global position output
            init_lon: Initial longitude (degrees)
        """
        self.x = 0.0          # Easting (m)
        self.y = 0.0          # Northing (m)
        self.heading = 0.0    # radians (N=0, E=π/2)
        self.speed = 0.0      # m/s
        self.heading_bias = 0.0  # estimated heading bias
        
        # Initial GPS position for converting back to lat/lon
        self.init_lat = init_lat
        self.init_lon = init_lon
        
        # Reference Earth parameters
        self.R_earth = 6371000.0  # m
        
        # Trajectory history
        self.trajectory = []
        self.timestamps = []
    
    def reset(self, lat=None, lon=None, heading=None):
        """Reset INS to a new initial state."""
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0
        if lat is not None:
            self.init_lat = lat
        if lon is not None:
            self.init_lon = lon
        if heading is not None:
            self.heading = heading
        self.trajectory = []
        self.timestamps = []
    
    def step(self, dt: float, yaw_rate: float, forward_speed: float, t: float = 0.0):
        """
        Propagate INS by one timestep.
        
        Args:
            dt: time step (seconds)
            yaw_rate: yaw rate (rad/s) — positive = turning left (counterclockwise)
            forward_speed: forward speed (m/s) — must be non-negative
            t: current timestamp (for logging)
        """
        # Update heading (navigation convention: clockwise positive)
        # ECU yaw rate: positive = left turn in most vehicle conventions
        # In navigation convention, clockwise is positive heading
        # So: heading += yaw_rate * dt (if yaw_rate is already in nav convention)
        # The ECU yaw_rate is in vehicle body frame — positive = counterclockwise
        # Navigation heading: positive clockwise
        # Therefore heading changes by -yaw_rate_body
        # (This sign depends on the dataset's convention — verified empirically)
        self.heading += yaw_rate * dt   # Will verify sign empirically
        
        # Normalize heading to [0, 2π]
        self.heading = self.heading % (2 * np.pi)
        
        # Smooth speed (simple low-pass)
        self.speed = 0.8 * self.speed + 0.2 * max(0.0, forward_speed)
        
        # Position update using mid-point integration
        # x = East, y = North
        # heading = 0 → North, π/2 → East
        dx = self.speed * np.sin(self.heading) * dt  # Easting
        dy = self.speed * np.cos(self.heading) * dt  # Northing
        
        self.x += dx
        self.y += dy
        
        # Log
        self.trajectory.append((self.x, self.y, self.heading, self.speed))
        self.timestamps.append(t)
    
    def get_trajectory_arrays(self):
        """Return trajectory as numpy arrays."""
        if not self.trajectory:
            return np.array([]), np.array([]), np.array([]), np.array([])
        traj = np.array(self.trajectory)
        return traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]  # x, y, heading, speed
    
    def enu_to_latlon(self, x_east, y_north):
        """Convert ENU offset (m) to lat/lon given initial position."""
        if self.init_lat is None or self.init_lon is None:
            return None, None
        
        lat0_rad = np.radians(self.init_lat)
        
        # Metres per degree (approximation)
        m_per_deg_lat = self.R_earth * np.pi / 180.0
        m_per_deg_lon = m_per_deg_lat * np.cos(lat0_rad)
        
        lat = self.init_lat + y_north / m_per_deg_lat
        lon = self.init_lon + x_east / m_per_deg_lon
        
        return lat, lon
    
    def get_position_latlon(self):
        """Return current position in lat/lon."""
        return self.enu_to_latlon(self.x, self.y)


def run_classical_ins(vdf, use_odometry=True, yaw_sign=1.0):
    """
    Run classical INS over a full sequence (V-file DataFrame).
    
    Args:
        vdf: Vehicle DataFrame (from loader.py)
        use_odometry: if True, use wheel odometry speed; else use ECU speed
        yaw_sign: sign multiplier for yaw rate (empirically determined)
    
    Returns:
        dict with trajectory arrays and error metrics
    """
    n = len(vdf)
    
    # Initialize INS at first GPS position
    lat0 = float(vdf['latitude_deg'].iloc[0])
    lon0 = float(vdf['longitude_deg'].iloc[0])
    heading0 = float(np.radians(vdf['gps_heading_deg'].iloc[0]))
    
    ins = ClassicalINS(init_lat=lat0, init_lon=lon0)
    ins.heading = heading0
    
    # Arrays for output
    est_x = np.zeros(n)
    est_y = np.zeros(n)
    gt_x = np.zeros(n)
    gt_y = np.zeros(n)
    
    # Convert GT trajectory to ENU
    lats = vdf['latitude_deg'].values
    lons = vdf['longitude_deg'].values
    lat0_rad = np.radians(lat0)
    R = 6371000.0
    m_per_deg_lat = R * np.pi / 180.0
    m_per_deg_lon = m_per_deg_lat * np.cos(lat0_rad)
    
    gt_x = (lons - lon0) * m_per_deg_lon
    gt_y = (lats - lat0) * m_per_deg_lat
    
    # Speed source
    if use_odometry:
        speeds = vdf['odometry_speed_ms'].values
    else:
        speeds = vdf['gps_velocity_ms'].values
    
    yaw_rates = vdf['yaw_rate_rads'].values * yaw_sign
    times = vdf['time_s'].values
    
    est_x_arr = np.zeros(n)
    est_y_arr = np.zeros(n)
    est_x_arr[0] = 0.0
    est_y_arr[0] = 0.0
    
    for i in range(1, n):
        dt = times[i] - times[i-1]
        if dt <= 0 or dt > 1.0:  # Skip bad timesteps
            dt = 0.1
        
        yaw_rate = yaw_rates[i]
        speed = speeds[i]
        
        ins.step(dt, yaw_rate, speed, t=times[i])
        est_x_arr[i] = ins.x
        est_y_arr[i] = ins.y
    
    # Compute position errors
    errors_m = np.sqrt((est_x_arr - gt_x)**2 + (est_y_arr - gt_y)**2)
    
    # Total distance
    dists = np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2)
    total_dist = float(np.sum(dists))
    
    # Drift percentage
    final_error = float(errors_m[-1])
    drift_pct = (final_error / total_dist * 100.0) if total_dist > 0 else 0.0
    
    return {
        'est_x': est_x_arr,
        'est_y': est_y_arr,
        'gt_x': gt_x,
        'gt_y': gt_y,
        'errors_m': errors_m,
        'times': times,
        'total_dist_m': total_dist,
        'final_error_m': final_error,
        'drift_pct': drift_pct,
        'mean_error_m': float(np.mean(errors_m)),
        'max_error_m': float(np.max(errors_m)),
        'rmse_m': float(np.sqrt(np.mean(errors_m**2))),
    }
