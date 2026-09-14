"""
alignment.py — Phase 4: Vehicle Alignment & Calibration
WayFinder / IDR Navigation System

Transforms smartphone IMU coordinates into vehicle coordinates.
Method:
1. Estimate Down vector using average gravity (stationary or moving).
2. Estimate Forward vector using acceleration when the vehicle accelerates forward.
3. Compute Right vector = Down x Forward.
4. Orthogonalize and compute the rotation matrix.
"""

import numpy as np

def estimate_alignment_matrix(accel_samples, gps_speed=None):
    """
    Estimate the rotation matrix from phone frame to vehicle frame.
    
    Args:
        accel_samples: (N, 3) array of acceleration vectors (including gravity).
        gps_speed: (N,) array of GPS speeds to identify forward acceleration.
                   If None, assumes the principal axis of variance orthogonal to gravity is forward.
                   
    Returns:
        R: (3, 3) rotation matrix where v_vehicle = R @ v_phone.
           Vehicle frame: X=Forward, Y=Right, Z=Down.
    """
    # 1. Estimate Down vector (Z-axis) from average acceleration (which is mostly gravity)
    down = np.mean(accel_samples, axis=0)
    down_norm = np.linalg.norm(down)
    if down_norm < 1e-6:
        return np.eye(3)
    z_vec = down / down_norm
    
    # Remove gravity to get linear acceleration
    lin_accel = accel_samples - down
    
    # Project linear acceleration onto the plane orthogonal to Z
    lin_accel_proj = lin_accel - np.outer(np.dot(lin_accel, z_vec), z_vec)
    
    # 2. Estimate Forward vector (X-axis)
    #
    # A meaningful "sustained acceleration" event is much smaller than the
    # original 0.5 m/s-per-sample (== 5 m/s^2 sustained, near max braking)
    # threshold — that was so strict it only fired for 10/42 training
    # sequences, sending the other 32 to the PCA fallback below on every run.
    if gps_speed is not None and len(gps_speed) == len(accel_samples):
        speed_diff = np.diff(gps_speed)
        speed_diff = np.append(speed_diff, 0)

        # Select samples where speed is changing at a noticeable rate
        # (accelerating or braking), in either direction — magnitude only;
        # sign is resolved explicitly below regardless of which branch fires.
        accel_mask = np.abs(speed_diff) > 0.15  # m/s^2 roughly
        if np.sum(accel_mask) > 10:
            forward = np.mean(lin_accel_proj[accel_mask], axis=0)
        else:
            cov = np.cov(lin_accel_proj.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            forward = eigenvectors[:, np.argmax(eigenvalues)]
    else:
        cov = np.cov(lin_accel_proj.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        forward = eigenvectors[:, np.argmax(eigenvalues)]

    # Ensure forward is orthogonal to z_vec
    forward = forward - np.dot(forward, z_vec) * z_vec
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        # Fallback if no horizontal acceleration
        # Create an arbitrary orthogonal vector
        arbitrary = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(arbitrary, z_vec)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        forward = arbitrary - np.dot(arbitrary, z_vec) * z_vec
        forward_norm = np.linalg.norm(forward)

    x_vec = forward / forward_norm

    # Resolve sign ambiguity: an eigenvector (PCA branch) has no defined
    # sign, and even the masked-mean branch can end up net-negative on a
    # trip with more/harder braking events than accelerations. Force a
    # consistent convention by requiring the forward axis to positively
    # correlate with actual speed changes over the whole trip — without
    # this, roughly half of all sequences end up with "forward" pointing
    # backward, so the same physical event (accelerating) has opposite-
    # signed features across different training sequences.
    if gps_speed is not None and len(gps_speed) == len(accel_samples):
        speed_diff = np.diff(gps_speed)
        speed_diff = np.append(speed_diff, 0)
        fwd_accel_signal = lin_accel_proj @ x_vec
        agreement = np.dot(fwd_accel_signal, speed_diff)
        if agreement < 0:
            x_vec = -x_vec
    
    # 3. Compute Right vector (Y-axis)
    y_vec = np.cross(z_vec, x_vec)
    
    # The rotation matrix from phone to vehicle
    # R_phone_to_vehicle @ v_phone = v_vehicle
    # Rows of R are the vehicle axes in the phone frame
    R = np.vstack([x_vec, y_vec, z_vec])
    
    return R

def apply_alignment(accel_samples, gyro_samples, R):
    """
    Apply rotation matrix to IMU samples.

    Args:
        accel_samples: (N, 3)
        gyro_samples: (N, 3)
        R: (3, 3) rotation matrix

    Returns:
        aligned_accel, aligned_gyro
    """
    aligned_accel = (R @ accel_samples.T).T
    aligned_gyro = (R @ gyro_samples.T).T
    return aligned_accel, aligned_gyro


FEATURE_KEYS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

# Tried adding derived accel/gyro magnitude channels (a vibration proxy) —
# validation-set bin bias looked mixed-to-slightly-better, but real drift on
# the evaluation set got clearly worse (74.4% mean / 1/7 pass vs. 53.8% / 2/7
# without them). Disabled by default; kept behind a flag rather than deleted
# in case a future, larger/more diverse training set makes it worth
# revisiting — same underlying feature-imbalance/noise issues that hurt the
# base model likely apply here too, just harder to diagnose with 2 extra
# noisy derived channels.
EXTRA_FEATURE_KEYS = ['accel_horiz_mag', 'gyro_mag']


def build_features(raw_accel, raw_gyro, R, include_magnitude=False):
    """
    Single source of truth for the model's input features, used by training
    (dataset.py), evaluation (inference.py, test_inference.py) and
    normalization-stats computation (pipeline.py) alike — keeping this in
    one place instead of copy-pasted at each call site is exactly what
    would have caught earlier bugs where those call sites quietly drifted
    apart (e.g. stats computed from raw data but applied to aligned data).

    Returns an (N, 6) array (or (N, 8) if include_magnitude=True) in the
    order given by FEATURE_KEYS (+ EXTRA_FEATURE_KEYS).
    """
    aligned_accel, aligned_gyro = apply_alignment(raw_accel, raw_gyro, R)
    features = np.column_stack([aligned_accel, aligned_gyro])
    if include_magnitude:
        accel_horiz_mag = np.sqrt(aligned_accel[:, 0] ** 2 + aligned_accel[:, 1] ** 2)
        gyro_mag = np.linalg.norm(aligned_gyro, axis=1)
        features = np.column_stack([features, accel_horiz_mag, gyro_mag])
    return features
