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
    if gps_speed is not None and len(gps_speed) == len(accel_samples):
        # Find periods of positive acceleration
        speed_diff = np.diff(gps_speed)
        speed_diff = np.append(speed_diff, 0)
        
        # Select samples where speed is increasing
        accel_mask = speed_diff > 0.5  # m/s^2 roughly
        if np.sum(accel_mask) > 10:
            forward = np.mean(lin_accel_proj[accel_mask], axis=0)
        else:
            # Fallback to PCA if not enough forward acceleration
            cov = np.cov(lin_accel_proj.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            forward = eigenvectors[:, np.argmax(eigenvalues)]
    else:
        # Use PCA on projected acceleration
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
