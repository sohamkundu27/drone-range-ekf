"""Pinhole camera model on a moving platform.

Conventions
-----------
World frame  : right-handed ENU (x = east, y = north, z = up), metres.
Camera frame : x = right, y = down, z = forward (optical axis), metres.
Pose         : R_wc (3x3) rotates camera-frame vectors into the world frame,
               t_w (3,) is the camera centre in the world frame.
               A world point p_w is, in camera coordinates,
                   p_c = R_wc^T (p_w - t_w)
Pixels       : u to the right, v down, origin at the top-left corner.

Pinhole model (no lens distortion):
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
where (X, Y, Z) = p_c. Z is the DEPTH along the optical axis (not the range).

Apparent size of the drone (what makes range observable from one camera):
    w_px = fx * S / Z
S is the drone's physical width in metres. Doubling the distance halves the box.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    """Camera intrinsics. Default: 1920x1080 sensor (the most common DUT Anti-UAV frame size;
    detector/measure_noise.py rescales all DUT pixel errors to a 1920-wide frame so they carry
    over), fx = 2500 px (~42 deg horizontal FOV)."""
    fx: float = 2500.0
    fy: float = 2500.0
    cx: float = 960.0
    cy: float = 540.0
    width: int = 1920
    height: int = 1080

    def K(self):
        """3x3 calibration matrix."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])


def world_to_camera(p_w, R_wc, t_w):
    """World point -> camera coordinates: p_c = R_wc^T (p_w - t_w)."""
    return R_wc.T @ (np.asarray(p_w, float) - t_w)


def project(p_w, R_wc, t_w, K: Intrinsics):
    """Project a world point to pixels. Returns (u, v, Z); Z <= 0 means behind the camera."""
    X, Y, Z = world_to_camera(p_w, R_wc, t_w)
    u = K.fx * X / Z + K.cx
    v = K.fy * Y / Z + K.cy
    return u, v, Z


def project_box(p_w, size_m, R_wc, t_w, K: Intrinsics):
    """Apparent box of a drone of physical width `size_m` centred at p_w.
    Returns (u, v, w_px, Z): box centre, box width in pixels, depth."""
    u, v, Z = project(p_w, R_wc, t_w, K)
    w_px = K.fx * size_m / Z
    return u, v, w_px, Z


def bearing(u, v, K: Intrinsics):
    """Inverse of `project` up to scale: unit ray (camera frame) through pixel (u, v).
    The pixel only tells us the DIRECTION to the drone, not how far away it is."""
    ray = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
    return ray / np.linalg.norm(ray)


def unproject(u, v, Z, K: Intrinsics):
    """Camera-frame point at pixel (u, v) and depth Z (used for initialisation and tests)."""
    return np.array([(u - K.cx) / K.fx * Z, (v - K.cy) / K.fy * Z, Z])


def is_visible(u, v, Z, K: Intrinsics, min_depth=1.0):
    """True if the point is in front of the camera and inside the image."""
    return (Z > min_depth) and (0.0 <= u < K.width) and (0.0 <= v < K.height)


def look_at(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """Rotation R_wc for a camera at cam_pos whose optical axis points at `target`.
    Columns of R_wc are the camera axes written in world coordinates:
        z_c = forward (towards the target)
        x_c = right   = forward x up   (normalised)
        y_c = down    = forward x right
    Assumes the target is not straight up or down (forward not parallel to up)."""
    z_c = np.asarray(target, float) - np.asarray(cam_pos, float)
    z_c /= np.linalg.norm(z_c)
    x_c = np.cross(z_c, up)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    return np.column_stack([x_c, y_c, z_c])


def skew(a):
    """Cross-product matrix: skew(a) @ b == np.cross(a, b)."""
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def rotation_from_vector(theta):
    """Rodrigues' formula: rotation by angle |theta| about axis theta/|theta|.
    Used to apply small attitude errors to the camera rotation."""
    angle = np.linalg.norm(theta)
    if angle < 1e-12:
        return np.eye(3) + skew(theta)
    k = skew(theta / angle)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)
