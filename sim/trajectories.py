"""Ground-truth trajectories for the target drone and the camera platform.

Everything is in the ENU world frame (metres, seconds). The camera starts at
CAM_START (30 m altitude). A scenario combines one drone path with one camera path and
returns time-aligned arrays (see `make_scenario`).

Drone paths
  approach_recede   : straight line from 200 m away to a 10 m closest approach at
                      t = T_CA, then flies away (range sweeps ~200 m -> 10 m -> ~86 m).
  constant_velocity : crossing target starting ~136 m from the camera, flying east at 6 m/s
                      (range then spans ~64-253 m depending on the camera path).
  gentle_turn       : same start and speed, but turning at 3 deg/s (radius ~115 m).

Camera paths (defined relative to the initial line of sight, LOS, from camera to drone)
  straight : flies straight along the LOS towards the drone at 4 m/s.
             The bearing to the drone barely changes -> almost no parallax (hard case).
  lateral  : flies perpendicular to the LOS at 4 m/s (constant velocity, sideways).
  weaving  : creeps forward at 2 m/s and weaves side to side (+/-15 m, 12 s period).
             The sideways ACCELERATION is what makes range observable from bearings alone.

The camera's optical axis follows the drone (an ideal gimbal) plus a slow +/-2 deg
pointing wobble, so the drone wanders around the image instead of sitting at the centre.
The EKF never sees these true poses: it only gets the noisy pose from sim/noise.py.
"""
import numpy as np

from sim.camera import look_at, rotation_from_vector

DT = 0.1            # 10 Hz detections
DURATION = 40.0     # seconds per run
T_CA = 28.0         # time of closest approach for approach_recede
CAM_START = np.array([0.0, 0.0, 30.0])

DRONE_PATHS = ("approach_recede", "constant_velocity", "gentle_turn")
CAMERA_PATHS = ("straight", "lateral", "weaving")


# ---------------------------------------------------------------- camera paths
def camera_path(name, t, los, lat):
    """Camera positions (N,3) and velocities (N,3). los/lat: horizontal unit vectors
    along and across the initial line of sight."""
    t = t[:, None]
    if name == "straight":
        p = CAM_START + 4.0 * los * t
        v = np.repeat(4.0 * los[None], len(t), axis=0)
    elif name == "lateral":
        p = CAM_START + 4.0 * lat * t
        v = np.repeat(4.0 * lat[None], len(t), axis=0)
    elif name == "weaving":
        A, w = 15.0, 2 * np.pi / 12.0                     # amplitude [m], angular freq [rad/s]
        p = CAM_START + 2.0 * los * t + A * np.sin(w * t) * lat
        v = 2.0 * los + A * w * np.cos(w * t) * lat
    else:
        raise ValueError(name)
    return p, v


# ---------------------------------------------------------------- drone paths
def drone_start(name):
    if name == "approach_recede":
        return np.array([0.0, np.sqrt(200.0**2 - 15.0**2), 45.0])   # 200 m from CAM_START
    return np.array([-60.0, 120.0, 50.0])                            # crossing targets


def drone_path(name, t, cam_p):
    """Drone positions (N,3) and velocities (N,3)."""
    p0 = drone_start(name)
    if name == "approach_recede":
        # Constant velocity chosen so the drone passes 10 m from the camera at T_CA
        # (6 m to the east and 8 m above the camera's position at that moment).
        k = int(round(T_CA / DT))
        p_ca = cam_p[k] + np.array([6.0, 0.0, 8.0])
        vel = (p_ca - p0) / T_CA
        return p0 + vel * t[:, None], np.repeat(vel[None], len(t), axis=0)
    if name == "constant_velocity":
        vel = np.array([6.0, 0.0, 0.0])
        return p0 + vel * t[:, None], np.repeat(vel[None], len(t), axis=0)
    if name == "gentle_turn":
        speed, omega = 6.0, np.deg2rad(3.0)               # m/s, rad/s (turning left)
        heading = omega * t                               # starts flying east (heading 0)
        radius = speed / omega
        # integrate v = speed*[cos h, sin h]: x = R sin h, y = R (1 - cos h)
        p = np.column_stack([p0[0] + radius * np.sin(heading),
                             p0[1] + radius * (1 - np.cos(heading)),
                             np.full(len(t), p0[2])])
        v = np.column_stack([speed * np.cos(heading), speed * np.sin(heading), np.zeros(len(t))])
        return p, v
    raise ValueError(name)


def cv_with_process_noise(p0, v0, t, q, rng):
    """Drone that follows EXACTLY the EKF's motion model: constant velocity plus white
    acceleration noise with spectral density q [m^2/s^3]. Used for the NEES consistency
    check, where the truth must come from the same model as the filter."""
    from ekf.ekf import cv_blocks
    x = np.concatenate([p0, v0])
    out = [x]
    for k in range(1, len(t)):
        F, Q = cv_blocks(t[k] - t[k - 1], q)
        x = F @ x + rng.multivariate_normal(np.zeros(6), Q)
        out.append(x)
    out = np.array(out)
    return out[:, :3], out[:, 3:]


# ---------------------------------------------------------------- scenario
def gimbal_rotation(cam_p, target_p, t):
    """Camera points at the drone, plus a slow +/-2 deg wobble in yaw and pitch."""
    wobble = np.deg2rad(2.0) * np.array([np.sin(0.7 * t), np.sin(0.45 * t + 1.0), 0.0])
    return look_at(cam_p, target_p) @ rotation_from_vector(wobble)


def make_scenario(drone_name, camera_name, dt=DT, duration=DURATION, drone_override=None):
    """Time-aligned ground truth. Returns a dict with
        t (N,), drone_p (N,3), drone_v (N,3), cam_p (N,3), cam_R (N,3,3), names.
    drone_override: optional (drone_p, drone_v) arrays to use instead of a named path."""
    t = np.arange(0.0, duration + 1e-9, dt)
    d0 = drone_start(drone_name)
    los = np.array([d0[0] - CAM_START[0], d0[1] - CAM_START[1], 0.0])
    los /= np.linalg.norm(los)
    lat = np.cross(los, [0.0, 0.0, 1.0])           # horizontal, 90 deg to the right of the LOS
    cam_p, cam_v = camera_path(camera_name, t, los, lat)
    if drone_override is None:
        drone_p, drone_v = drone_path(drone_name, t, cam_p)
    else:
        drone_p, drone_v = drone_override
    cam_R = np.array([gimbal_rotation(cam_p[k], drone_p[k], t[k]) for k in range(len(t))])
    return {"t": t, "drone_p": drone_p, "drone_v": drone_v, "cam_p": cam_p, "cam_v": cam_v,
            "cam_R": cam_R, "drone": drone_name, "camera": camera_name}
