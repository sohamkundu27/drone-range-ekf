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
  Added for the sweep-in-place study (experiments/run_sweep.py; NOT in DRONE_PATHS):
  dash  : starts where approach_recede starts (200 m out) and flies at the point 10 m from the
          SWEEP CENTRE (CAM_START + [6, 0, 8], the same miss offset as approach_recede) at 5 m/s,
          accelerates at 2 m/s^2 from t = 10 s until 15 m/s, then keeps 15 m/s past closest
          approach (closest approach to the centre at t ~ 21.6 s).
  jink  : exactly the approach_recede path and speed, plus a horizontal offset
          5 sin(2 pi t / 6 + phase) m perpendicular to the path (peak lateral accel 5.5 m/s^2).
          The phase is an argument (experiments/run_sweep.py draws it per run).
  orbit : horizontal circle of radius 80 m around the sweep centre at 8 m/s (0.1 rad/s),
          counter-clockwise, altitude 50 m (as constant_velocity / gentle_turn). Starts on the
          line through the centre perpendicular to the sweep axis (best parallax); after
          pi/2 / 0.1 = 15.7 s it lines up with the sweep axis, where the sweep gives no
          sideways parallax.

Camera paths (defined relative to the initial line of sight, LOS, from camera to drone)
  straight : flies straight along the LOS towards the drone at 4 m/s.
             The bearing to the drone barely changes -> almost no parallax (hard case).
  lateral  : flies perpendicular to the LOS at 4 m/s (constant velocity, sideways).
  weaving  : creeps forward at 2 m/s and weaves side to side (+/-15 m, 12 s period).
             The sideways ACCELERATION is what makes range observable from bearings alone.

Added camera paths (NOT in CAMERA_PATHS, so the main grid is unchanged):
  sweep : no forward motion; CAM_START + A sin(2 pi t / T) * lat, with lat the horizontal unit
          vector perpendicular to the initial LOS (fixed in the world; CAM_START is the sweep
          centre). Width A is a parameter (default 15 m); T = 12 sqrt(A / 15) s, so the peak
          acceleration A (2 pi / T)^2 = 4.1 m/s^2 is the same for every width.
  hover : stays at CAM_START (the gimbal still tracks the drone).

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
SWEEP_DRONE_PATHS = ("dash", "jink", "orbit")    # extra targets, used by experiments/run_sweep.py
SWEEP_CAMERA_PATHS = ("sweep", "hover")          # extra cameras, used by experiments/run_sweep.py
SWEEP_A, SWEEP_T = 15.0, 12.0                    # default sweep half-width [m] and period [s]


# ---------------------------------------------------------------- camera paths
def sweep_period(A):
    """Sweep period for half-width A: same peak acceleration A (2 pi / T)^2 as the default 15 m / 12 s."""
    return SWEEP_T * np.sqrt(A / SWEEP_A)


def camera_path(name, t, los, lat, sweep_A=SWEEP_A):
    """Camera positions (N,3) and velocities (N,3). los/lat: horizontal unit vectors
    along and across the initial line of sight. sweep_A: half-width of the 'sweep' camera [m]."""
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
    elif name == "sweep":
        A, w = sweep_A, 2 * np.pi / sweep_period(sweep_A)
        p = CAM_START + A * np.sin(w * t) * lat
        v = A * w * np.cos(w * t) * lat
    elif name == "hover":
        p = np.repeat(CAM_START[None], len(t), axis=0)
        v = np.zeros((len(t), 3))
    else:
        raise ValueError(name)
    return p, v


# ---------------------------------------------------------------- drone paths
ORBIT_R, ORBIT_SPEED, ORBIT_ALT = 80.0, 8.0, 50.0
JINK_AMP, JINK_PERIOD = 5.0, 6.0
CA_OFFSET = np.array([6.0, 0.0, 8.0])           # 10 m miss offset used by approach_recede and dash


def drone_start(name):
    if name in ("approach_recede", "dash", "jink"):
        return np.array([0.0, np.sqrt(200.0**2 - 15.0**2), 45.0])   # 200 m from CAM_START
    if name == "orbit":                         # on the initial LOS axis (+y), i.e. perpendicular
        return np.array([CAM_START[0], CAM_START[1] + ORBIT_R, ORBIT_ALT])   # to the sweep axis
    return np.array([-60.0, 120.0, 50.0])                            # crossing targets


def dash_distance(t):
    """Distance flown [m] and speed [m/s] of the dash target: 5 m/s, +2 m/s^2 from t = 10 s to 15 m/s."""
    t = np.asarray(t, float)
    d = np.where(t < 10, 5 * t, np.where(t < 15, 50 + 5 * (t - 10) + (t - 10) ** 2, 100 + 15 * (t - 15)))
    v = np.where(t < 10, 5.0, np.where(t < 15, 5 + 2 * (t - 10), 15.0))
    return d, v


def drone_path(name, t, cam_p, jink_phase=0.0):
    """Drone positions (N,3) and velocities (N,3). jink_phase [rad] only affects 'jink'."""
    p0 = drone_start(name)
    if name == "dash":                          # relative to the sweep centre, not the moving camera
        u = (CAM_START + CA_OFFSET) - p0
        u /= np.linalg.norm(u)
        d, sp = dash_distance(t)
        return p0 + d[:, None] * u, sp[:, None] * u
    if name == "jink":
        p, v = drone_path("approach_recede", t, cam_p)
        n = np.cross(v[0] / np.linalg.norm(v[0]), [0.0, 0.0, 1.0])
        n /= np.linalg.norm(n)                  # horizontal, perpendicular to the path
        w = 2 * np.pi / JINK_PERIOD
        return (p + JINK_AMP * np.sin(w * t + jink_phase)[:, None] * n,
                v + JINK_AMP * w * np.cos(w * t + jink_phase)[:, None] * n)
    if name == "orbit":                         # counter-clockwise around the sweep centre
        omega = ORBIT_SPEED / ORBIT_R
        th = np.arctan2(p0[1] - CAM_START[1], p0[0] - CAM_START[0]) + omega * t
        p = np.column_stack([CAM_START[0] + ORBIT_R * np.cos(th), CAM_START[1] + ORBIT_R * np.sin(th),
                             np.full(len(t), ORBIT_ALT)])
        v = np.column_stack([-ORBIT_SPEED * np.sin(th), ORBIT_SPEED * np.cos(th), np.zeros(len(t))])
        return p, v
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


def make_scenario(drone_name, camera_name, dt=DT, duration=DURATION, drone_override=None,
                  sweep_A=SWEEP_A, jink_phase=0.0):
    """Time-aligned ground truth. Returns a dict with
        t (N,), drone_p (N,3), drone_v (N,3), cam_p (N,3), cam_R (N,3,3), names.
    drone_override: optional (drone_p, drone_v) arrays to use instead of a named path.
    sweep_A: half-width of the 'sweep' camera [m]; jink_phase: phase of the 'jink' target [rad].
    Both are ignored by the other paths, so existing calls give identical arrays."""
    t = np.arange(0.0, duration + 1e-9, dt)
    d0 = drone_start(drone_name)
    los = np.array([d0[0] - CAM_START[0], d0[1] - CAM_START[1], 0.0])
    los /= np.linalg.norm(los)
    lat = np.cross(los, [0.0, 0.0, 1.0])           # horizontal, 90 deg to the right of the LOS
    cam_p, cam_v = camera_path(camera_name, t, los, lat, sweep_A=sweep_A)
    if drone_override is None:
        drone_p, drone_v = drone_path(drone_name, t, cam_p, jink_phase=jink_phase)
    else:
        drone_p, drone_v = drone_override
    cam_R = np.array([gimbal_rotation(cam_p[k], drone_p[k], t[k]) for k in range(len(t))])
    return {"t": t, "drone_p": drone_p, "drone_v": drone_v, "cam_p": cam_p, "cam_v": cam_v,
            "cam_R": cam_R, "drone": drone_name, "camera": camera_name}
