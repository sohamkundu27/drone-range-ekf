"""Random target paths for the sweep-in-place study (experiments/run_random_paths.py). SIMULATION ONLY.

Adds one target type, "random_path", without changing any existing path, camera or noise model. Everything is
deterministic from the run seed: the path uses its OWN generator default_rng([seed, 2]) (the jink phase uses
[seed, 1]), so detector noise, GPS/INS drift and the true-size draw (default_rng(seed) in experiments/mc.py run_one)
are identical to every other sweep config with the same seed.

Path model (ENU, metres, seconds; the sweep centre is CAM_START, 30 m altitude)
  start      : range from the sweep centre uniform in [30, 200] m, azimuth uniform in [0, 360) deg, height
               uniform in [-25, +50] m relative to the centre (5-80 m altitude; |height| <= 0.9 x range, so
               the start is never steeply above or below the camera)
  in view    : as for every existing target, the ideal gimbal points the camera at the drone (look_at with the
               +/-2 deg wobble, trajectories.gimbal_rotation) and a frame counts only if camera.is_visible is
               true. The path is redrawn (same generator) if the drone is not visible at t = 0 or ever comes within
               2 m horizontally of the camera, where look_at is undefined (straight up / down). Redraws are logged.
  velocity   : initial direction uniform on the sphere, speed uniform in [0, 15] m/s
  segments   : durations uniform in [4, 10] s; each draws a constant acceleration, direction uniform on the
               sphere, magnitude uniform in [0, 3] m/s^2. Base speed capped at 15 m/s.
  sidestep   : with probability 0.3 per segment, at a uniform time inside it, a +/-5 m horizontal sidestep
               perpendicular to the current velocity, over 1.0 s with a sine velocity pulse (peak extra speed
               7.9 m/s, peak lateral acceleration 5 pi^2 / 2 = 24.7 m/s^2: a hard dodge, as specified)
  altitude   : kept within 5-150 m: the initial climb / sink rate is clipped so the drone can stop in time, and when the drone could not stop before the floor / ceiling at 3 m/s^2, the
               segment acceleration is replaced by a 3 m/s^2 vertical brake
  integration: explicit Euler at dt / 10 = 0.01 s, sampled at the 10 Hz output times
The sweep camera and its axis are built exactly as trajectories.make_scenario does: axis = horizontal unit vector
perpendicular to the initial line of sight from the sweep centre to the drone.
"""
import numpy as np

from sim.camera import Intrinsics, is_visible, project_box
from sim.trajectories import CAM_START, DT, DURATION, SWEEP_A, camera_path, gimbal_rotation

R_MIN, R_MAX = 30.0, 200.0
DZ_MIN, DZ_MAX = -25.0, 50.0
V_MAX, A_MAX = 15.0, 3.0
SEG_MIN, SEG_MAX = 4.0, 10.0
P_SIDESTEP, SIDESTEP_D, SIDESTEP_T = 0.3, 5.0, 1.0
Z_FLOOR, Z_CEIL = 5.0, 150.0
FINE = 10
MIN_HORIZ = 2.0
RNG_STREAM = 2


def _unit(rng):
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def draw_path(rng, t, a_max=A_MAX, p_sidestep=P_SIDESTEP):
    """One random path sampled at the (uniform) times t. Returns p (N,3), v (N,3) and its parameters.
    a_max / p_sidestep: only for diagnostic controls (0 / 0 = constant-velocity random paths)."""
    T, dt = float(t[-1]), float(t[1] - t[0])
    r0 = rng.uniform(R_MIN, R_MAX)
    dz = rng.uniform(max(DZ_MIN, -0.9 * r0), min(DZ_MAX, 0.9 * r0))
    az = rng.uniform(0.0, 2 * np.pi)                       # clockwise from +y
    h = np.sqrt(r0 ** 2 - dz ** 2)
    p0 = CAM_START + np.array([h * np.sin(az), h * np.cos(az), dz])
    s0 = rng.uniform(0.0, V_MAX)
    v0 = s0 * _unit(rng)
    # a drone must be able to stop (3 m/s^2) before the altitude floor / ceiling: clip the initial climb / sink rate
    vz_dn = np.sqrt(2 * A_MAX * max(p0[2] - Z_FLOOR - 0.5, 0.0))
    vz_up = np.sqrt(2 * A_MAX * max(Z_CEIL - p0[2] - 0.5, 0.0))
    vz_clipped = not (-vz_dn <= v0[2] <= vz_up)
    v0[2] = np.clip(v0[2], -vz_dn, vz_up)
    segs, side = [], []                                    # segs: (t_start, accel); side: (t0, sign, fallback angle)
    ts = 0.0
    while ts < T:
        d = rng.uniform(SEG_MIN, SEG_MAX)
        segs.append((ts, rng.uniform(0.0, a_max) * _unit(rng)))
        if rng.random() < p_sidestep:
            t0 = ts + rng.uniform(0.0, d)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            fb = rng.uniform(0.0, 2 * np.pi)
            if t0 < T:
                side.append([t0, sign, fb, None])
        ts += d
    seg_t = np.array([s[0] for s in segs])

    h_f = dt / FINE
    n_out = len(t)
    P, V = np.zeros((n_out, 3)), np.zeros((n_out, 3))
    p, vb = p0.copy(), v0.copy()
    braking = 0
    acc_base_max, acc_tot_max = 0.0, 0.0
    v_prev_tot = None
    k_side = SIDESTEP_D * np.pi / (2 * SIDESTEP_T)         # peak sidestep speed
    for i in range((n_out - 1) * FINE + 1):
        tt = i * h_f
        vs = np.zeros(3)
        for s in side:                                     # sine velocity pulse, integral = +/-5 m
            if s[0] <= tt <= s[0] + SIDESTEP_T:
                if s[3] is None:                           # direction fixed at the sidestep start
                    vh = np.array([vb[0], vb[1], 0.0])
                    if np.linalg.norm(vh) > 0.5:
                        n = np.cross(vh / np.linalg.norm(vh), [0.0, 0.0, 1.0])
                    else:
                        n = np.array([np.cos(s[2]), np.sin(s[2]), 0.0])
                    s[3] = n / np.linalg.norm(n)
                vs += s[1] * k_side * np.sin(np.pi * (tt - s[0]) / SIDESTEP_T) * s[3]
        v_tot = vb + vs
        if v_prev_tot is not None:
            acc_tot_max = max(acc_tot_max, float(np.linalg.norm(v_tot - v_prev_tot)) / h_f)
        v_prev_tot = v_tot
        if i % FINE == 0:
            P[i // FINE], V[i // FINE] = p, v_tot
        a = segs[int(np.searchsorted(seg_t, tt, side="right")) - 1][1]
        stop = vb[2] ** 2 / (2 * A_MAX) + 0.5
        if vb[2] < 0 and p[2] - Z_FLOOR <= stop:
            a, braking = np.array([0.0, 0.0, A_MAX]), braking + 1
        elif vb[2] > 0 and Z_CEIL - p[2] <= stop:
            a, braking = np.array([0.0, 0.0, -A_MAX]), braking + 1
        vb_new = vb + a * h_f
        sp = np.linalg.norm(vb_new)
        if sp > V_MAX:
            vb_new *= V_MAX / sp
        acc_base_max = max(acc_base_max, float(np.linalg.norm(vb_new - vb)) / h_f)
        p = p + v_tot * h_f
        vb = vb_new
    speed = np.linalg.norm(V, axis=1)
    rng_c = np.linalg.norm(P - CAM_START, axis=1)
    params = {
        "start_range_m": r0, "start_azimuth_deg": float(np.degrees(az)),
        "start_elevation_deg": float(np.degrees(np.arcsin(dz / r0))), "start_altitude_m": float(p0[2]),
        "initial_speed_m_s": float(np.linalg.norm(v0)), "initial_vz_clipped": vz_clipped, "n_segments": int(np.sum(seg_t < T)), "n_sidesteps": len(side),
        "first_sidestep_t_s": (min(s[0] for s in side) if side else np.nan),
        "max_segment_accel_m_s2": float(max(np.linalg.norm(a) for s0_, a in segs if s0_ < T)),
        "max_base_accel_m_s2": acc_base_max, "max_total_accel_m_s2": acc_tot_max,
        "mean_speed_m_s": float(speed.mean()), "max_speed_m_s": float(speed.max()),
        "min_altitude_m": float(P[:, 2].min()), "max_altitude_m": float(P[:, 2].max()),
        "frac_time_altitude_brake": braking / ((n_out - 1) * FINE + 1),
        "final_range_from_centre_m": float(rng_c[-1]),
    }
    return P, V, params


def make_random_scenario(seed, dt=DT, duration=DURATION, sweep_A=SWEEP_A, **path_kw):
    """Scenario dict like trajectories.make_scenario (sweep camera), for the random path of this seed,
    plus 'path': the path parameters."""
    t = np.arange(0.0, duration + 1e-9, dt)
    rng = np.random.default_rng([seed, RNG_STREAM])
    K = Intrinsics()
    for redraws in range(1000):
        p, v, prm = draw_path(rng, t, **path_kw)
        los = np.array([p[0, 0] - CAM_START[0], p[0, 1] - CAM_START[1], 0.0])
        los /= np.linalg.norm(los)
        lat = np.cross(los, [0.0, 0.0, 1.0])
        cam_p, cam_v = camera_path("sweep", t, los, lat, sweep_A=sweep_A)
        horiz = np.linalg.norm((p - cam_p)[:, :2], axis=1)
        u, vv, _, Z = project_box(p[0], 0.35, gimbal_rotation(cam_p[0], p[0], t[0]), cam_p[0], K)
        if horiz.min() >= MIN_HORIZ and is_visible(u, vv, Z, K):
            break
    else:
        raise RuntimeError(f"seed {seed}: no valid path in 1000 draws")
    cam_R = np.array([gimbal_rotation(cam_p[k], p[k], t[k]) for k in range(len(t))])
    r = np.linalg.norm(p - cam_p, axis=1)
    prm.update({"seed": int(seed), "n_redraws": redraws, "min_range_to_camera_m": float(r.min()),
                "frac_time_within_200m": float(np.mean(r <= 200.0)),
                "visible_at_t0": bool(is_visible(u, vv, Z, K))})
    return {"t": t, "drone_p": p, "drone_v": v, "cam_p": cam_p, "cam_v": cam_v, "cam_R": cam_R,
            "drone": "random_path", "camera": "sweep", "path": prm}
