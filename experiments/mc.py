"""Monte Carlo runner: simulate detections for a scenario and run the EKF on them.

One run:
  1. draw the drone's true size S_true (from the variant's truth distribution),
  2. for every frame: true box from the pinhole model -> noisy detection (or a miss),
     true camera pose -> noisy GPS/INS pose,
  3. EKF: initialise on the first detection, then predict + update every frame,
  4. record estimates, covariances and errors (NaN before initialisation).

Variants (what the filter assumes vs. what the truth is):
  bearing_only : no box-width updates (still initialised from the first box)
  size_prior   : prior S ~ N(0.35, 0.15^2); truth drawn from the same distribution
  tight_prior  : prior S ~ N(0.35, 0.03^2) (drone model known); truth from it
  wrong_prior  : prior S ~ N(0.35, 0.15^2), but the true drone is 0.60 m
  wrong_tight  : prior S ~ N(0.35, 0.03^2), but the true drone is 0.60 m (worst case)
"""
from dataclasses import dataclass

import numpy as np

from ekf.ekf import IP, NX, DroneEKF
from ekf.nees import nees
from sim.camera import Intrinsics, is_visible, project_box
from sim.noise import DetectorNoise, PoseNoise, noisy_pose


@dataclass
class Variant:
    name: str
    prior: tuple            # (mean, std) the filter assumes for S [m]
    truth: tuple            # (mean, std) the true S is drawn from; std 0 = fixed value
    use_width: bool = True


VARIANTS = {
    "bearing_only": Variant("bearing_only", (0.35, 0.15), (0.35, 0.15), use_width=False),
    "size_prior": Variant("size_prior", (0.35, 0.15), (0.35, 0.15)),
    "tight_prior": Variant("tight_prior", (0.35, 0.03), (0.35, 0.03)),
    "wrong_prior": Variant("wrong_prior", (0.35, 0.15), (0.60, 0.0)),
    "wrong_tight": Variant("wrong_tight", (0.35, 0.03), (0.60, 0.0)),
}

MIN_TRUE_SIZE = 0.10    # truth draws below 10 cm are clipped (affects ~5% of size_prior draws)
Q_FILTER = 0.1          # EKF process noise density [m^2/s^3]
N_ITER = 3              # iterated EKF update (3 Gauss-Newton steps); 1 = plain EKF.
                        # The plain EKF was measured to be overconfident here (STUDY.md).


def draw_true_size(variant, rng):
    mean, std = variant.truth
    return max(MIN_TRUE_SIZE, mean + std * rng.standard_normal()) if std > 0 else mean


def run_one(scn, variant, seed, pose_noise=None, det_noise=None, K=None, q=Q_FILTER,
            S_true=None, filter_pose_noise=None, n_iter=N_ITER, sigma_v0=5.0,
            width_sigma_scale=1.0):
    """Simulate one Monte Carlo run on scenario `scn`. Returns a dict of time series.
    pose_noise        : the TRUE GPS/INS error model (sim/noise.py PoseNoise)
    filter_pose_noise : what the FILTER assumes (default: the true model)
    n_iter            : 1 = EKF, >1 = iterated EKF update
    sigma_v0          : initial drone velocity std per axis the filter assumes [m/s]
    width_sigma_scale : the filter assumes width noise this many times the true sigma (R inflation)"""
    rng = np.random.default_rng(seed)
    K = K or Intrinsics()
    pose_noise = pose_noise or PoseNoise()
    det_noise = det_noise or DetectorNoise()
    if S_true is None:
        S_true = draw_true_size(variant, rng)
    if filter_pose_noise is None:
        filter_pose_noise = pose_noise

    t = scn["t"]
    n = len(t)
    dt = t[1] - t[0]
    b_t, b_th, e_t, e_th = pose_noise.sample_errors(n, dt, rng)   # drift part, total error
    ekf = DroneEKF(K, q=q, size_prior=variant.prior, pose_noise=filter_pose_noise,
                   use_width=variant.use_width, n_iter=n_iter, sigma_v0=sigma_v0)

    out = {"t": t, "S_true": S_true, "x": np.full((n, NX), np.nan), "P_diag": np.full((n, NX), np.nan),
           "r_true": np.full(n, np.nan), "r_est": np.full(n, np.nan), "r_sigma": np.full(n, np.nan),
           "nees_pos": np.full(n, np.nan), "nees_state": np.full(n, np.nan), "nees_range": np.full(n, np.nan),
           "nis": np.full(n, np.nan), "detected": np.zeros(n, bool), "w_true": np.full(n, np.nan)}
    for k in range(n):
        # --- ground truth for this frame
        p, R_true, t_true = scn["drone_p"][k], scn["cam_R"][k], scn["cam_p"][k]
        x_true = np.concatenate([p, scn["drone_v"][k], [S_true], b_t[k], b_th[k]])
        out["r_true"][k] = np.linalg.norm(p - t_true)
        u, v, w, Z = project_box(p, S_true, R_true, t_true, K)
        out["w_true"][k] = w

        # --- what the sensors report
        det = det_noise.measure(u, v, w, rng) if is_visible(u, v, Z, K) else None
        R_wc, t_w = noisy_pose(R_true, t_true, e_t[k], e_th[k])
        R_det = None
        if det is not None:
            out["detected"][k] = True
            su, sl = det_noise.sigma_uv(det[2]), det_noise.sigma_logw(det[2])  # from MEASURED width
            sl = sl * width_sigma_scale
            R_det = np.diag([su**2, su**2, sl**2])

        # --- filter
        if ekf.x is None:
            if det is None:
                continue
            ekf.initialize(det, R_det, R_wc, t_w)
        else:
            ekf.predict(dt)
            if det is not None:
                _, _, out["nis"][k] = ekf.update(det, R_det, R_wc, t_w)

        # --- record
        out["x"][k] = ekf.x
        out["P_diag"][k] = np.diag(ekf.P)
        out["nees_pos"][k] = nees(x_true[IP] - ekf.x[IP], ekf.P[IP, IP])
        out["nees_state"][k] = nees(x_true - ekf.x, ekf.P)
        r, sr = ekf.range_estimate(t_w)
        out["r_est"][k], out["r_sigma"][k] = r, sr
        out["nees_range"][k] = ((out["r_true"][k] - r) / sr) ** 2
    return out
