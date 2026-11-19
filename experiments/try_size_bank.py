"""NEGATIVE RESULT, kept for reproducibility: a bank of EKFs over size hypotheses.

Idea (range-parameterised EKF, as in bearings-only tracking): run one EKF per size hypothesis
S_i = 0.10, 0.15, ..., 0.80 m, each with a narrow prior N(S_i, sigma_i^2), weighted by the
Gaussian prior N(0.35, 0.15^2) at S_i. After every detection, multiply each weight by that
filter's measurement likelihood N(innovation; 0, S_inn). The output is the mixture mean and
covariance. It was meant to fix the over-confidence of the single EKF in straight-on geometry.

It made things WORSE (see STUDY.md): the weights lock onto wrong hypotheses, because the
hypotheses differ in small modelled effects (velocity prior, process noise, range-dependent
pixel noise) that the likelihood then over-weights.

    python experiments/try_size_bank.py straight 0.02
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ekf.ekf import DroneEKF  # noqa: E402
from experiments.mc import VARIANTS, draw_true_size  # noqa: E402
from sim.camera import Intrinsics, is_visible, project_box  # noqa: E402
from sim.noise import DetectorNoise, PoseNoise, noisy_pose  # noqa: E402
from sim.trajectories import make_scenario  # noqa: E402

MEANS = np.arange(0.10, 0.81, 0.05)


def run(args):
    camera, seed, sigma_i = args
    scn = make_scenario("approach_recede", camera)
    K, pn, dn = Intrinsics(), PoseNoise(), DetectorNoise()
    rng = np.random.default_rng(seed)
    S_true = draw_true_size(VARIANTS["size_prior"], rng)
    n, dt = len(scn["t"]), 0.1
    _, _, e_t, e_th = pn.sample_errors(n, dt, rng)
    w = np.exp(-0.5 * ((MEANS - 0.35) / 0.15) ** 2)
    w /= w.sum()
    filters = [DroneEKF(K, size_prior=(m, sigma_i), pose_noise=pn) for m in MEANS]
    err, sig, r_true = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for k in range(n):
        p, R, c = scn["drone_p"][k], scn["cam_R"][k], scn["cam_p"][k]
        u, v, wd, Z = project_box(p, S_true, R, c, K)
        r_true[k] = np.linalg.norm(p - c)
        det = dn.measure(u, v, wd, rng) if is_visible(u, v, Z, K) else None
        R_meas, t_meas = noisy_pose(R, c, e_t[k], e_th[k])
        if det is not None:
            R_det = np.diag([dn.sigma_uv(det[2]) ** 2] * 2 + [dn.sigma_logw(det[2]) ** 2])
        if filters[0].x is None:
            if det is None:
                continue
            for f in filters:
                f.initialize(det, R_det, R_meas, t_meas)
        else:
            log_lik = np.zeros(len(filters))
            for i, f in enumerate(filters):
                f.predict(dt)
                if det is not None:
                    _, S_inn, nis = f.update(det, R_det, R_meas, t_meas)
                    log_lik[i] = -0.5 * nis - 0.5 * np.log(np.linalg.det(2 * np.pi * S_inn))
            if det is not None:
                lw = np.log(w) + log_lik
                w = np.exp(lw - lw.max())
                w = np.maximum(w / w.sum(), 1e-12)
                w /= w.sum()
        ranges = np.array([f.range_estimate(t_meas) for f in filters])
        mean = np.sum(w * ranges[:, 0])
        sig[k] = np.sqrt(np.sum(w * (ranges[:, 1] ** 2 + (ranges[:, 0] - mean) ** 2)))
        err[k] = mean - r_true[k]
    return err, sig, r_true


if __name__ == "__main__":
    camera, sigma_i = sys.argv[1], float(sys.argv[2])
    with Pool(8) as pool:
        res = pool.map(run, [(camera, 1000 + i, sigma_i) for i in range(200)])
    E, S, T = (np.array([r[j] for r in res]) for j in range(3))
    rel = E / T
    print(f"{camera} sigma_i={sigma_i}: RMS rel err (t>=10 s) {np.sqrt(np.nanmean(rel[:, 100:] ** 2)):.3f}, "
          f"median |rel| {np.nanmedian(np.abs(rel[:, 100:])):.3f}, "
          f"2-sigma coverage {np.nanmean((np.abs(E) <= 2 * S)[:, 20:]):.3f}, "
          f"ANEES range {np.nanmean((E / S)[:, 20:] ** 2):.2f}")
