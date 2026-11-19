"""Synthetic measurement noise: detector box noise, missed detections, camera pose noise.

Detector noise (per detection, all independent between frames):
  - box centre:  u_meas = u + n_u,  v_meas = v + n_v,   n ~ N(0, sigma_uv(w)^2)
  - box width:   w_meas = w * exp(n_w),                 n_w ~ N(0, sigma_logw(w)^2)
    i.e. MULTIPLICATIVE noise: a 10% error is as likely on a 5 px box as on a 50 px box.
    In log space it is plain additive Gaussian noise: log w_meas = log w + n_w.
  - miss:        the detection is dropped with probability 1 - p_detect(w).
All three depend on the true box width w (small boxes are noisier and missed more often).

Where the numbers come from:
  If experiments/detector_noise.json exists (written by detector/measure_noise.py from
  matched true positives on the held-out DUT test set), sigma_uv, sigma_logw and the
  detection rate are interpolated from its box-width bins (outside the measured widths the
  default curve's shape is used to extrapolate). Otherwise these documented DEFAULTS are used:
      sigma_uv(w)   = sqrt(0.5^2 + (0.05 w)^2)   px   (0.5 px floor + 5% of the box size)
      sigma_logw(w) = sqrt(0.08^2 + (1.0 / w)^2)      (8% + a 1 px edge error relative to w)
      p_detect(w)   = 0.95 / (1 + exp(-(w - 6) / 1.5)) (50% of the max at 6 px wide)

Camera pose noise (simulated GPS/INS; the EKF receives truth + this noise):
  Each pose error is a slowly drifting part plus a small white jitter:
      e = b + j
  b : first-order Gauss-Markov ("random drift"), stationary std sigma, correlation time tau:
          b_{k+1} = a b_k + sqrt(1 - a^2) * sigma * n_k,   a = exp(-dt / tau)
      Position: sigma = 1.5 m per axis, tau = 60 s.  Attitude: sigma = 0.5 deg, tau = 30 s.
  j : white noise, independent every frame. Position 0.05 m, attitude 0.02 deg.
  Why not plain white noise at 1.5 m? A real GPS/INS solution is smooth: it does not jump
  1.5 m between two frames 0.1 s apart. Its error is a slowly varying offset. White 1.5 m
  jitter makes the camera appear to teleport, and the filter then explains "the camera moved
  but the bearing did not change" as "the drone is very far away", biasing range HIGH
  (+22% mean range error with an earlier 7-state filter during development; that version is
  not kept, see STUDY.md section 8, item 1). That case is kept as a stress test, white_pose_noise():
  PoseNoise(tau_pos=0, tau_att=0, sigma_pos_white=0, sigma_att_white=0).
  Position:  t_meas = t_true + e_t
  Attitude:  R_meas = R_true @ Rot(e_theta)

All functions take an explicit numpy Generator for reproducibility.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.camera import rotation_from_vector

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_NOISE_JSON = ROOT / "experiments" / "detector_noise.json"


class DetectorNoise:
    """Box-size-dependent detector noise. `source` says where the numbers came from."""

    def __init__(self, json_path=DETECTOR_NOISE_JSON, use_json=True):
        self.table = None
        self.source = "defaults"
        if use_json and Path(json_path).exists():
            d = json.loads(Path(json_path).read_text())
            bins = d["bins"]
            # interpolate in log(width): bins are roughly log-spaced
            self.table = {
                "logw": np.log([b["w_median_px"] for b in bins]),
                "sigma_uv": np.array([b["sigma_center_px"] for b in bins]),
                "sigma_logw": np.array([b["sigma_logw"] for b in bins]),
                "p_detect": np.array([b["recall"] for b in bins]),
            }
            self.source = str(Path(json_path).name)

    def _interp(self, key, w, default_fn):
        """Interpolate a measured curve in log(width). Outside the measured widths we scale the
        default curve so it joins the measured end point (e.g. detection rate keeps falling for
        boxes smaller than any the test set had, instead of staying flat)."""
        logw = np.log(max(w, 1e-3))
        lo, hi = self.table["logw"][0], self.table["logw"][-1]
        if logw < lo or logw > hi:
            edge = lo if logw < lo else hi
            edge_val = float(np.interp(edge, self.table["logw"], self.table[key]))
            return edge_val * default_fn(w) / default_fn(float(np.exp(edge)))
        return float(np.interp(logw, self.table["logw"], self.table[key]))

    @staticmethod
    def default_sigma_uv(w):
        return float(np.sqrt(0.5**2 + (0.05 * w) ** 2))

    @staticmethod
    def default_sigma_logw(w):
        return float(np.sqrt(0.08**2 + (1.0 / max(w, 1e-3)) ** 2))

    @staticmethod
    def default_p_detect(w):
        return float(0.95 / (1.0 + np.exp(-(w - 6.0) / 1.5)))

    def sigma_uv(self, w):
        if self.table is not None:
            return self._interp("sigma_uv", w, self.default_sigma_uv)
        return self.default_sigma_uv(w)

    def sigma_logw(self, w):
        if self.table is not None:
            return self._interp("sigma_logw", w, self.default_sigma_logw)
        return self.default_sigma_logw(w)

    def p_detect(self, w):
        if self.table is not None:
            return min(1.0, self._interp("p_detect", w, self.default_p_detect))
        return self.default_p_detect(w)

    def measure(self, u, v, w, rng, allow_miss=True):
        """Noisy detection (u, v, w) of a true box, or None if missed."""
        if allow_miss and rng.random() > self.p_detect(w):
            return None
        su, sl = self.sigma_uv(w), self.sigma_logw(w)
        return (u + su * rng.standard_normal(),
                v + su * rng.standard_normal(),
                w * np.exp(sl * rng.standard_normal()))


@dataclass
class PoseNoise:
    """Simulated GPS/INS errors on the camera pose (drift b + white jitter j, see top)."""
    sigma_pos: float = 1.5                      # m, drift std per axis
    tau_pos: float = 60.0                       # s, drift correlation time (0 = white)
    sigma_pos_white: float = 0.05               # m, white jitter per axis
    sigma_att: float = np.deg2rad(0.5)          # rad, drift std per axis
    tau_att: float = 30.0                       # s (0 = white)
    sigma_att_white: float = np.deg2rad(0.02)   # rad, white jitter per axis

    def sample_errors(self, n, dt, rng):
        """Returns (b_t, b_theta, e_t, e_theta), each (n, 3):
        b = drift part (part of the EKF state, used as truth for NEES), e = total error."""
        b_t = _gauss_markov(n, dt, self.sigma_pos, self.tau_pos, rng)
        b_th = _gauss_markov(n, dt, self.sigma_att, self.tau_att, rng)
        e_t = b_t + self.sigma_pos_white * rng.standard_normal((n, 3))
        e_th = b_th + self.sigma_att_white * rng.standard_normal((n, 3))
        return b_t, b_th, e_t, e_th


def white_pose_noise(sigma_pos=1.5, sigma_att=np.deg2rad(0.5)):
    """Stress-test model: the whole pose error is white (independent every frame)."""
    return PoseNoise(sigma_pos=sigma_pos, tau_pos=0.0, sigma_pos_white=0.0,
                     sigma_att=sigma_att, tau_att=0.0, sigma_att_white=0.0)


def _gauss_markov(n, dt, sigma, tau, rng):
    """n x 3 first-order Gauss-Markov sequence with stationary std `sigma`
    (white noise when tau == 0). Starts in its stationary distribution."""
    white = rng.standard_normal((n, 3)) * sigma
    if tau <= 0:
        return white
    a = np.exp(-dt / tau)
    e = np.zeros((n, 3))
    e[0] = white[0]
    for k in range(1, n):
        e[k] = a * e[k - 1] + np.sqrt(1 - a**2) * white[k]
    return e


def noisy_pose(R_true, t_true, e_t, e_theta):
    """Apply one frame's pose errors: what the GPS/IMU reports."""
    return R_true @ rotation_from_vector(e_theta), t_true + e_t
