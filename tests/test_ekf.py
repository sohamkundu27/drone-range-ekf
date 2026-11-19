"""EKF tests: analytic Jacobians vs. finite differences, convergence on noise-free data,
and NEES consistency when the truth follows the filter's own model."""
import numpy as np

from ekf.ekf import IBR, IBT, NX, measurement, measurement_jacobian
from ekf.nees import anees_bounds
from experiments.mc import VARIANTS, run_one
from sim.camera import Intrinsics, look_at, rotation_from_vector
from sim.noise import DetectorNoise, PoseNoise
from sim.trajectories import cv_with_process_noise, make_scenario

K = Intrinsics()


def numerical_jacobian(f, x, eps=1e-6):
    f0 = f(x)
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = eps
        J[:, i] = (f(x + dx) - f(x - dx)) / (2 * eps)     # central difference
    return J


def setup(rng):
    t_w = rng.uniform(-20, 20, 3)
    p = t_w + np.array([0, 1, 0]) * rng.uniform(15, 200) + rng.normal(0, 3, 3)
    R = look_at(t_w, p + rng.normal(0, 3, 3))
    x = np.concatenate([p, rng.normal(0, 5, 3), [rng.uniform(0.2, 0.8)],
                        rng.normal(0, 1.5, 3), rng.normal(0, np.deg2rad(0.5), 3)])
    return x, R, t_w


def test_measurement_jacobian_matches_numerical():
    """All 13 columns of H, including the camera-drift columns, at random states."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, R, t_w = setup(rng)
        for use_width in (True, False):
            H = measurement_jacobian(x, R, t_w, K, use_width)
            Hn = numerical_jacobian(lambda xx: measurement(xx, R, t_w, K, use_width), x)
            assert H.shape == (3 if use_width else 2, NX)
            # position, velocity, size and position-drift columns: exact
            cols = [i for i in range(NX) if i not in range(IBR.start, IBR.stop)]
            assert np.allclose(H[:, cols], Hn[:, cols], rtol=1e-5, atol=1e-6)
            # attitude-drift columns use the small-angle form, exact only to first order in
            # the drift angle (~0.01 rad here), so allow a 3% relative mismatch
            scale = np.abs(Hn[:, IBR]).max()
            assert np.abs(H[:, IBR] - Hn[:, IBR]).max() < 0.03 * scale


def test_attitude_jacobian_exact_at_zero_drift():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x, R, t_w = setup(rng)
        x[IBR] = 0.0
        H = measurement_jacobian(x, R, t_w, K)
        Hn = numerical_jacobian(lambda xx: measurement(xx, R, t_w, K), x, eps=1e-7)
        assert np.allclose(H[:, IBR], Hn[:, IBR], rtol=1e-4, atol=1e-4)
        assert np.allclose(H[:, IBT], Hn[:, IBT], rtol=1e-5, atol=1e-6)


class PerfectDetector(DetectorNoise):
    """Returns the true box every frame (no noise, no misses)."""

    def __init__(self):
        super().__init__(use_json=False)

    def measure(self, u, v, w, rng, allow_miss=True):
        return (u, v, w)


NO_POSE_NOISE = PoseNoise(sigma_pos=0.0, sigma_pos_white=0.0, sigma_att=0.0, sigma_att_white=0.0)


def run_noise_free(camera, S_true):
    """Perfect detections and camera poses; the filter still assumes its usual noise model."""
    scn = make_scenario("approach_recede", camera)
    out = run_one(scn, VARIANTS["size_prior"], seed=0, pose_noise=NO_POSE_NOISE,
                  det_noise=PerfectDetector(), S_true=S_true, filter_pose_noise=PoseNoise())
    return scn, out, np.abs(out["r_est"] - out["r_true"]) / out["r_true"]


def test_converges_on_noise_free_data():
    """Weaving camera (range observable), true drone 0.55 m but prior mean 0.35 m:
    the first range is ~36% off; with perfect data range, size and velocity must converge."""
    scn, out, rel_err = run_noise_free("weaving", 0.55)
    assert rel_err[0] > 0.3
    assert np.nanmax(rel_err[200:]) < 0.02                   # < 2% after 20 s
    assert abs(out["x"][-1, 6] - 0.55) < 0.01                # learned the size
    assert np.linalg.norm(out["x"][-1, 3:6] - scn["drone_v"][-1]) < 0.2


def test_range_not_observable_with_constant_velocity_camera():
    """Camera flying straight at constant velocity towards a constant-velocity drone: the
    bearing never tells the range apart from the drone's speed, so even with perfect data the
    range error stays at the size-prior error (0.35 / 0.55 - 1 = -36%). This is physics, not a bug."""
    _, out, rel_err = run_noise_free("straight", 0.55)
    assert np.nanmin(rel_err[100:]) > 0.3
    assert abs(out["x"][-1, 6] - 0.35) < 0.02                # size stays at the prior mean


def test_nees_consistent_when_model_matches():
    """Truth generated from the filter's own model (CV + white acceleration with the same q,
    size from the prior, Gauss-Markov pose drift + white jitter, Gaussian box noise).
    The time-averaged position ANEES over the second half should be near 3 (its dof).
    MEASURED (200 runs, see STUDY.md): ~5 at 80 m, ~10 at 150 m, i.e. the iterated EKF is
    mildly OVERCONFIDENT because of linearisation error, worse at long range. This test only
    checks "roughly consistent" (within a factor 2) so it catches real regressions such as the
    plain EKF (ANEES ~25 here) or a wrong Jacobian, without hiding the known residual."""
    q = 0.1
    n_runs = 60
    rng = np.random.default_rng(123)
    t = np.arange(0, 30.0 + 1e-9, 0.1)
    nees_all = []
    for i in range(n_runs):
        p0 = np.array([0.0, 80.0, 45.0]) + rng.normal(0, 5, 3)
        v0 = rng.normal(0, 3, 3)
        dp, dv = cv_with_process_noise(p0, v0, t, q, rng)
        scn = make_scenario("constant_velocity", "weaving", duration=30.0, drone_override=(dp, dv))
        out = run_one(scn, VARIANTS["size_prior"], seed=1000 + i, q=q)
        nees_all.append(out["nees_pos"])
    anees_t = np.nanmean(np.array(nees_all), axis=0)
    late = float(np.mean(anees_t[150:]))
    lo, hi = anees_bounds(3, n_runs)
    print("late ANEES", late, "single-step 95% bounds", lo, hi)
    assert 0.5 * 3 < late < 2.0 * 3
