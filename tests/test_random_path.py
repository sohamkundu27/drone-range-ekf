"""Tests for the random target paths (sim/random_path.py) used by experiments/run_random_paths.py."""
import numpy as np
import pytest

from sim.camera import Intrinsics, is_visible, project_box
from sim.random_path import (A_MAX, R_MAX, R_MIN, SIDESTEP_D, V_MAX, Z_CEIL, Z_FLOOR, draw_path,
                             make_random_scenario)
from sim.trajectories import CAM_START

SEEDS = range(1000, 1040)


@pytest.fixture(scope="module")
def scenarios():
    return [make_random_scenario(s) for s in SEEDS]


def test_deterministic_from_seed():
    a, b = make_random_scenario(1234), make_random_scenario(1234)
    for k in ("drone_p", "drone_v", "cam_p", "cam_R"):
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(a["drone_p"], make_random_scenario(1235)["drone_p"])


def test_start_and_in_view(scenarios):
    K = Intrinsics()
    for s in scenarios:
        p0 = s["drone_p"][0]
        assert R_MIN <= np.linalg.norm(p0 - CAM_START) <= R_MAX + 1e-9
        assert -25.0 - 1e-9 <= p0[2] - CAM_START[2] <= 50.0 + 1e-9
        u, v, _, Z = project_box(p0, 0.35, s["cam_R"][0], s["cam_p"][0], K)
        assert is_visible(u, v, Z, K)
        assert np.all(np.linalg.norm((s["drone_p"] - s["cam_p"])[:, :2], axis=1) >= 2.0)


def test_limits(scenarios):
    for s in scenarios:
        z = s["drone_p"][:, 2]
        assert z.min() >= Z_FLOOR - 0.05 and z.max() <= Z_CEIL + 0.05
        # base speed is capped at 15 m/s; each sidestep adds at most 5 pi / 2 = 7.9 m/s sideways (its direction is
        # fixed at its start while the base velocity may turn, and two sidesteps can overlap, so only a loose bound)
        assert np.linalg.norm(s["drone_v"], axis=1).max() <= V_MAX + 2 * SIDESTEP_D * np.pi / 2
        assert s["path"]["max_base_accel_m_s2"] <= A_MAX + 1e-6


def test_velocity_matches_position(scenarios):
    for s in scenarios[:10]:
        fd = np.diff(s["drone_p"], axis=0) / 0.1
        mid = 0.5 * (s["drone_v"][1:] + s["drone_v"][:-1])
        assert np.abs(fd - mid).max() < 0.5          # Euler at 0.01 s vs 10 Hz samples


def test_sidestep_is_5m_sideways():
    """With no random accelerations, a sidestep shifts the path by 5 m horizontally, perpendicular to the velocity."""
    t = np.arange(0.0, 40.0 + 1e-9, 0.1)
    checked = 0
    for seed in range(60):
        p, v, prm = draw_path(np.random.default_rng(seed), t, a_max=0.0, p_sidestep=1.0)
        t0 = prm["first_sidestep_t_s"]
        if prm["frac_time_altitude_brake"] > 0 or not (1.0 < t0 < 30.0):
            continue                                   # want constant base velocity around one clean sidestep
        k0, k1 = int(np.floor(t0 / 0.1)) - 2, int(np.ceil((t0 + 1.0) / 0.1)) + 2
        vb = v[k0]                                     # base velocity before the sidestep
        if np.linalg.norm(v[k1] - vb) > 1e-9 or np.hypot(vb[0], vb[1]) < 0.5:
            continue                                   # a second sidestep overlaps, or no horizontal direction
        shift = p[k1] - p[k0] - vb * (t[k1] - t[k0])
        assert abs(np.linalg.norm(shift) - SIDESTEP_D) < 0.05
        assert abs(shift[2]) < 1e-9 and abs(shift[:2] @ vb[:2]) < 1e-6
        checked += 1
    assert checked >= 5
