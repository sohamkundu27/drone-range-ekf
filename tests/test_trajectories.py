"""Trajectory tests for the sweep-in-place study (sim/trajectories.py): the new 'sweep' / 'hover'
cameras, the new 'dash' / 'jink' / 'orbit' targets, and that the existing paths did not change."""
import hashlib

import numpy as np
import pytest

from sim.trajectories import (CAM_START, CAMERA_PATHS, DRONE_PATHS, JINK_AMP, JINK_PERIOD, ORBIT_R,
                              ORBIT_SPEED, make_scenario, sweep_period)

PEAK_ACC = 15.0 * (2 * np.pi / 12.0) ** 2          # 4.112 m/s^2

# SHA-256 of every array make_scenario returned for the existing paths, recorded BEFORE the sweep
# study's edits (commit 6060b93). Any change to an existing path, even in the last bit, fails here.
BEFORE = {
    "approach_recede/straight": "c9835797cc387df4bde5c4ee8ee8f30ce8cf68396f47f74b48f89a6354d56012",
    "approach_recede/lateral": "70ec06252a00bc39a3a01fa3cf074306230e711aefce7a03eb465728ab3eea52",
    "approach_recede/weaving": "a8ece8e319e148862567ffc3d7d8fb9aefa1eea53715386a2e4b2b6d3f3d0c71",
    "constant_velocity/straight": "1fcfca4f65035dc6c7ef43ccde608c528ddd3ef12179034ccee095b6e7d11d28",
    "constant_velocity/lateral": "262fb3c0d2e0e686357b4c85dc54a781c40b813b2ccbe73693969673fcf4986e",
    "constant_velocity/weaving": "898f1c4196d7a6e8f42406710702706648ae46b89741cced31e3b11048ce48fc",
    "gentle_turn/straight": "36c2448ab4f0e055a754bd1a49841c92c554bbc040fe1cb276f39a3edc108563",
    "gentle_turn/lateral": "e392dc80cc20ceeb50ee0b6155b92c1e402747c7bed5929904c853aed819a459",
    "gentle_turn/weaving": "e7049f283dabcc78175e249837aaef98fb89f80399d36fdbe4b1f06b9ac3c9f3",
}


def scenario_hash(s):
    h = hashlib.sha256()
    for k in ("t", "drone_p", "drone_v", "cam_p", "cam_v", "cam_R"):
        h.update(np.ascontiguousarray(s[k]).tobytes())
    return h.hexdigest()


def num_acc(v, dt):
    return np.gradient(v, dt, axis=0)


def test_existing_paths_bit_identical():
    for d in DRONE_PATHS:
        for c in CAMERA_PATHS:
            assert scenario_hash(make_scenario(d, c)) == BEFORE[f"{d}/{c}"], f"{d}/{c} changed"


@pytest.mark.parametrize("A", [5.0, 10.0, 15.0, 30.0])
def test_sweep_width_and_peak_accel(A):
    s = make_scenario("approach_recede", "sweep", sweep_A=A, duration=40.0, dt=0.01)
    off = s["cam_p"] - CAM_START
    lat = np.array([1.0, 0.0, 0.0])                       # initial LOS is +y, so the sweep axis is +x
    assert np.allclose(off - np.outer(off @ lat, lat), 0.0)          # no motion off the sweep axis
    assert np.abs(off @ lat).max() <= A + 1e-9
    assert np.abs(off @ lat).max() > 0.999 * A                       # reaches +/-A
    acc = np.linalg.norm(num_acc(s["cam_v"], 0.01), axis=1)
    assert abs(acc.max() / PEAK_ACC - 1) < 0.01
    assert abs(A * (2 * np.pi / sweep_period(A)) ** 2 / PEAK_ACC - 1) < 1e-12   # analytic, exact


def test_sweep_velocity_is_derivative_of_position():
    s = make_scenario("approach_recede", "sweep", dt=0.001, duration=12.0)
    assert np.allclose(np.gradient(s["cam_p"], 0.001, axis=0)[1:-1], s["cam_v"][1:-1], atol=1e-4)


def test_hover_constant():
    s = make_scenario("approach_recede", "hover")
    assert np.all(s["cam_p"] == CAM_START) and np.all(s["cam_v"] == 0)


def test_dash_speed_profile_and_closest_approach():
    s = make_scenario("dash", "hover", dt=0.01)
    t, sp = s["t"], np.linalg.norm(s["drone_v"], axis=1)
    assert np.allclose(sp[t < 10], 5.0)
    ramp = (t > 10.01) & (t < 14.99)
    assert np.allclose(np.gradient(sp, 0.01)[ramp], 2.0, atol=1e-6)
    assert np.allclose(sp[t >= 15], 15.0)
    r = np.linalg.norm(s["drone_p"] - CAM_START, axis=1)
    assert abs(r[0] - 200.0) < 1e-6
    k = r.argmin()
    assert 21.0 <= t[k] <= 22.0 and abs(r[k] - 10.0) < 0.05          # 10.0 m at t = 21.65 s
    assert r[-1] > 200                                               # keeps going past closest approach
    pos_v = np.gradient(s["drone_p"], 0.01, axis=0)
    assert np.allclose(pos_v[1:-1], s["drone_v"][1:-1], atol=0.02)


@pytest.mark.parametrize("phase", [0.0, 1.0, 4.0])
def test_jink_lateral_accel_and_base_path(phase):
    s = make_scenario("jink", "weaving", dt=0.01, jink_phase=phase)
    base = make_scenario("approach_recede", "weaving", dt=0.01)
    off = s["drone_p"] - base["drone_p"]
    path = base["drone_v"][0] / np.linalg.norm(base["drone_v"][0])
    assert np.allclose(off @ path, 0.0, atol=1e-9) and np.allclose(off[:, 2], 0.0)   # horizontal, perpendicular
    assert abs(np.linalg.norm(off, axis=1).max() - JINK_AMP) < 1e-3
    acc = np.linalg.norm(np.gradient(s["drone_v"] - base["drone_v"], 0.01, axis=0), axis=1)
    peak = JINK_AMP * (2 * np.pi / JINK_PERIOD) ** 2
    assert abs(peak - 5.48) < 0.01 and abs(acc[1:-1].max() / peak - 1) < 0.01


def test_orbit_radius_speed_rate():
    s = make_scenario("orbit", "sweep", dt=0.01)
    rel = s["drone_p"][:, :2] - CAM_START[:2]
    assert np.allclose(np.linalg.norm(rel, axis=1), ORBIT_R)
    assert np.allclose(np.linalg.norm(s["drone_v"], axis=1), ORBIT_SPEED)
    th = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    assert np.allclose(np.gradient(th, 0.01), 0.1)                   # counter-clockwise, 5.73 deg/s
    assert abs(np.degrees(th[-1] - th[0]) - 229.2) < 0.1
    assert np.allclose(rel[0], [0.0, ORBIT_R])                       # starts on the axis perpendicular to the sweep
    k = np.argmin(np.abs(th - np.pi))                                # lines up with the sweep (x) axis
    assert abs(s["t"][k] - np.pi / 2 / 0.1) < 0.01                   # t = 15.7 s
