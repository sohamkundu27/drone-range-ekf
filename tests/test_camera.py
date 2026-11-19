"""Camera model tests: projection / back-projection round trip and basic conventions."""
import numpy as np

from sim.camera import (Intrinsics, bearing, is_visible, look_at, project, project_box,
                        rotation_from_vector, unproject, world_to_camera)

K = Intrinsics()


def random_pose(rng):
    t_w = rng.uniform(-50, 50, 3)
    target = t_w + rng.uniform(-100, 100, 3) + np.array([0, 150, 0])
    return look_at(t_w, target), t_w, target


def test_rotation_is_orthonormal():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R, _, _ = random_pose(rng)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_look_at_target_projects_to_principal_point():
    rng = np.random.default_rng(1)
    R, t_w, target = random_pose(rng)
    u, v, Z = project(target, R, t_w, K)
    assert np.isclose(u, K.cx) and np.isclose(v, K.cy) and Z > 0


def test_camera_axes_convention():
    # camera at origin looking north (+y world); world up (+z) must appear as image "up" (v < cy)
    R = look_at([0, 0, 0], [0, 100, 0])
    u, v, _ = project([0, 100, 10], R, np.zeros(3), K)
    assert v < K.cy and np.isclose(u, K.cx)
    # world east (+x) must appear to the right (u > cx) when looking north
    u, v, _ = project([10, 100, 0], R, np.zeros(3), K)
    assert u > K.cx


def test_project_unproject_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(100):
        R, t_w, target = random_pose(rng)
        p_w = target + rng.normal(0, 5, 3)
        u, v, Z = project(p_w, R, t_w, K)
        p_c = unproject(u, v, Z, K)
        assert np.allclose(R @ p_c + t_w, p_w, atol=1e-8)
        # bearing is the unit vector towards the point
        assert np.allclose(bearing(u, v, K), p_c / np.linalg.norm(p_c), atol=1e-10)


def test_box_width_inverse_depth():
    R = look_at([0, 0, 0], [0, 100, 0])
    _, _, w1, _ = project_box([0, 100, 0], 0.35, R, np.zeros(3), K)
    _, _, w2, _ = project_box([0, 200, 0], 0.35, R, np.zeros(3), K)
    assert np.isclose(w1, K.fx * 0.35 / 100) and np.isclose(w1, 2 * w2)


def test_visibility():
    R = look_at([0, 0, 0], [0, 100, 0])
    assert is_visible(*project([0, 100, 0], R, np.zeros(3), K), K)
    assert not is_visible(*project([0, -100, 0], R, np.zeros(3), K), K)     # behind
    assert not is_visible(*project([200, 100, 0], R, np.zeros(3), K), K)    # outside the image


def test_small_rotation_matches_rodrigues():
    th = np.array([1e-3, -2e-3, 0.5e-3])
    R = rotation_from_vector(th)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.allclose(R, np.eye(3) + np.array([[0, -th[2], th[1]], [th[2], 0, -th[0]], [-th[1], th[0], 0]]), atol=1e-5)
    assert np.allclose(world_to_camera([1, 2, 3], np.eye(3), np.zeros(3)), [1, 2, 3])
