"""EKF tracking a drone's 3D position and range from a moving monocular camera (stub).

The filter is NOT implemented yet. Planned design:

State (world frame), nearly-constant-velocity model:
    x = [p_x, p_y, p_z, v_x, v_y, v_z]            (option: append log(size_m) if the
                                                   target's physical size is unknown)
Process model: x_{k+1} = F(dt) x_k + w, with white-acceleration noise q (tunable).

Measurement (per detection), using the known camera pose (R_wc, t_w) and intrinsics K:
    z = [u, v, w_px]   pixel centroid + apparent box width
    h(x) = project(R_wc^T (p - t_w)) for (u, v);   w_px = fx * size_m / Z_c
The Jacobian H is computed analytically and checked against finite differences in tests.
Range r = ||p - t_w|| and its variance come from the state covariance.

Planned API:
- `DroneEKF(q, R, size_m, init_range_prior)`
- `predict(dt)`
- `update(z, cam_pose, K) -> innovation, S` with chi-square gating (reject clutter or birds)
- `state`, `cov`, `range_estimate() -> (r, sigma_r)`
- Initialisation from the first detection: back-project the bearing, with range taken
  from box size (or a wide prior when size is unknown).
"""
