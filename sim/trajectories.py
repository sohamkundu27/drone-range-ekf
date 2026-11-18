"""Ground-truth trajectories for the target drone and the camera platform (stub).

Planned contents:
- Target drone motion models: constant velocity, coordinated turn, figure-eight /
  Lissajous, hover with random-walk jitter, and an approach/recede pass (range sweeps
  from ~5 m to ~200 m to stress the range estimate).
- Camera platform motion: static, linear pan, orbit, and a "chase" drone with its own
  attitude changes. Outputs a pose (R_wc, t_w) at each timestamp.
- `sample(traj, t) -> (p, v, a)` at arbitrary times, so camera frame rate and
  detection dropouts can be varied independently.
- A seeded scenario builder that returns a dict of time-aligned arrays
  (t, target_p, target_v, cam_R, cam_t) for experiments/ scripts.
"""
