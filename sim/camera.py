"""Pinhole camera model mounted on a moving platform (stub, not implemented yet).

Planned contents:
- `Intrinsics`: fx, fy, cx, cy, image width/height, and optional radial distortion
  (default: zero distortion, sized to match the detector's input resolution).
- `CameraPose`: world-from-camera rotation R_wc and position t_w at time t. The camera
  platform moves along its own trajectory (see trajectories.py), so the filter must
  account for ego-motion.
- `project(p_w, pose, K) -> (u, v), in_front`: world point to pixel coordinates.
- `project_box(p_w, size_m, pose, K) -> (u, v, w_px, h_px)`: apparent bounding box of a
  drone with physical extent `size_m`. Box size is what makes monocular range observable:
  w_px ~ fx * size_m / Z.
- `bearing(u, v, K) -> unit ray in the camera frame`, the inverse of `project`.
- Field-of-view and visibility checks, used to drop detections outside the image.
Conventions: right-handed world frame (ENU), camera frame x-right, y-down, z-forward.
"""
