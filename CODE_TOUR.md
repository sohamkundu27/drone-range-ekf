# CODE_TOUR.md: a guided walk through the core code

This walks through the four files that matter for range estimation, function by function:

1. `sim/camera.py` — geometry: how a 3D point becomes a pixel and a box width.
2. `sim/noise.py` — what the "sensors" get wrong: detector noise, missed detections, GPS/INS error.
3. `ekf/ekf.py` — the filter itself.
4. `experiments/mc.py` — one Monte Carlo run: simulate the truth, corrupt it, filter it, score it.

It assumes basic linear algebra (matrix products, transposes, inverses) but no Kalman filter
background. Each function says what math it implements and where STUDY.md explains the idea.
Line numbers are for the current version of each file.

Read in this order: camera → noise → the EKF primer below → ekf.py → mc.py.

---------------------------------------------------------------------------------------------

## 0. The EKF in one page (read this before ekf.py)

A Kalman filter keeps a **belief** about something it cannot see directly (here, the drone's
position, velocity and size). The belief is a Gaussian:

* a **mean** `x` (13 numbers: the best guess), and
* a **covariance** `P` (13×13: how unsure it is, and which errors are linked).

`sqrt(P[i, i])` is the standard deviation of state `i`. An off-diagonal entry `P[i, j]` says
"if I'm wrong about `i` in one direction, I'm probably wrong about `j` in a related way."

Every frame it does two things:

1. **Predict.** Time passes, so move the belief forward with a motion model
   (`x ← F x`), and grow the uncertainty, because the drone might have accelerated
   (`P ← F P Fᵀ + Q`). `Q` is how much new uncertainty one time step adds.
2. **Update.** A measurement `z` arrives. Compute what the filter *expected* to see,
   `h(x)`. The difference `y = z − h(x)` is the **innovation** (the surprise). Then:

   ```
   S = H P Hᵀ + R           how big the surprise could plausibly be
                            (state uncertainty seen through H, plus measurement noise R)
   K = P Hᵀ S⁻¹             Kalman gain: how much to trust this measurement
   x ← x + K y              move the guess toward the measurement
   P ← (I − K H) P          shrink the uncertainty
   ```

`H` is the Jacobian of `h`, the matrix of partial derivatives ∂h/∂x. It answers: "if state `j`
changed a little, how would each predicted measurement change?" The **Extended** Kalman filter
is the version for a non-linear `h` (a pinhole camera divides by depth). It linearises `h`
around the current guess using `H`. That linearisation is the source of most of this
project's trouble (STUDY.md §5.5, §8).

Two checks of honesty (STUDY.md §6):

* **NEES** = eᵀ P⁻¹ e, with e = true state − estimate. It averages to the number of components
  if `P` is honest.
* **NIS** = yᵀ S⁻¹ y. The same check, but it needs no ground truth.

---------------------------------------------------------------------------------------------

## 1. `sim/camera.py` — pinhole geometry (STUDY.md §2)

**Frames.** The world is ENU: x = east, y = north, z = up, in metres. The camera frame is
x = right, y = down, z = forward. The camera pose is `(R_wc, t_w)`:

* `t_w` is where the camera is.
* `R_wc` turns camera-frame directions into world directions. Its **columns** are the camera's
  right/down/forward axes written in world coordinates.

| function | line | what it does / the math |
|---|---|---|
| `Intrinsics` | 28 | Holds `fx, fy` (focal length in pixels, 2500), `cx, cy` (image centre, 960, 540), and the image size 1920×1080. `K()` returns the usual 3×3 calibration matrix. fx = 2500 on a 1920-px-wide image gives a horizontal field of view of 2·atan(960/2500) ≈ 42°. |
| `world_to_camera` | 46 | `p_c = R_wcᵀ (p_w − t_w)`. Subtract the camera position, then rotate into camera axes. `R_wcᵀ` is the inverse of `R_wc`, because rotation matrices are orthonormal. |
| `project` | 51 | The pinhole model: `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`, where `(X, Y, Z) = p_c`. Dividing by Z is perspective (far things look closer to the centre). Returns Z as well, because Z ≤ 0 means the point is behind the camera. Note: Z is *depth along the optical axis*, not range; range = √(X²+Y²+Z²). |
| `project_box` | 59 | Adds the box width: `w_px = fx·S/Z`, where S is the drone's real width in metres. This one line is the whole "range from size" idea: if you know S, then Z = fx·S/w. |
| `bearing` | 67 | The inverse of `project`, except for distance: the unit ray `[(u−cx)/fx, (v−cy)/fy, 1]` normalised. Every point along this ray lands on the same pixel, which is why one pixel gives direction but not range. |
| `unproject` | 74 | The ray scaled to a chosen depth Z: `[(u−cx)/fx·Z, (v−cy)/fy·Z, Z]`. Used by the EKF initialisation and by the round-trip test (`tests/test_camera.py`). |
| `is_visible` | 79 | In front of the camera (Z > 1 m) and inside the image bounds. If not visible, the simulated detector returns nothing. |
| `look_at` | 84 | Builds `R_wc` for a camera pointing at a target. Forward = unit vector to the target; right = forward × up; down = forward × right. The three are stacked as columns. This is the simulated gimbal. |
| `skew` | 99 | The matrix `[a]×` such that `skew(a) @ b = a × b`. It turns a cross product into a matrix product, which is how rotation derivatives are written. |
| `rotation_from_vector` | 106 | Rodrigues' formula: rotation by angle \|θ\| about axis θ/\|θ\|, `R = I + sin(φ)·k + (1 − cos φ)·k²`, with `k = skew(axis)`. For tiny angles it is `I + skew(θ)`. Used to apply small attitude errors. |

**What to remember:** the measurement depends on the drone only through `p_c`, i.e. through
*where the drone is relative to the camera*. That fact is the root of the unobservability in
STUDY.md §7.

---------------------------------------------------------------------------------------------

## 2. `sim/noise.py` — what the sensors get wrong (STUDY.md §1, §3, §4)

### Detector noise

| function | line | what it does / the math |
|---|---|---|
| `DetectorNoise.__init__` | 54 | Loads `experiments/detector_noise.json` if it exists: the real detector's box errors per box-width bin, measured on held-out DUT test images (`detector/measure_noise.py`). It stores four arrays: `log(width)` and, per bin, the centre σ, the log-width σ and the detection rate. `self.source` records which noise model was used, and it is printed in `results_sim.md`. |
| `_interp` | 69 | Looks up a value for box width w by linear interpolation in **log(w)** (the bins are roughly log-spaced). Outside the measured range (below ~22 px or above ~320 px) it uses the *shape* of the default curve, scaled so that it joins the last measured point. This matters a lot: a 0.35 m drone at 200 m is 4.4 px wide, far below the smallest measured bin, so long-range noise here is an **extrapolation**. |
| `default_sigma_uv` | 82 | Default centre noise: `√(0.5² + (0.05·w)²)` px. A 0.5 px floor plus 5% of the box size. |
| `default_sigma_logw` | 86 | Default width noise in log space: `√(0.08² + (1/w)²)`. An 8% error plus a 1-pixel edge error relative to the width, which dominates for tiny boxes. |
| `default_p_detect` | 90 | A logistic curve: `0.95 / (1 + exp(−(w − 6)/1.5))`. Half of the maximum at 6 px; at most 95%. |
| `sigma_uv`, `sigma_logw`, `p_detect` | 93-106 | Use the measured table if loaded, otherwise the defaults. |
| `measure` | 108 | One noisy detection. With probability `1 − p_detect(w)` it returns `None` (a miss). Otherwise it returns `(u + σ_uv·n₁, v + σ_uv·n₂, w·exp(σ_logw·n₃))`, where each n is N(0, 1). The width noise is **multiplicative**: `log w_meas = log w + noise`, so a 10% error is equally likely on a big or a small box (STUDY.md §5.3, Q4). |

### Camera pose (GPS/INS) noise

| function | line | what it does / the math |
|---|---|---|
| `PoseNoise` | 119 | The settings: position drift 1.5 m per axis with a 60 s correlation time; attitude drift 0.5° with 30 s; white jitter 5 cm and 0.02°. |
| `sample_errors` | 128 | Makes the whole error history for one run. **Drift** `b` comes from `_gauss_markov`; the **total** error is `e = b + white jitter`. It returns both: `b` is what the EKF tries to estimate (its true value is used for NEES), and `e` is what actually corrupts the pose. |
| `_gauss_markov` | 144 | A first-order Gauss–Markov process: `b_{k+1} = a·b_k + √(1 − a²)·σ·n_k`, with `a = exp(−dt/τ)`. The variance stays at σ², but consecutive values are strongly correlated (a = 0.998 for dt = 0.1 s, τ = 60 s). The error wanders slowly instead of jumping, like a real GPS/INS solution. τ = 0 gives white noise. |
| `white_pose_noise` | 138 | The stress-test model (experiment D): all pose error white, independent every frame. |
| `noisy_pose` | 158 | What the GPS/IMU reports: `t_meas = t_true + e_t` and `R_meas = R_true @ Rot(e_θ)`. The attitude error is a small rotation applied in the camera frame. Note the size of it: 0.5° × fx = 22 px of apparent shift, versus ~1 px of detector error. **Pose error, not the detector, limits the bearing** (STUDY.md §2). |

**Why the drift matters:** a slowly drifting error is nearly the same in consecutive frames.
The filter must not treat it as independent noise, or it will average it away. That is why
the drift is part of the EKF state (STUDY.md §4, §8.1).

---------------------------------------------------------------------------------------------

## 3. `ekf/ekf.py` — the filter (STUDY.md §4, §5)

### The state (lines 1-53)

```
x = [ p (3), v (3), S, b_t (3), b_th (3) ]        13 numbers
      IP     IV     IS  IBT       IBR             index names used everywhere
```

* `p`, `v`: the drone's position and velocity in the world frame.
* `S`: its physical width (one number, constant).
* `b_t`, `b_th`: the current GPS position drift and IMU attitude drift.

The header docstring (lines 11-22) is the "why S is a state" argument. In short: the size
error is the same in every frame. If it were modelled as per-frame noise, the filter would
average it down by √N and become overconfident. As a state with no process noise, its
uncertainty only shrinks when something (parallax) really tells the filter the range.

### Motion model

| function | line | what it does / the math |
|---|---|---|
| `cv_blocks` | 57 | The **constant-velocity** model for [p, v] with random acceleration of spectral density q. `F = [[I, dt·I], [0, I]]`, i.e. `p ← p + dt·v` and `v ← v`. `Q = q·[[dt³/3·I, dt²/2·I], [dt²/2·I, dt·I]]`. Where Q comes from: a white acceleration integrated over dt changes v with variance q·dt, and p (integrated twice) with variance q·dt³/3, and the two are correlated. With q = 0.1 m²/s³, the velocity can drift by about √(0.1 · 10) ≈ 1 m/s over 10 s. |
| `gm_coeff` | 67 | `a = exp(−dt/τ)`, the Gauss–Markov decay per step (0 for white noise). |

### Measurement model

| function | line | what it does / the math |
|---|---|---|
| `corrected_pose` | 73 | The filter's best guess of the **true** camera pose: remove the estimated drift. `t = t_meas − b_t` and `R = R_meas @ Rot(−b_th)`, undoing how `noisy_pose` added it. |
| `measurement` | 78 | **h(x)**: what the filter expects to see. Put the estimated drone into the corrected camera frame, `(X, Y, Z) = Rᵀ(p − t)`, then predict `[u, v, log w]` with `log w = log fx + log S − log Z`. In log form the width is a sum, so its derivatives are simple. The bearing-only variant returns only `[u, v]`. **Known bug:** nothing prevents Z ≤ 0 here. If an update step puts the drone behind the camera, `log Z` is NaN and the run is ruined (STUDY.md §5.5, §8.6). |
| `d_h_d_pc` | 89 | Derivatives of `[u, v, log w]` with respect to `(X, Y, Z)`: `∂u/∂X = fx/Z`, `∂u/∂Z = −fx·X/Z²`, `∂v/∂Y = fy/Z`, `∂v/∂Z = −fy·Y/Z²`, `∂(log w)/∂Z = −1/Z`. These are just the quotient rule applied to the pinhole formulas. |
| `measurement_jacobian` | 99 | **H = ∂h/∂x** (3×13) by the chain rule: `∂h/∂x = ∂h/∂p_c · ∂p_c/∂x`. Because `p_c = Rᵀ(p − t_meas + b_t)`, both `∂p_c/∂p` and `∂p_c/∂b_t` are `Rᵀ`: moving the drone 1 m east looks the same as the camera's position error moving 1 m. For attitude, a small rotation δ changes `p_c` by about `δ × p_c`, so `∂p_c/∂b_th = −skew(p_c)`, which is exact only at zero drift. Velocity columns are zero: one image shows position, not velocity. Velocity is learned from how positions change over time, through the predict step. Only the width row depends on S: `∂(log S)/∂S = 1/S`. `tests/test_ekf.py` checks every column against finite differences (STUDY.md §5.4). |

### The filter class

| method | line | what it does / the math |
|---|---|---|
| `__init__` | 121 | Stores the settings: process noise q, size prior (0.35, 0.15), initial velocity σ (5 m/s), the pose-noise model the filter *assumes*, bearing-only on/off, the number of update iterations (3), and the minimum size (5 cm). |
| `initialize` | 135 | Builds the first belief from the first detection (STUDY.md §5.6). Direction from the pixel: `m = [(u−cx)/fx, (v−cy)/fy, 1]`. Depth from the width and the prior size: `Z0 = fx·S0/w`. Position: `p0 = t_meas + R_meas·(m·Z0)`. Covariance by **first-order error propagation**: `P0 = J Σ Jᵀ`. Σ holds the variances of the inputs (u, v, log w, S, b_t, b_th), and J says how p0 and the other states depend on each input. The S column is the important one, `∂p0/∂S = R_meas·m·Z0/S0`: a bigger drone means farther away along the same ray. So P0 is a long, thin ellipsoid along the line of sight (about ±43% in range, about ±0.5° in direction), and S and range start correlated. `J[IP, b_t] = −I` with `J[IBT, b_t] = I` creates a negative p/b_t correlation: a GPS error moves the estimated drone and the estimated camera together, so it cancels in range. Velocity starts at 0 ± 5 m/s. |
| `predict` | 172 | `x ← F x` and `P ← F P Fᵀ + Q`, with F and Q assembled from blocks: constant velocity for [p, v], identity for S (no process noise), `a·I` for the drifts. Drift noise is `(1 − a²)·σ²`, which keeps the drift variance at σ² forever. |
| `measurement_noise` | 186 | R for one detection: the detector's σ² plus the **white** part of the pose noise mapped into pixels. Since H's b_t and b_th columns are exactly "∂pixels/∂pose error", the jitter contributes `σ²·H_bt H_btᵀ` and `σ²·H_bθ H_bθᵀ`. |
| `update` | 195 | The **iterated EKF** update (STUDY.md §5.5). It converts the box width to `log w`, then repeats 3 times: linearise at the current iterate `x_i`; compute the innovation `y = z − h(x_i) − H(x⁻ − x_i)`, i.e. the surprise corrected for having linearised away from the prior `x⁻`; compute `S = HPHᵀ + R` and `K = PHᵀS⁻¹`; and set `x_i = x⁻ + K y`. Iteration 1 is exactly the ordinary EKF. Later iterations are Gauss–Newton steps that re-linearise closer to the answer, which matters when the first guess is 40% off in range. After each step S is clamped to ≥ 5 cm, but Z is not checked (the NaN bug). The covariance uses the **Joseph form**, `(I − KH)P(I − KH)ᵀ + KRKᵀ`. It is equal to `(I − KH)P` mathematically, but stays symmetric and positive definite despite rounding. It returns the first-iteration innovation, its covariance and the **NIS** `yᵀS⁻¹y`. |
| `range_estimate` | 224 | Range `r = \|p − (t_meas − b_t)\|` and its σ by linearisation: `σ_r² = J P Jᵀ + jitter²`, with `J = [d on p, d on b_t]`, where d is the unit line of sight. This is where the p/b_t correlation pays off: a common drift largely cancels (STUDY.md §5.7). |

---------------------------------------------------------------------------------------------

## 4. `experiments/mc.py` — one Monte Carlo run (STUDY.md §3, §6)

| item | line | what it does |
|---|---|---|
| `Variant`, `VARIANTS` | 28-41 | Each variant pairs **what the filter assumes** (`prior`) with **what the truth is** (`truth`). `size_prior` and `tight_prior` are honest: the truth is drawn from the prior. `wrong_prior` and `wrong_tight` are deliberately wrong: a 0.60 m drone with a 0.35 m prior. `bearing_only` switches off width updates. |
| constants | 43-45 | `MIN_TRUE_SIZE = 0.10`: truth draws below 10 cm are clipped (P(N(0.35, 0.15²) < 0.10) ≈ 5%). `Q_FILTER = 0.1`. `N_ITER = 3`. |
| `draw_true_size` | 49 | One true size per run: `max(0.10, mean + std·n)`, or the fixed value. |
| `run_one` | 54 | One complete run. See below. |

**`run_one`, step by step:**

1. **Setup (63-77).** One random generator seeded with `seed`, so a run is exactly repeatable.
   Draw the true size, then the whole GPS/INS error history (`sample_errors`). Build the
   filter with the variant's prior and the pose-noise model the *filter* assumes. That model
   can deliberately differ from the true one (experiment D).
2. **Each frame (83-118):**
   * **Truth (85-89):** true drone position, camera pose and the true box `project_box`.
     The true state vector `x_true` includes the true drift, for NEES.
   * **Sensors (92-99):** a noisy detection, or `None` if missed or out of view, and the noisy
     pose. The filter's R uses the noise σ evaluated at the **measured** width, because a real
     filter never knows the true width. `width_sigma_scale` lets experiment F tell the filter
     the width is noisier than it is.
   * **Filter (102-109):** initialise on the first detection; afterwards predict every frame
     and update only when there is a detection. A miss means predict only, and the
     uncertainty grows.
   * **Record (112-118):** estimate and covariance diagonal, position NEES (3 dof), full-state
     NEES (13 dof), range estimate ± σ, and range NEES = `((r_true − r)/σ_r)²` (1 dof).
     Before initialisation everything stays NaN, and so does every frame after the state has
     become NaN (STUDY.md §8.6).
3. **Returns** a dict of time series. `run_experiments.py` calls `run_one` 200 times per
   configuration, with seeds 1000-1199 (the *same* seeds in every configuration, so
   comparisons are paired), and summarises: RMS error, bias, coverage, ANEES, and so on.

**How the pieces connect:**

```
trajectories.make_scenario ──> true drone path, true camera path (R, t)
                                   │
          camera.project_box  <────┤                 noise.PoseNoise.sample_errors
                 │                 │                          │
   noise.DetectorNoise.measure     └──> noise.noisy_pose  <───┘
                 │                               │
                 └──────────> DroneEKF.predict / update <────┘
                                    │
                 range_estimate, NEES (ekf/nees.py) ──> run_experiments.summarize
```

---------------------------------------------------------------------------------------------

## 5. Things to be ready to explain line by line

* **Why log w:** multiplicative noise becomes additive (`noise.py` `measure`; `ekf.py` `measurement`).
* **Why `H[:, IP] == H[:, IBT]`:** only `p − t` is visible, so a camera-position error and a
  drone-position error look identical (`measurement_jacobian`).
* **Why the attitude Jacobian is `−skew(p_c)`:** a small rotation δ moves a vector by about δ × v.
* **Why Joseph form:** numerical safety, not different math (`update`).
* **Why the iterated update:** re-linearising near the answer; and its failure mode, which is
  overshooting behind the camera (`update`, the NaN bug).
* **Where honesty is checked:** NEES/NIS are computed in `run_one`, using `ekf/nees.py`;
  chi-square bounds come from `anees_bounds`.
