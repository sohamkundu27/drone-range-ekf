# STUDY.md: how this project works, explained from scratch

This is a study guide for presenting the project. It explains every component in plain
English, derives the math, lists what went wrong along the way, and ends with 25 questions an
engineer might ask (with short answers). Numbers quoted here come from
`experiments/results_sim.md` and `experiments/detector_results.md`. **Range accuracy is from
simulation only.**

---------------------------------------------------------------------------------------------

## 0. The problem in one paragraph

A camera on a moving platform (another drone, a vehicle, a mast on a truck) sees a small
drone. A neural network draws a box around it. From that box we want the drone's **3D position,
velocity and range**, plus an **honest uncertainty**: "it is 80 m away, ±6 m", where the ±6 m is
actually right about 95% of the time at 2σ. One camera cannot measure depth directly, so
range has to come from (a) how big the drone looks, which needs its true size, which we only
know roughly, and/or (b) how the bearing changes as the camera moves (triangulation over time).
The Extended Kalman Filter (EKF) fuses both.

Pipeline:

```
image -> YOLO detector -> box (u, v, w) --+
                                          +-> EKF (state p, v, S, pose drift) -> position, velocity,
GPS/INS -> camera pose (R, t) ------------+                                     range ± sigma
```

---------------------------------------------------------------------------------------------

## 1. The detector (`detector/`)

* **Model:** YOLO26s (Ultralytics), 4 classes: `drone`, `bird`, `airplane`, `helicopter`.
  Birds and aircraft are trained as their own classes (hard negatives), so the detector learns
  to call them something other than "drone" and the tracker is not fed confusers.
* **Data:** DUT Anti-UAV for drones (5,200 train images), COCO and Open Images for birds,
  airplanes and helicopters. Det-Fly is never trained on; it is a cross-dataset test.
* **Rebalancing (`scripts/rebalance_train.py`):** the raw union had 13,389 negative images and
  5,200 drone images (drones = 28%). We keep all drone images and 5,000 negatives (drones = 51%).
  The negatives kept are ranked by (small box) + (smooth, sky-like background, measured as the
  fraction of low-gradient pixels outside the boxes), because the confusers that matter for an
  anti-drone camera are small birds against the sky, not close-up photos of parrots.
  Median box size of kept birds dropped from 28% to 10% of the image.
* **Image size 1280:** the median DUT drone is ~32 px in a 1920×1080 frame. At the default 640
  input it would shrink to ~11 px, too small to detect reliably. At 1280 it is ~21 px.
* **Training:** 7-hour time budget, batch 8, early stopping (patience 30 epochs on validation
  fitness). A first attempt with patience 15 was stopped by hand after 4 epochs: validation
  mAP *dropped* after epoch 1 (the COCO-pretrained head is briefly good at birds/airplanes
  before re-training starts), so patience 15 risked stopping at epoch 16 on a bad model.
  The final run stopped early at epoch 106 (best epoch 76) after 6.0 hours.
* **Results** (full tables in `experiments/detector_results.md`):
  * DUT Anti-UAV test (in-domain, 2,245 drones): P 0.987, R 0.967, mAP50 0.979, mAP50-95 0.749.
  * Other classes (COCO + Open Images test): bird mAP50 0.664 (R 0.59, the weakest class),
    airplane 0.856, helicopter 0.944.
  * False "drone" boxes on 1,534 held-out bird/airplane/helicopter images: 5 at conf ≥ 0.25,
    1 at conf ≥ 0.5. Caveat: these are ordinary photos, mostly not tiny birds against sky.
  * Det-Fly **preliminary** (1,000 images, one sequence, never trained on): P 0.842 but
    **R 0.285, mAP50 0.304**. This is a large domain gap. Looking at missed frames: Det-Fly
    often films the drone from above against farmland, roads and sports fields, or against a
    dark hillside at dusk, while DUT drones are mostly against sky. Size is not the cause (the
    missed drones are not smaller than the detected ones). The fix would be training data with
    ground backgrounds (e.g. the rest of Det-Fly, with a proper train/test split by sequence).
  * Latency (**desktop RTX 3080**, batch 1, end-to-end): PyTorch FP16 5.4 ms (187 FPS),
    PyTorch FP32 7.0 ms, ONNX Runtime (CUDA) FP16 8.0 ms, FP32 11.1 ms. ONNX Runtime was
    *slower* than PyTorch here; TensorRT was not benchmarked.
* **Measured box noise (`detector/measure_noise.py`):** on the held-out DUT test set, we match
  predictions to ground truth and measure, as a function of box width, the centre error (px),
  the width error as log(w_pred / w_true), and the detection rate. These numbers drive the
  simulator, so the simulated detector is as noisy as the real one, at least on DUT-like data.
  DUT images have mixed resolutions, so pixels are rescaled to the simulator's 1920-wide frame.
  Measured: centre σ ≈ 0.9 px for 16-32 px boxes, rising to ~2.9 px for > 160 px boxes; width
  σ ≈ 9% at 16-24 px, ~5% at 32-96 px, ~3% for large boxes; recall 0.96-0.99. Real errors have
  heavier tails than a Gaussian (4-14% of errors beyond 3 robust σ). **Gap:** DUT has almost no
  test drones below 16 px, but a 0.35 m drone at 200 m is only 4.4 px wide. Below 16 px the
  simulator uses its default curves, scaled to match the measured 16-24 px bin.

---------------------------------------------------------------------------------------------

## 2. The pinhole camera (`sim/camera.py`)

**Frames.** World = ENU (x east, y north, z up). Camera = x right, y down, z forward (the
optical axis). The camera pose is a rotation `R_wc` (camera axes written in world coordinates)
and a position `t_w`. A world point in camera coordinates:

    p_c = R_wc^T (p_w - t_w) = (X, Y, Z)

Rotations are orthonormal, so the inverse of `R_wc` is its transpose.

**Projection.** A pinhole camera maps similar triangles: a point at depth Z and sideways
offset X lands at pixel

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

`fx, fy` = focal length in pixels (2500 here, a ~42° horizontal field of view on a 1920×1080
sensor), `cx, cy` = principal point (image centre). Z is the **depth along the optical axis**,
not the range: range = |p_c| = Z·sqrt(1 + (X/Z)² + (Y/Z)²). Near the image centre they are almost
equal.

**Bearing (back-projection).** Inverting the projection gives only a direction:

    ray = [(u - cx)/fx, (v - cy)/fy, 1]   (any point Z * ray projects to the same pixel)

That missing scale is the whole problem of monocular range.

**Range from size.** A drone of physical width S at depth Z appears w pixels wide:

    w = fx * S / Z      =>   Z = fx * S / w

If S is known, one box gives the depth. With S = 0.35 m and fx = 2500: 4.4 px at 200 m,
8.8 px at 100 m, 87 px at 10 m. **A relative error in S is the same relative error in range:**
a 0.60 m drone assumed to be 0.35 m appears 0.35/0.60 = 58% as far away as it really is (-42%).

**Pose error in pixels.** 0.5° of attitude error = fx · 0.0087 rad = **22 px**. That is far larger
than the detector's centre error (~1 px), so the camera pose, not the detector, limits bearing
accuracy. This is why the pose noise must be modelled carefully (section 5).

---------------------------------------------------------------------------------------------

## 3. The simulation (`sim/trajectories.py`, `sim/noise.py`, `experiments/mc.py`)

**Why simulate?** To judge range accuracy and uncertainty we need ground-truth range, which the
image datasets do not have. The simulation gives exact truth, so every error is measurable.

**Drone paths:** `approach_recede` (straight line from 200 m to a 10 m closest approach at
t = 28 s, then away to ~86 m), `constant_velocity` (crossing, ~130 m), `gentle_turn`
(3°/s turn). **Camera paths:** `straight` (4 m/s along the line of sight), `lateral` (4 m/s
sideways), `weaving` (2 m/s forward plus a ±15 m, 12 s side-to-side sine). The camera
gimbal points at the drone with a slow ±2° wobble. 10 Hz, 40 s per run.

A detail worth knowing: for `approach_recede`, the drone's velocity is chosen so that it
passes 10 m from the camera. Because both camera and drone then move at constant velocity, the
**relative** trajectory (drone minus camera) is *identical* for the straight and lateral
cameras. Only relative motion is visible in the image, so those two cases produce the same
measurements, and the filter does the same on both (section 7).

**Detector noise per frame (`DetectorNoise`):** centre u, v get Gaussian noise σ_uv(w);
width gets *multiplicative* noise, w_meas = w·exp(n), n ~ N(0, σ_logw(w)²); the detection is
dropped with probability 1 - p_detect(w). All three depend on box width and come from the real
detector's measurements (or documented defaults if those are missing).

**Camera pose noise (`PoseNoise`):** each error = slow drift + small white jitter.
Position drift σ = 1.5 m with a 60 s correlation time, attitude drift σ = 0.5° with 30 s, and
white jitter of 5 cm and 0.02°. The drift is a first-order Gauss-Markov process:

    b_{k+1} = a·b_k + sqrt(1 - a²)·σ·n_k,    a = exp(-dt/τ)

(stays at standard deviation σ forever, but changes slowly: consecutive frames are almost the
same). This is how real GPS/INS errors behave: they wander slowly rather than jumping 1.5 m
every 0.1 s.

**Monte Carlo:** every configuration is run 200 times with different random seeds (noise,
misses, the drone's true size). The same seeds are reused across configurations, so
comparisons are paired.

---------------------------------------------------------------------------------------------

## 4. Why S (the drone's size) is a state

The width measurement only tells the filter **S / Z**. Suppose we instead fixed S = 0.35 m and
said "the size error is part of the measurement noise". That error is **the same in every
frame** (it is one physical drone). A Kalman filter assumes measurement noise is **independent**
between frames, so after N frames it believes the size error averaged down by √N. After 100
frames it would claim to know the range 10× better than it does: confidently wrong.

Treating S as a **random constant state** fixes this: S has a prior N(0.35, 0.15²), no process
noise (it never changes), and its uncertainty only shrinks when some *other* information
(parallax from camera motion) pins down the range. Then the width tells us S as well. The filter
keeps track of the correlation between S and range: if the drone is bigger, it is farther away.

**The same argument applies to the GPS/INS drift.** It is nearly the same in consecutive frames,
so it is also a state (`b_t`, `b_th`), not white noise. We found this the hard way (section 8).

---------------------------------------------------------------------------------------------

## 5. The EKF (`ekf/ekf.py`)

### 5.1 State (13 numbers)

    x = [p (3), v (3), S, b_t (3), b_th (3)]

drone position and velocity (world), drone width, camera-position drift, camera-attitude drift.

### 5.2 Motion model (predict)

Nearly-constant velocity: the drone keeps its velocity, except for a random (white) acceleration
with spectral density q = 0.1 m²/s³.

    p_{k+1} = p_k + dt·v_k        v_{k+1} = v_k + w_v        S_{k+1} = S_k
    b_{k+1} = a·b_k + w_b         (a = exp(-dt/τ))

In matrix form x⁻ = F x, P⁻ = F P Fᵀ + Q. Integrating a white acceleration over dt gives, per axis,

    Var(p) = q dt³/3,   Cov(p, v) = q dt²/2,   Var(v) = q dt

(`cv_blocks`). The drift blocks get Q = (1 - a²) σ², which keeps their variance at σ².

### 5.3 Measurement model

GPS/INS gives `R_meas, t_meas`. The filter's best guess of the true pose removes the drift:
t = t_meas - b_t, R = R_meas · Rot(-b_th). Then

    p_c = Rᵀ (p - t) = (X, Y, Z)
    h(x) = [ fx X/Z + cx,   fy Y/Z + cy,   log fx + log S - log Z ]

The third measurement is **log(width)**. The width noise is multiplicative (a percentage), and
in log space it becomes additive Gaussian noise, which is what the Kalman update assumes. It
carries the same information as the width itself.

### 5.4 Jacobians (analytic)

The EKF linearises h around the current estimate. With the chain rule, through p_c:

    dh/dp_c = [[fx/Z, 0,    -fx X/Z²],
               [0,    fy/Z, -fy Y/Z²],
               [0,    0,    -1/Z    ]]

    dp_c/dp    = Rᵀ                (rotation into the camera frame)
    dp_c/db_t  = Rᵀ                (only p - t_meas + b_t matters)
    dp_c/db_th = -skew(p_c)        (a small rotation d changes p_c by d × p_c)
    d(log S)/dS = 1/S              (only the width row depends on S)
    dh/dv = 0                      (the camera does not see velocity directly)

`tests/test_ekf.py` checks every column against central finite differences
(f(x+ε) - f(x-ε)) / 2ε. The attitude-drift columns use the small-angle form, which is exact
only to first order in the drift angle (~0.01 rad), so that test allows 3% there and is exact
at zero drift.

### 5.5 Update (iterated EKF)

    y = z - h(x)               innovation (what we saw minus what we expected)
    S = H P Hᵀ + R             innovation covariance
    K = P Hᵀ S⁻¹               Kalman gain (how much to trust the measurement vs the prediction)
    x ← x + K y
    P ← (I - K H) P (I - K H)ᵀ + K R Kᵀ     (Joseph form: same result, stays symmetric/positive)

**Iterated EKF (3 iterations):** after the first update, re-linearise h at the new estimate and
redo the update (a Gauss-Newton step on the same measurement):

    x_{i+1} = x⁻ + K_i [z - h(x_i) - H_i (x⁻ - x_i)]

With iteration 1 this is exactly the plain EKF. We use it because right after initialisation
the range is ~40% uncertain, and a single linearisation at a range that far off makes
the plain EKF badly overconfident (measured below). R contains the detector noise plus the
white part of the pose noise, mapped to pixels through the b_t and b_th columns of H.

**S is kept ≥ 0.05 m** after each update because h uses log S. This only triggers in badly wrong
cases.

### 5.6 Initialisation (first detection)

Direction from the pixel, depth from the width and the size prior: Z0 = fx · 0.35 / w.
Position p0 = t_meas + R_meas · [m_x Z0, m_y Z0, Z0]. The covariance is P0 = J Σ Jᵀ, where J is
the derivative of the initial state with respect to (u, v, log w, S, b_t, b_th) and Σ holds
their variances. The S column links position and size: a bigger drone means farther along the
ray. The result is a long, thin ellipsoid **along the line of sight** (range ±40%, direction
±0.5°). Velocity starts at 0 ± 5 m/s per axis.

### 5.7 Range output

r = |p - t|, with t = t_meas - b_t. Its variance is J P Jᵀ with dr/dp = d and dr/db_t = d
(d = unit line of sight), plus the white GPS jitter. A constant GPS offset moves our estimates
of both the drone and the camera, so it cancels in the range; the strong negative p/b_t
covariance handles that automatically.

---------------------------------------------------------------------------------------------

## 6. NEES and NIS: is the uncertainty honest? (`ekf/nees.py`)

**NEES** (normalised estimation error squared), at one time step of one run:

    e = (x_true - x̂)ᵀ P⁻¹ (x_true - x̂)

If the filter is right about its uncertainty, e has a chi-square distribution with d degrees of
freedom (d = number of components checked: 3 for position, 1 for range), with **mean d**.

**ANEES** = the average of e over N Monte Carlo runs. N·ANEES ~ χ²(N·d), so with N = 200 and
d = 1, the 95% interval for a consistent filter is [0.81, 1.21]; for d = 3 it is [2.67, 3.35].

* ANEES above the interval: errors are bigger than the filter claims. **Overconfident:
  dangerous**, because downstream logic trusts a ±σ that is too small.
* Below: errors are smaller than claimed. **Conservative: safe but wasteful.**

**NIS** = the same formula on the innovation y with covariance S. It needs no ground truth, so
it can be checked on real data too. NIS can look fine while NEES is bad: the filter can explain
the measurements perfectly while being wrong along a direction the measurements cannot see.
That is exactly what happens in the straight-on case.

**2σ coverage** = the fraction of time steps where |range error| ≤ 2σ_r. It should be ~95%.
It is the most intuitive consistency number to present.

---------------------------------------------------------------------------------------------

## 7. Why straight-on (and constant-velocity) camera motion is hard

1. **Bearing alone gives no range.** Every point along the ray projects to the same pixel.
2. **Triangulation needs the bearing to change in a way the drone's own motion cannot explain.**
   The drone's velocity is unknown. If the camera moves at constant velocity, the drone could
   simply be moving with a velocity that cancels it: a near drone moving slowly and a far drone
   moving fast (scaled up together with its size) produce *identical* pixel tracks. Formally,
   scaling (p - c, v - v_c, S) by the same factor λ leaves u, v and log w unchanged at every
   time step. That is the unobservable direction.
3. **Flying straight at the drone** is the extreme case: the bearing barely changes, so there is
   no parallax at all. The box grows (looming), but looming only gives time to contact
   Z / (closing speed), not Z itself, unless the closing speed is known.
4. **What breaks the ambiguity is camera *acceleration*** (weaving), because a constant-velocity
   drone cannot mimic a curved relative path. This is the classic bearings-only result: "the
   observer must manoeuvre". Lateral motion alone is not enough against a moving target. It is
   only enough against a stationary target or one with known velocity.

So with straight or lateral constant-velocity camera motion, the range accuracy can never be
better than the size prior (±43%). The **test** `test_range_not_observable_with_constant_velocity_camera`
shows this with perfect data: the range error stays at -36% forever. With the weaving camera, the
same perfect data converge to <1% and the filter learns S (`test_converges_on_noise_free_data`).

---------------------------------------------------------------------------------------------

## 8. What went wrong while building this (and what we did)

These are reported, not hidden. Each one is reproducible.

1. **White GPS noise biased range high.** The first simulator added independent 1.5 m GPS noise
   every frame. The reported camera position then "jumps" 1.5 m between frames while the true
   bearing does not change. The only explanation the filter can find for "camera moved, bearing
   did not change" is "the drone is very far away", so range was biased **high** (+22% mean
   error, matched-model test, measured during development with the earlier 7-state filter;
   that version is not kept). Real GPS/INS errors drift slowly, so we changed the simulator to
   drift + small jitter and put the drift in the state. The white-noise case is kept as stress
   test D (`D_whitepose_*`): with the final filter and a straight-on camera it is badly
   inconsistent (range ANEES ~500, bias -67%: here the filter shrinks S instead). With a weaving
   camera it is usable (16% RMS, 78% coverage).
2. **The plain EKF was overconfident.** With realistic noise, plain-EKF position ANEES was ~25
   (ideal 3) in the development matched-model test. The iterated EKF (3 iterations) brought it
   to ~5. In the committed experiment C (`C_plainEKF_*`, same scenarios as the main grid) the plain
   EKF has range RMS 145% (straight), 146% (lateral) and 61% (weaving), 2σ coverage 14-25%, and
   91-97% of runs end outside 3σ. The iterated EKF on the same data: 42%, 45% and 12.5%.
3. **Residual over-confidence remains, worst at long range.** With the truth drawn from the
   filter's own model (experiment E, weaving camera), the iterated EKF's position ANEES is 3.9
   at 40 m, 4.8 at 80 m and 10.0 at 150 m (ideal 3; consistent band [2.67, 3.35]); range ANEES
   is 1.8, 2.6 and 7.2 (ideal 1), and range 2σ coverage is 87%, 81% and 60%. Linearisation
   error grows with range uncertainty. With a lateral camera, even model-truth runs are
   inconsistent (position ANEES 15, 10 and 31), which is item 4.
4. **Spurious information in unobservable geometry (the biggest issue).** In the straight and
   lateral cases, the filter's reported range σ shrinks to ~7% while the actual RMS error stays
   ~42-45% (68th percentile 32%), and only 33-36% of time steps are within 2σ. Diagnosis (with perfect data the σ stays
   correctly at the prior level, and adding noise sources one at a time):
   the **box-width noise** causes it. Each noisy width update changes the estimated ratio
   S/range, which rotates the unobservable "scale" direction slightly. The covariance is still
   large along the *old* direction, and the next linearisation partially "observes" it. This is
   the same mechanism as the known EKF-SLAM inconsistency (Huang, Mourikis & Roumeliotis). It
   is not fixed in this project. Known remedies: a scale-free / inverse-depth parameterisation,
   an observability-constrained EKF, or first-estimate Jacobians.
   * Tried, **did not help**: an uninformative velocity prior (σ_v0 = 30 m/s). Same result.
   * Tried, **made it worse**: a bank of EKFs over size hypotheses (`experiments/try_size_bank.py`).
     The weights lock onto wrong sizes (median error 95% straight-on).
   * Tried, **partial mitigation**: telling the filter the width noise is 2× larger
     (experiment F). Weaving coverage improves from 78% to 92% with no loss of accuracy (RMS
     12.5% → 11.5%); straight-on only goes from 33% to 44%, and the RMS error gets worse
     (42% → 52%).
5. **Near-perfect sensors break the Cartesian EKF.** With 1 cm / 0.01° pose noise
   (`D_precise_pose_*`), the measurements are so precise relative to the 40% range uncertainty
   that the linearisation collapses the covariance at the wrong range. More precise sensors
   giving worse consistency is a classic sign that the parameterisation, not the sensors, is
   the problem.

**Practical takeaway:** in the observable case (weaving camera), the filter gives ~6% RMS range
error out to 80 m (13% at 80-120 m, 31% at 120-160 m, 67% at 160-200 m, where the drone is
4-5 px wide), with 78% 2σ coverage: somewhat overconfident, and honest (95%) only with the
tight prior. In the unobservable case, the *error* is what physics
allows (the size prior, ±43%), but the filter's *reported* σ is too small. A deployed system
should not trust the EKF's σ when the camera is not manoeuvring. A simple guard: report
"range limited by size prior" whenever the accumulated parallax angle is small.

---------------------------------------------------------------------------------------------

## 9. Why not stereo or a depth camera?

* **Stereo** measures depth from disparity d = f·B/Z. A disparity error σ_d gives
  σ_Z = Z²·σ_d / (f·B): error grows with the **square** of range. With f = 2500 px, a 0.3 m
  baseline (about what fits on a small drone) and σ_d = 0.5 px: ±1.7 m at 50 m (3%),
  ±6.7 m at 100 m (7%), **±27 m at 200 m (13%)**. But at 200 m the drone is 4.4 px wide and the
  disparity is only 3.75 px, so matching a 4-pixel blob against sky between two cameras is
  unreliable, and σ_d = 0.5 px is optimistic. A 0.01° relative rotation between the cameras
  (thermal flex) is a 0.44 px disparity bias, i.e. a 12% range bias at 200 m. Stereo also needs
  a second synchronised, calibrated camera. It is useful at short range and complements this
  approach rather than replacing it.
* **Depth cameras** (structured light, time-of-flight) are built for indoor ranges (~0.3-10 m).
  Sunlight swamps their active illumination outdoors, and a 0.35 m target at 100 m returns
  almost no signal.
* **LiDAR** works outdoors but is heavy and power-hungry for a small platform, and its sparse
  scan lines may put only one or two points on a 0.35 m drone at 100 m.
* **Monocular + EKF** is cheap, passive and works at any range where the detector works, but its
  range accuracy is bounded by the size prior unless the camera manoeuvres. This project
  quantifies exactly that trade-off.

---------------------------------------------------------------------------------------------

## 10. 25 questions an engineer might ask (with short answers)

1. **Why not just use w = fx·S/Z and skip the filter?** One box gives range only through the size
   prior (±43%), it is noisy frame to frame, and it gives no velocity. The filter fuses many
   frames, adds parallax from camera motion, estimates velocity, and reports uncertainty.
2. **Where do 0.35 ± 0.15 m come from?** Public specs of common drones: small consumer
   quadcopters are ~0.2-0.35 m wide, larger ones ~0.5-0.6 m. A Gaussian is a rough model and
   allows negative sizes in its far tail; true sizes are clipped at 0.10 m in the simulation.
3. **What if the drone is a different type (e.g. a 0.6 m drone)?** That is the `wrong_prior`
   experiment. Range is then underestimated by ~42% at first. With weaving it mostly recovers
   (S is learned from parallax: final S 0.58 for a true 0.60; range bias -8%, but coverage only
   40%). Straight-on it stays ~27% too short, and the filter claims ±6% (coverage 3%).
4. **Why log(width) instead of width?** Width noise is a percentage. In log space it is additive and
   Gaussian, which matches the Kalman assumptions, and the Jacobian is simple (1/S, -1/Z).
5. **Why is S a state and not a fixed parameter?** Its error is constant across frames.
   Treating it as white noise makes the filter average it away and become overconfident (section 4).
6. **Why are the GPS/INS errors states?** Same reason: they drift slowly. When they were simulated
   as white noise, range was biased +22% (section 8.1).
7. **Is the camera pose really known?** No: it is measured with simulated GPS/INS error. The
   filter estimates the drift but cannot remove it entirely. A constant offset cancels in range
   but shifts the absolute (world) position.
8. **Why an iterated EKF instead of a plain EKF?** The plain EKF was badly overconfident (range
   ANEES 87-131 vs 1, RMS error 61-146%, experiment C), because the first linearisation happens
   at a range that can be 40% wrong. Iterating the update re-linearises at the corrected range.
9. **Why not a UKF or a particle filter?** Both are reasonable. A UKF handles curvature better
   but does not fix unobservability. A particle filter can represent the banana-shaped
   range/size uncertainty but costs ~1000× more. The project brief asked for a hand-written
   EKF; the next step would be an inverse-depth parameterisation.
10. **How do you know the Jacobians are right?** `tests/test_ekf.py` compares every analytic
    column against central finite differences at 50 random states.
11. **What does NEES tell you that RMSE doesn't?** RMSE says how big the errors are. NEES says
    whether the filter *knows* how big they are. A filter with small RMSE but huge NEES will
    reject good data or trigger false alarms downstream.
12. **Your filter is inconsistent in straight-on flight. Isn't that a failure?** Yes, and it is
    reported. The error itself is what physics allows (the size prior); the problem is that the
    *reported* σ is ~5× too small. Section 8.4 explains the mechanism and remedies.
13. **Why does lateral motion not help?** Against a drone with unknown constant velocity, a
    constant-velocity camera adds nothing that the drone's velocity cannot absorb. Only
    acceleration (weaving, turns) makes range observable.
14. **How much weaving is needed?** In the simulation, ±15 m at a 12 s period (peak acceleration
    4 m/s²) made range observable at 100-200 m. Roughly, the accumulated bearing change must be
    large compared with the bearing noise (~0.02° jitter plus slowly drifting 0.5°).
15. **What limits accuracy at 200 m?** The drone is ~4 px wide: the width is only good to ~20-30%,
    detections are often missed, and parallax angles are small.
16. **Why 10 Hz and not 30 Hz?** For simulation speed, and because with white noise more frames
    would make the filter look better than real, correlated errors allow. The filter takes any dt.
17. **What happens with missed detections?** The filter just predicts (velocity carries the
    estimate forward) and the uncertainty grows. The detection rate vs box size comes from the
    real detector.
18. **What about false positives (birds)?** The detector has explicit bird/airplane/helicopter
    classes. On 1,534 held-out bird/airplane/helicopter images it produced 5 false drones at
    conf ≥ 0.25 and 1 at conf ≥ 0.5, but those are mostly ordinary photos, not distant birds
    against sky, so the real rate is likely higher. The filter itself does not do gating or
    data association yet (single target, no clutter).
19. **Is the detector noise in the simulation realistic?** Its σ vs box size is measured on real DUT
    test images (robust σ). But it is Gaussian, while the real errors have heavier tails (tail
    fractions are reported), and DUT is one camera/domain. DUT has almost no test drones under
    16 px, so the long-range end (4-16 px) still uses the default curves, scaled to match.
20. **How good is the detector on new data?** Poor. On Det-Fly (**preliminary**: only 1,000 of
    13,271 images, one sequence) recall is 0.285 and mAP50 0.304, vs 0.967 / 0.979 on DUT. The
    misses are drones seen against ground clutter or dark hillsides; DUT is mostly sky. 4K
    frames are downscaled to 1280, but the drones are still ~46 px wide, so size is not the cause.
21. **How fast is it?** 5.4 ms end-to-end with PyTorch FP16 (187 FPS), 7.0 ms FP32; ONNX
    Runtime CUDA 8.0 ms FP16 / 11.1 ms FP32. Those are **desktop RTX 3080** numbers, not
    embedded. Jetson-class hardware would be several times slower; TensorRT was not measured.
22. **Why YOLO26s and not a bigger model?** It fits in 12 GB at 1280 px with batch 8 and trains in
    the 7 h budget. Small objects are limited more by input resolution than by model size.
23. **Does the box width really equal fx·S/Z?** Only approximately: the box depends on the
    drone's orientation (a quadcopter seen from the side vs. diagonally), on propeller blur,
    and on how tightly the detector draws boxes. Orientation-dependent apparent size is part of
    why the size prior is broad. A measured log-width bias is reported for the detector.
24. **Has any of this been tested on real flights?** No. Range accuracy is simulation-only. There
    was no hardware, no synchronised GPS/INS and video, and no ground-truth range.
25. **What would you do next?** (1) An inverse-depth or scale-free parameterisation (or an
    observability-constrained EKF) to fix the over-confidence in unobservable geometry;
    (2) an observability guard that reports "size-prior limited" when parallax is small;
    (3) real flight data with RTK-GPS ground truth; (4) gating and multi-target data association;
    (5) the full Det-Fly and Drone-vs-Bird sets; (6) TensorRT on embedded hardware.
