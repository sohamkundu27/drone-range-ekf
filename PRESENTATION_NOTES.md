# PRESENTATION_NOTES.md: suggested 45-minute talk

About 38 minutes of slides plus about 7 minutes of questions. Times are cumulative targets.
Every number below comes from a committed file (`experiments/*.md` / `*.json`). Figures
are in `figures/`. Three labels must appear on the slides where they apply:
**simulation only** (range), **preliminary** (Det-Fly), **desktop GPU** (latency).

---------------------------------------------------------------------------------------------

## Part 1: Problem and approach (0:00 – 0:06)

| # | slide | figure / visual | speaker note (one line) |
|---|---|---|---|
| 1 | Title: monocular drone detection + range EKF with honest uncertainty | — | "One week: a detector, a filter, and a careful look at when the filter's uncertainty can be trusted." |
| 2 | The problem | sketch: camera platform, distant drone, unknown range | "From one moving camera we want the drone's 3D position, velocity and range, plus an uncertainty that is right 95% of the time at 2σ." |
| 3 | Why one camera is hard | pinhole sketch: one pixel = one ray | "A pixel gives a direction, not a distance; range must come from the box size (needs the drone's size) or from parallax (needs camera motion)." |
| 4 | Pipeline | the ASCII pipeline in STUDY.md §0, redrawn as boxes | "Detector box → bearing + width; GPS/INS → camera pose; a 13-state EKF fuses them; evaluated in a NumPy simulation with ground truth." |

## Part 2: Detector (0:06 – 0:14)

| # | slide | figure / visual | speaker note |
|---|---|---|---|
| 5 | Data and rebalancing | table from `experiments/train_balance_stats.json` | "DUT Anti-UAV drones plus COCO/Open Images birds, airplanes and helicopters as hard negatives; rebalanced from 28% to 51% drone images, preferring small confusers against sky." |
| 6 | Training | small table: YOLO26s, imgsz 1280, batch 8, patience 30 | "Image size 1280 because the median DUT drone box is ~32 px and would shrink to ~11 px at 640; early stopping at epoch 106 (best 76) after 6.0 h." |
| 7 | Accuracy | table from `experiments/detector_results.md` | "In-domain DUT: P 0.987, R 0.967, mAP50 0.979; birds are the weakest class (R 0.59); 5 false drones in 1,534 bird/aircraft images at conf 0.25." |
| 8 | Cross-dataset: Det-Fly (**PRELIMINARY**) | a few missed Det-Fly frames (generate locally; dataset images are not committed) | "On 1,000 Det-Fly images recall falls to 0.285: drones seen from above against fields and hillsides, while DUT is mostly sky; size is not the cause." |
| 9 | Latency (**DESKTOP GPU**, RTX 3080) | latency table | "5.4 ms PyTorch FP16, 8.0 ms ONNX Runtime FP16 on CUDA; ONNX ran on the GPU only because torch loads the CUDA libraries first; no embedded numbers." |
| 10 | Real box noise → simulator | `figures/detector_noise.png` | "Measured on held-out DUT drones: ~0.9 px centre error and ~5-9% width error for small boxes; below 16 px there is no data, so long range uses scaled default curves." |

## Part 3: Geometry (0:14 – 0:20)

| # | slide | figure / visual | speaker note |
|---|---|---|---|
| 11 | Pinhole model and bearing | equations u = fx·X/Z + cx, v = fy·Y/Z + cy | "Divide by depth; 0.5° of attitude error is 22 px at fx = 2500, so the GPS/INS, not the detector, limits the bearing." |
| 12 | Range from size, and the size prior | w = fx·S/Z; table from STUDY.md §4.1 | "A 20% size error is a 20% range error; consumer drones span 0.21-0.37 m, but 0.7-0.8 m professional drones sit 2-3σ out in the prior's tail." |
| 13 | Observability: the camera must accelerate | sketch of the scale ambiguity (near-slow vs far-fast drone) | "Against an unknown constant-velocity drone, any constant-velocity camera, straight or sideways, can be explained by a scaled drone; only camera acceleration (weaving) breaks it." |

## Part 4: The EKF (0:20 – 0:28)

| # | slide | figure / visual | speaker note |
|---|---|---|---|
| 14 | State: p, v, S, GPS drift, attitude drift | the state vector | "S is a random constant, not noise: its error is the same every frame, and treating it as white noise would average it away and make the filter overconfident." |
| 15 | Models and Jacobians | measurement [u, v, log w]; H columns | "Constant velocity with white acceleration; log width makes the noise additive; every analytic Jacobian column is tested against finite differences." |
| 16 | Initialisation and the iterated update | P0 ellipsoid along the line of sight | "The first box gives direction ± 0.5° but range ± 43%; one linearisation that far off is overconfident, so the update iterates 3 Gauss-Newton steps." |
| 17 | How we judge honesty: NEES, NIS, coverage | χ² bands: [0.81, 1.21] (1 dof), [2.67, 3.35] (3 dof) for 200 runs | "RMSE says how wrong; NEES says whether the filter knows how wrong; coverage should be 95% at 2σ." |

## Part 5: Results, SIMULATION ONLY (0:28 – 0:35)

| # | slide | figure / visual | speaker note |
|---|---|---|---|
| 18 | Experiment design | grid: 3 camera motions × 5 size variants, 200 runs each, seeded | "Drone approaches from 200 m to 10 m and recedes; camera straight, sideways or weaving; fully reproducible from a fresh clone, byte for byte." |
| 19 | Claimed vs actual error | `figures/range_error_envelopes.png` | "Weaving: the filter's ±2σ band and the real errors agree; straight and lateral: real errors are far outside the band." |
| 20 | Actual vs reported σ | `figures/uncertainty_vs_motion.png` | "Straight-on: actual error ~32% (68th percentile), filter claims ~7%; weaving: 6% vs 4%, roughly honest." |
| 21 | Accuracy vs range | `figures/rmse_vs_range.png` | "Weaving with the 0.35 ± 0.15 prior: ~6% RMS out to 80 m, 13% at 80-120 m, 67% at 160-200 m where the drone is 4-5 px wide." |
| 22 | Consistency over time | `figures/nees_time.png` (+ `consistency_model_truth.png` as backup) | "Only weaving with a tight prior is consistent (coverage 95%, ANEES 1.07); even with model-matched truth, long range drifts overconfident (ANEES 10 at 150 m)." |

## Part 6: Limitations and what failed (0:35 – 0:38)

| # | slide | figure / visual | speaker note |
|---|---|---|---|
| 23 | What failed (shown, not hidden) | bullet list from README "What failed" | "Overconfident without manoeuvre (33% coverage); a wrong prior stays 27% short; 321 of 9,600 runs crash to NaN when an update step puts the drone behind the camera, and the averages exclude them." |
| 24 | Assumptions | README Limitations list | "Simulated pose with no time-sync errors, single target with no clutter, Gaussian detector noise, sim-only range, partial Det-Fly, no hardware." |

## Part 7: Next steps (0:38 – 0:40)

| # | slide | figure / visual | speaker note |
|---|---|---|---|
| 25 | Next steps | numbered list | "Inverse-depth or observability-constrained filter; a step-size guard for Z > 0; an observability flag when parallax is small; real flights with RTK ground truth; ground-background training data." |

## Q&A (0:40 – 0:45)

Backup slides worth having ready:

* `figures/consistency_model_truth.png` (model-truth NEES by start range).
* `experiments/nan_runs.md` (the NaN table).
* The straight-vs-lateral explanation (STUDY.md §3): same pixels, but the filter's world-frame
  velocity prior differs.
* The size-prior table (STUDY.md §4.1).

The 25 likely questions with short answers are in STUDY.md §10.

---------------------------------------------------------------------------------------------

## Things not to say

* Don't call the range accuracy "validated"; it is simulation-only.
* Don't present the NEES unit test as proof of consistency. It is lenient: it passes at 4.71,
  while a consistent filter would give 2.41-3.65.
* Don't quote the +22% or "ANEES ~25 → ~5" numbers as results. They are development
  observations that cannot be reproduced from the repo (STUDY.md §8).
* Don't quote ONNX latency as embedded performance; it is desktop GPU, and ONNX Runtime was
  slower than PyTorch here.
