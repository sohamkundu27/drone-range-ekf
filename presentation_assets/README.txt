presentation_assets/  -  everything for the slides in one folder (scp-able, about 4 MB)

Detector: YOLO26s, 4 classes (drone, bird, airplane, helicopter), imgsz 1280,
run runs/detect/drone4_bal (stopped early at epoch 106, best epoch 76).
Range EKF: hand-written NumPy filter, evaluated in SIMULATION ONLY.

WHICH NUMBERS ARE WHICH (read this before putting a number on a slide)

  VALIDATION split  = data/yolo/drone4_bal.yaml "val": COCO val + DUT Anti-UAV val +
                      Open Images val, 4,316 images (2,620 drones, 1,494 birds,
                      1,116 airplanes, 134 helicopters). Used during training to pick
                      the best epoch. Files: results.png, PR_curve.png,
                      confusion_matrix.png, confusion_matrix_conf025*.
  TEST split        = held-out, never used in training or model selection:
                      DUT Anti-UAV test (2,245 drones), COCO + Open Images test, and
                      Det-Fly (PRELIMINARY: 1,000 images of one sequence, a different
                      dataset). Files: detector_results.md, detector_noise.png,
                      dut_good_*.jpg, detfly_missed_*.jpg.
  The headline detector numbers in README.md / STUDY.md / the slides (e.g. DUT drone
  mAP50 0.979, recall 0.967) are TEST numbers. Validation plots will not show the same
  values. Never quote mAP from the conf-0.25 confusion-matrix run (see below).
  Simulation files have no val/test split: they are 200 Monte Carlo runs per
  configuration with fixed seeds (SIMULATION ONLY).
  Latency is DESKTOP GPU (RTX 3080), not embedded hardware.


1. DETECTOR: TRAINING / VALIDATION PLOTS

results.png
    Copied unchanged from runs/detect/drone4_bal/ (made by Ultralytics). Training curves
    per epoch: train/val box, class and L1 losses (YOLO26 has no DFL loss), and VALIDATION
    precision, recall, mAP50 and mAP50-95. The validation losses and mAP flatten after
    about epoch 60-75 while the training losses keep falling; training stopped early at
    epoch 106 (patience 30) and kept the best epoch, 76.

PR_curve.png
    Copied from runs/detect/drone4_bal/BoxPR_curve.png. Precision-recall curve per class
    for the best weights on the VALIDATION split.

confusion_matrix.png
    Copied unchanged from runs/detect/drone4_bal/. Ultralytics' end-of-training confusion
    matrix (raw counts) on the VALIDATION split. CAUTION: it was made at the validator's
    default confidence threshold of 0.001 (args.yaml has conf: null), so it counts very
    low-confidence boxes and overstates false positives. Prefer the conf-0.25 version.

confusion_matrix_conf025.png
    NEW. Confusion matrix on the same VALIDATION split, best.pt, conf 0.25, NMS IoU 0.5,
    prediction-truth match at IoU 0.5. A separate validation call made only for this plot
    (detector/confusion_matrix_conf.py); do not quote mAP from it. Normalised by TRUE
    class: each column sums to 1. x-axis = TRUE class, y-axis = PREDICTED class. The
    "background" column (true background = no object there) has no total to divide by,
    so it shows raw counts, marked "(count)".

confusion_matrix_conf025_counts.png
    NEW. The same matrix as raw counts (log colour scale). Same axes.

confusion_matrix_conf025.json
    NEW. The raw matrix behind both plots, indexed matrix[predicted][true], plus settings.
    Key numbers (VALIDATION, conf 0.25):
      true drones 2,620: 2,477 detected as drone, 139 missed (predicted background),
        4 given another class (1 bird, 1 airplane, 2 helicopter)
      false drone boxes on background (no object): 90
      true birds, airplanes or helicopters predicted as drone: 0
      other confusions: 28 airplanes called helicopter, 2 helicopters called airplane,
        2 airplanes called bird, 1 bird called airplane
      missed (predicted background): 473 birds, 190 airplanes, 18 helicopters


2. DETECTOR: TEST RESULTS AND EXAMPLES

detector_results.md
    Copy of experiments/detector_results.md. TEST-split accuracy (DUT test, COCO + Open
    Images test, Det-Fly PRELIMINARY), false drones on 1,534 held-out bird/aircraft TEST
    images, DESKTOP-GPU latency (RTX 3080), and box-error statistics on DUT TEST drones.

detector_noise.png
    Copy of figures/detector_noise.png. Box-centre error, box-width error and detection
    rate vs box width, measured on matched true positives in the DUT TEST set (pixels
    rescaled to 1920-wide frames), with the simulator's default curves. These measured
    curves drive the simulated detections.

dut_good_small.jpg, dut_good_medium.jpg, dut_good_large.jpg
    Drawn by detector/presentation_examples.py. DUT Anti-UAV TEST images 02151.jpg
    (drone 28 px wide, conf 0.74, IoU 0.89), 01727.jpg (56 px, conf 0.84, IoU 0.91) and
    01049.jpg (289 px, conf 0.90, IoU 0.95). Green = ground truth, red = prediction;
    small drones get a zoom inset. Chosen from a seeded sample of 300 test images: frames
    with one drone, conf >= 0.7 and IoU >= 0.7 (256 qualified), at the 10th, 50th and
    90th percentile of drone width. They show GOOD detections, not typical failures.

detfly_missed_1.jpg, detfly_missed_2.jpg
    Drawn by detector/presentation_examples.py. Det-Fly (PRELIMINARY) frames
    010_0100136.jpg and 010_0100900.jpg where the drone was MISSED (no drone box at conf
    >= 0.25 overlaps the truth). Both are seen from above against ground (grass next to a
    running track; farmland), 62 px and 29 px wide at the 1280 detector input, so size is
    not the cause. In the first, the detector put a false drone box (conf 0.35, red) on two
    people on the track. 200 of the 300 sampled Det-Fly frames were misses (recall 0.285).

dataset_stats.md
    Copy of figures/dataset_stats.md. Per dataset and split (train / val / test): image
    and object counts per class, median box size in pixels, share of boxes under 16 and
    32 px, median size after resizing to 640 and 1280, typical resolution. No metrics.


3. RANGE EKF: SIMULATION ONLY (200 seeded Monte Carlo runs per configuration)

range_error_envelopes.png
    Range error over time (drone approaches 200 m -> 10 m -> 86 m) for straight, lateral
    and weaving camera motion, with the filter's claimed +-2 sigma band and the actual
    central 95% of errors. Top: size prior 0.35 +- 0.15 m. Bottom: wrong prior (true 0.60 m).

rmse_vs_range.png
    RMS range error vs true range, per camera motion and size-information variant.

uncertainty_vs_motion.png
    Actual error (68th percentile of |error|, about 1 sigma) vs the sigma the filter
    reports. Ticks far below the bar = overconfident (straight and lateral cases).

nees_time.png
    Range NEES over time (ideal 1, grey band = 95% bounds if consistent). Only the weaving
    camera with a tight prior is consistent.

consistency_model_truth.png
    Position NEES (ideal 3) when the truth is drawn from the filter's own model, for start
    ranges 40 / 80 / 150 m: overconfidence grows with range (ANEES 3.9 / 4.8 / 10.0 weaving).

results_sim.md
    Copy of experiments/results_sim.md: every simulation configuration (main grid A, other
    paths B, plain EKF C, stress tests D, model-truth E, width x2 F) with RMS error, bias,
    reported sigma, 2-sigma coverage, ANEES, NIS, divergence; plus RMS error by range bin.
    Metrics are averaged over time steps that did not crash to NaN (see nan_runs.md).

nan_runs.md
    Copy of experiments/nan_runs.md: 321 of 9,600 runs whose filter state became NaN (an
    update put the drone behind the camera), per configuration, and how the "final outside
    3 sigma" numbers change if those runs count as diverged.


4. DOCUMENTS (copies of the repo-root files)

README.md            project overview, results tables, what failed, how to reproduce
STUDY.md             full explanation of every component and the math, problems found,
                     25 likely questions with answers
VERIFICATION.md      reproducibility check: fresh clone reproduces all simulation results
                     byte for byte; doc numbers vs result files; figure and test review
CODE_TOUR.md         function-by-function walkthrough of the camera, noise, EKF and
                     Monte Carlo code
PRESENTATION_NOTES.md  suggested 45-minute talk: slide list, figure per slide, speaker notes

    These are COPIES as of this commit; the files in the repo root are the reference.
    Their image links (figures/...) point at the repo layout, so pictures will not show
    inside this folder; the same PNGs are here by name.


SOURCES AND LICENSES (see DATA.md)

DUT Anti-UAV: Zhao et al., "Vision-based Anti-UAV Detection and Tracking", IEEE T-ITS 2022
    (arXiv:2205.10851). Repository license Apache-2.0; no separate data license stated.
Det-Fly: Zheng et al., "Air-to-Air Visual Detection of Micro-UAVs: An Experimental
    Evaluation of Deep Learning", RA-L 2021. Repository license MIT.
COCO / Open Images (validation plots only; no photos from them are included here).
Cite DUT Anti-UAV and Det-Fly on any slide that shows their images.
