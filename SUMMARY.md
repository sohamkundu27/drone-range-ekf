> **Snapshot from the end of the project build (2026-11-19), kept verbatim below.** A later
> verification pass found and documented more: 321 of 9,600 simulated runs crash to NaN and
> are excluded from the averages (README "What failed" #8, `experiments/nan_runs.md`); the
> ONNX latency rows did run on the CUDA provider (README, Latency); the NEES test is lenient
> (4.71 vs a consistent 2.41-3.65); and the size prior is now checked against manufacturer
> specs (STUDY.md §4.1). The README and STUDY.md are the current reference.

Everything is finished: committed, pushed to `main`, and `git status` is clean. No weights, ONNX files, `data/`, `runs/`, `.venv` or credentials are tracked. Detector training stopped early at epoch 106 after 6.0 hours; the best epoch was 76. All 12 tests pass.

## Results

**Detector (YOLO26s, imgsz 1280)**
- **DUT test (in-domain, 2,245 drones):** precision 0.987, recall 0.967, mAP50 0.979, mAP50-95 0.749.
- **Other classes:** bird mAP50 0.664 (recall 0.59, the weakest), airplane 0.856, helicopter 0.944.
- **Det-Fly, preliminary (1,000 images, one sequence):** precision 0.842, but recall 0.285 and mAP50 0.304. Box size is not the cause. The missed frames show the drone from above against farmland, roads or a sports field, or against a dark hillside, while DUT drones are mostly against sky.
- **False drones on 1,534 bird/airplane/helicopter images:** 5 at confidence ≥ 0.25 and 1 at ≥ 0.5.
- **Latency (desktop GPU, RTX 3080):** PyTorch FP16 5.4 ms (187 FPS), FP32 7.0 ms; ONNX Runtime FP16 8.0 ms, FP32 11.1 ms.
- **Real box errors, used by the simulator:** centre error about 0.9 px for small boxes; width error about 9% at 16–24 px and about 5% at 32–96 px. Real errors have heavier tails than a Gaussian.

**Range EKF (simulation only, 200 runs per configuration)**

| Camera | Size prior | RMS range error | 2σ coverage (ideal 95%) |
|---|---|---|---|
| Weaving | 0.35 ± 0.15 m | 12.5% (about 6% out to 80 m) | 78% |
| Weaving | tight, 0.35 ± 0.03 m | 4.0% | 95% (consistency score 1.07, ideal 1) |
| Straight or lateral | 0.35 ± 0.15 m | 42–45% | 33–36% |

## What failed (it's all in the README and STUDY.md)
1. **The filter is overconfident when the camera doesn't manoeuvre.** Straight or lateral at constant velocity, it reports about ±7% while the real error is around 42%. Range is genuinely not observable there; the cause is the box-width noise in the linearisation. Not fixed.
2. **Wrong size prior (true drone 0.60 m, prior 0.35 m):** straight-on, range stays 27% too short while the filter claims ±6% (3% coverage). The weaving camera learns the size (0.58 m) but coverage is only 40%.
3. **Long range is inconsistent** even when the truth follows the filter's own model: consistency score 10.0 at 150 m, ideal 3.
4. **Other failures:**
   - The plain (non-iterated) EKF is far worse.
   - Near-perfect pose sensors break the straight-on case.
   - Inflating the assumed width noise only partly helps.
   - A bank of filters over sizes made things worse.
5. **Detector:** the low Det-Fly recall, as above.
6. **Two script bugs found and fixed this session:**
   - Out-of-memory crash: prediction now runs one image at a time.
   - DUT test images have mixed resolutions: box errors are now converted to the simulator's 1920-wide frame.

## What you should double-check before presenting
- **Long-range noise is extrapolated.** DUT has almost no drones under 16 px, so the 4–16 px range (about 55–200 m) uses default noise curves scaled to the smallest measured bin.
- **The false-positive rate is probably optimistic:** those images are ordinary photos, not tiny birds against sky.
- **Two STUDY.md §8 numbers can't be reproduced from the repo:** the "+22% bias" and the "consistency score ~25 → ~5" came from development code that isn't kept. They are labelled as such.
- **The NEES test is lenient.** It passes at about 4.7 against an ideal 3, with a 2× tolerance. Expect questions on that.
- **ONNX Runtime being slower than PyTorch** may be a setup issue: the log had a "No registered plugin EP device" warning, although it did run on the GPU.
- **The 0.35 ± 0.15 m size prior has no cited source table.** Have your sources ready.
- **The detector's precision/recall** are Ultralytics' values at the best-F1 confidence threshold for each class.
