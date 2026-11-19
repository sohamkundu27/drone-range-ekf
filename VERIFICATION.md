# VERIFICATION.md: are the results real and reproducible?

Verification run on 2026-11-19 against commit `6bb122a` (HEAD = origin/main). Nothing in the
experiment code, parameters or results was changed for this report. Scratch scripts used for
the checks lived outside the repo and are described below so the checks can be repeated.

## Summary

| # | check | result | note |
|---|---|---|---|
| 1 | Git state: clean, HEAD = origin/main, nothing forbidden tracked | PASS | 66 tracked files; no data/, runs/, weights, .venv or credentials |
| 2 | Fresh clone: install + `pytest -q` | PASS | 12 passed, 0 failed (Python 3.12.3, numpy 2.5.4, torch 2.14.1+cu130) |
| 3 | Fresh clone: `run_experiments.py` + `make_figures.py` reproduce the committed results | PASS | `results_sim.md`, `results_sim.json`, the 5 simulation figures and `nan_runs.{md,json}` byte-identical; 0 differences (`detector_noise.png` is not regenerated, needs data) |
| 4a | README.md numbers vs `experiments/` | PASS | detector, latency, simulation table, failure list, NaN counts all match (rounding only) |
| 4b | STUDY.md numbers vs `experiments/` | PARTIAL | 1 small mismatch (-36% quoted, -37.3% actual); 1 number has no committed source |
| 4c | Detector metrics re-run from the local weights and data | PASS | DUT and Det-Fly P/R/mAP identical to the committed JSON (only timing fields differ) |
| 5 | Figures exist, are non-empty and match `results_sim.md` | PASS | 6 PNGs; values read off and recomputed with the figure code match the tables |
| 6a | Camera project/unproject round trip | PASS | test passes; extra check with 2,000 arbitrary rotations: max error 3.4e-13 m |
| 6b | Jacobian test | PASS | passes; and FAILS (as it should) when one column of H is sign-flipped in memory |
| 6c | Noise-free run converges | PASS | weaving: 36% initial error, max 0.68% after 20 s, S 0.548 (true 0.55) |
| 6d | Unit tests test what their names claim | PARTIAL | 11 of 12 do; the NEES test is a loose regression guard, not a consistency test (documented); one docstring number is off |

Overall: the results are real and reproducible. A fresh clone regenerates every simulation result and figure byte for byte, the detector metrics re-run identically from the local weights, and the documents quote the result files correctly. The only number found wrong is a minor one (-36% vs -37.3%) in a test docstring and STUDY.md. The known weaknesses (overconfidence without camera manoeuvre, 321 NaN runs hidden by `nanmean`, a lenient NEES test) are real and already documented; this pass confirms them rather than finding new ones.

## Details for anything not PASS

### 4b. STUDY.md mismatches

| file:line | quoted | actual | source |
|---|---|---|---|
| STUDY.md:392 | "the range error stays at **-36%** forever" (straight camera, perfect data) | **-37.3%** (constant from t = 10 s to the end; final S 0.345) | rerun of `tests/test_ekf.py::run_noise_free("straight", 0.55)` |
| tests/test_ekf.py:97 (docstring) | "range error stays at the size-prior error (0.35 / 0.55 - 1 = -36%)" | -37.3%; the -36.4% is the formula, the filter ends slightly below it because S drifts from 0.35 to 0.345 | same |

The test itself only asserts |error| > 30% after 10 s and |S - 0.35| < 0.02, so it passes;
only the quoted number is imprecise. Not edited here (this pass records, it does not fix).

Numbers in STUDY.md that **cannot be checked against any committed file** (not mismatches,
but not verifiable from the repo either):

| file:line | quoted | why it cannot be checked |
|---|---|---|
| STUDY.md:156 | straight-vs-lateral run-by-run difference "drops from ~9% to ~2%" with a flat velocity prior | came from a one-off diagnostic run (σ_v0 = 1000); no experiment config or result file holds it |

STUDY.md:441 (size-bank filter "median error 95% straight-on") has no committed output file, but `python experiments/try_size_bank.py straight 0.02` reruns it and gives median |rel err| 0.946, RMS 1.553, 2σ coverage 10.5%, so the quote is confirmed. (Its log also shows the `log(Z)` NaN warning of the known Z < 0 crash.)

The items already labelled **DEVELOPMENT OBSERVATION** in STUDY.md §8 (+22% bias, plain-EKF
ANEES ~25 → ~5, σ_v0 = 30 trial) are not reproducible by design and are labelled as such.

Everything else checked in STUDY.md matched, including: detector and latency numbers
(lines 54-66), training-set balance (5,200 drone images = 28% of the raw union with 13,389
negatives; 51% after keeping 5,000 negatives; bird median size 28% → 10%), detector noise
(lines 84-88), trajectory distances (approach 200 → 10 m at t = 28 s → 86 m; crossing starts
135.6 m, spans 64.1-253.4 m; recomputed from `sim/trajectories.py`), χ² bands ([0.81, 1.21],
[2.67, 3.35], [2.41, 3.65], recomputed with scipy), experiment C/D/E/F numbers (lines
408-461), the NaN counts (lines 305, 451-461 vs `experiments/nan_runs.md`), the stereo
arithmetic (line 478-483: ±1.7 / 6.7 / 27 m, 3.75 px disparity, 0.44 px → 12% bias), the
size-prior statistics (consumer mean 0.303 m, sd 0.064; professional 2.33σ and 3.07σ), and
the NEES test values (4.71 iterated, 89.87 plain, both recomputed).

### 6d. Do the unit tests test what their names claim?

Read all 12 tests (`tests/test_camera.py`, `tests/test_ekf.py`):

| test | verdict |
|---|---|
| `test_rotation_is_orthonormal` | yes: RᵀR = I and det R = 1 for 20 `look_at` poses |
| `test_look_at_target_projects_to_principal_point` | yes |
| `test_camera_axes_convention` | yes: world up → image up (v < cy), east → right when looking north |
| `test_project_unproject_round_trip` | yes: 100 random poses, error < 1e-8 m, and bearing = unit vector to the point. Only `look_at` poses; the extra 2,000-rotation check above covers arbitrary R |
| `test_box_width_inverse_depth` | yes: w = fx·S/Z and doubling Z halves w |
| `test_visibility` | yes: in front / behind / outside the image |
| `test_small_rotation_matches_rodrigues` | partly: checks orthonormality and the **first-order** small-angle form I + [θ]×, not the full Rodrigues formula; it also contains an unrelated `world_to_camera` identity check |
| `test_measurement_jacobian_matches_numerical` | yes: all 13 columns of H vs central differences at 50 random states, with and without the width row; attitude-drift columns allowed 3% (small-angle form). Mutation check: flipping the sign of the S column makes it fail |
| `test_attitude_jacobian_exact_at_zero_drift` | yes |
| `test_converges_on_noise_free_data` | yes: perfect detections and poses, weaving camera, true S 0.55 vs prior 0.35. Asserts initial error > 30%, < 2% after 20 s (measured 0.68%), S within 0.01 (0.548), velocity within 0.2 m/s (0.022). Mutation check: with the update disabled it fails |
| `test_range_not_observable_with_constant_velocity_camera` | yes, it shows range does not converge with perfect data (error ≥ 30% after 10 s, S stays at the prior). Docstring number off (-36% vs -37.3%, see 4b) |
| `test_nees_consistent_when_model_matches` | **the name over-claims.** It passes for 1.5 < ANEES < 6.0; measured 4.71, while a consistent filter would give 2.41-3.65 for 60 runs. It is a regression guard (the plain EKF gives 89.87 and fails), not a proof of consistency. This is stated in its docstring and in the README. 15 of the 60 runs have NaN NEES, but only in the first 0.0-0.8 s before the first detection (the filter has not started); none in the scored window t ≥ 15 s, so the docstring's "no NaN runs" is correct |

## Could not verify

* **Detector training itself** (epochs, 6.0 h, best epoch 76): checked only against the local,
  untracked `runs/detect/drone4_bal/results.csv` (last epoch 106, best fitness at epoch 76,
  6.02 h). A fresh clone has no training log or weights, so a reviewer cannot check these.
* **Detector metrics in a fresh clone**: they need the ~13 GB of datasets and the trained
  weights, which are git-ignored. Re-run here from the local copies (check 4c), not from the clone.
* **`figures/detector_noise.png` and `experiments/detector_noise.json`**: need data and weights;
  not regenerated. The figure's plotted values match the committed JSON/markdown table.
* **Latency** (`experiments/latency.json`): hardware-dependent; not re-timed. The README and
  STUDY.md values match the committed JSON. Re-running `eval.py` showed similar per-image
  inference time (4.86 ms vs 4.86 ms committed), but that is not the latency benchmark.
* **Size-prior spec-sheet table** (STUDY.md §4.1): the 8 manufacturer pages were read during
  the previous verification pass; they were not re-fetched for this report.
* **Det-Fly "missed drones are not smaller than detected ones"**: came from a 200-image
  scratch analysis that is not committed. The "~46 px at 1280 input" part is checked
  (`figures/dataset_stats.md`: median width 138 px on 3840-px frames = 46 px at 1280).
* **The straight-vs-lateral "~9% → ~2%" diagnostic** (STUDY.md:156): see 4b.
* **Real-world range accuracy**: everything on range is simulation-only by design; there is no
  real flight data with ground truth in this project.

## How the checks were done

* Step 1: `git fetch`, `git rev-parse HEAD origin/main`, `git status`, `git ls-files` grepped for
  data/runs/.venv paths, weight extensions (.pt .pth .onnx .engine .ckpt .safetensors) and
  credential names.
* Step 2-3 result: the clone was at `6bb122a`. `pytest -q`: 12 passed in 5.0 s. `run_experiments.py`: 48 configurations × 200 runs, 2 min 11 s wall clock. `make_figures.py` and `check_nan_runs.py` then ran without error, and `git status` in the clone was empty, i.e. every regenerated tracked file (`results_sim.md`, `results_sim.json`, `nan_runs.md`, `nan_runs.json`, 5 simulation figures) is byte-identical to the commit. There is no Monte Carlo variation to judge: all randomness is seeded (`seed = 1000 + run index`). `detector_noise.png` is not regenerated by these scripts (needs data and weights).
* Step 2-3 method: `git clone https://github.com/sohamkundu27/drone-range-ekf.git` into an empty temp
  directory, new venv, the README install steps (torch cu130, `requirements.txt`, pytest), then
  `pytest -q`, `run_experiments.py`, `make_figures.py`, and a byte and value comparison against
  the committed files.
* Step 4: a script extracted every number from README.md and STUDY.md and flagged those not
  present in any `experiments/*.json|md` or `figures/dataset_stats.*` file at the quoted
  precision; flagged numbers were then traced by hand (constants, arithmetic, recomputation).
  Because that scan is loose (a wrong number can match an unrelated value), every row of the
  README simulation table, detector table and latency table, and the result statements in
  STUDY.md §2, §5, §7, §8 and §10, were also compared to their specific source row by hand.
* Step 5: each PNG opened and read; values compared to `results_sim.md`; the figure code's own
  binning function rerun on the table's range bins reproduces the table exactly.
* Step 6: tests read line by line; values behind them recomputed; two in-memory mutations
  (no file edits) to confirm the Jacobian and convergence tests can fail.
