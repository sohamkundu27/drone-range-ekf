# Constant-speed vs accelerating cameras (SIMULATION ONLY)

Main grid A (approach_recede drone, 200 runs, seeds 1000..1199). Old = experiments/results_sim.md (constant-speed `straight` 4 m/s along the line of sight, `lateral` 4 m/s sideways); new = experiments/accel_grid/results_sim.md (`surge_15` / `lat_surge_15`: 2 t + 15 sin(2 pi t / 12) along the same axis, peak acceleration 4.1 m/s^2, same as the weave). `weaving` is in both and was rerun.

Columns: RMS rel err over t >= 10 s; 2-sigma coverage over t >= 2 s (ideal 95%); ANEES range (ideal 1.0); NaN runs = runs whose state became NaN after initialisation (of 200); reported sigma = median sigma/r the filter reports, t >= 10 s; actual 68th pct = 68th percentile of |range error| / r over runs and t >= 10 s (about 1 sigma; an honest filter has reported sigma close to it); S final = mean final size estimate (mean true size).

## size_prior

| camera | RMS rel err | 2-sigma coverage | ANEES range | NaN runs | reported sigma/r | actual 68th pct | S final (true) |
|---|---|---|---|---|---|---|---|
| lateral | 44.9% | 35.9% | 41.61 | 4 | 6.8% | 32.3% | 0.381 (0.371) |
| lateral, accelerating | 12.6% | 78.9% | 4.04 | 14 | 3.7% | 5.8% | 0.387 (0.371) |
| straight | 42.5% | 33.3% | 23.43 | 7 | 6.7% | 32.0% | 0.382 (0.371) |
| straight, accelerating | 27.6% | 50.6% | 13.63 | 7 | 5.3% | 17.6% | 0.372 (0.371) |
| weaving | 12.5% | 77.7% | 3.92 | 14 | 3.7% | 5.9% | 0.387 (0.371) |
| weaving (rerun) | 12.5% | 77.7% | 3.92 | 14 | 3.7% | 5.9% | 0.387 (0.371) |

## tight_prior

| camera | RMS rel err | 2-sigma coverage | ANEES range | NaN runs | reported sigma/r | actual 68th pct | S final (true) |
|---|---|---|---|---|---|---|---|
| lateral | 11.3% | 71.7% | 3.49 | 0 | 6.0% | 11.6% | 0.375 (0.354) |
| lateral, accelerating | 4.0% | 95.1% | 1.06 | 0 | 3.7% | 3.7% | 0.360 (0.354) |
| straight | 11.3% | 71.0% | 3.53 | 0 | 6.0% | 11.6% | 0.377 (0.354) |
| straight, accelerating | 8.7% | 81.4% | 2.48 | 0 | 5.3% | 7.2% | 0.360 (0.354) |
| weaving | 4.0% | 94.9% | 1.07 | 0 | 3.7% | 3.7% | 0.360 (0.354) |
| weaving (rerun) | 4.0% | 94.9% | 1.07 | 0 | 3.7% | 3.7% | 0.360 (0.354) |

## wrong_prior

| camera | RMS rel err | 2-sigma coverage | ANEES range | NaN runs | reported sigma/r | actual 68th pct | S final (true) |
|---|---|---|---|---|---|---|---|
| lateral | 29.5% | 2.6% | 59.93 | 0 | 5.9% | 31.8% | 0.477 (0.600) |
| lateral, accelerating | 9.3% | 41.0% | 11.22 | 0 | 3.3% | 8.9% | 0.579 (0.600) |
| straight | 28.7% | 3.2% | 47.74 | 0 | 5.9% | 31.8% | 0.481 (0.600) |
| straight, accelerating | 21.6% | 10.3% | 33.97 | 0 | 4.6% | 22.9% | 0.551 (0.600) |
| weaving | 9.7% | 40.3% | 13.00 | 0 | 3.3% | 9.0% | 0.578 (0.600) |
| weaving (rerun) | 9.7% | 40.3% | 13.00 | 0 | 3.3% | 9.0% | 0.578 (0.600) |

## bearing_only

| camera | RMS rel err | 2-sigma coverage | ANEES range | NaN runs | reported sigma/r | actual 68th pct | S final (true) |
|---|---|---|---|---|---|---|---|
| lateral | 161.2% | 17.8% | 29.83 | 0 | 9.2% | 133.3% | 0.454 (0.371) |
| lateral, accelerating | 124.1% | 90.4% | 35.61 | 0 | 6.4% | 8.7% | 0.389 (0.371) |
| straight | 134.8% | 21.5% | 26.76 | 0 | 9.3% | 118.2% | 0.439 (0.371) |
| straight, accelerating | 81.3% | 45.9% | 11.15 | 0 | 10.3% | 69.9% | 0.408 (0.371) |
| weaving | 321.4% | 89.8% | 51.95 | 0 | 6.2% | 8.8% | 0.393 (0.371) |
| weaving (rerun) | 321.4% | 89.8% | 51.95 | 0 | 6.2% | 8.8% | 0.393 (0.371) |

## Perfect data

tests/test_ekf.py `run_noise_free` (true size 0.55 m, prior 0.35 m, size_prior, no pose noise, perfect detector). Time after which |range error| stays below 2% until the end of the 40 s run (see figures_accel/perfect_data_error_vs_time.png).

| camera | final range error | S final (true 0.55) | under 2% from |
|---|---|---|---|
| lateral | -42.1% | 0.319 m | never |
| lateral, accelerating | -0.4% | 0.548 m | t = 3.8 s |
| straight | -37.3% | 0.345 m | never |
| straight, accelerating | -0.7% | 0.546 m | t = 26.9 s |
| weaving | -0.3% | 0.548 m | t = 3.9 s |

## Weaving reproduction check

All 18 weaving configurations (A, B, C, D, E, F) were rerun: 18 of 18 reproduce every metric in experiments/results_sim.json exactly, and 18 of 18 rows (main table and range-bin table) are identical in experiments/results_sim.md.

Grid runtime: 83 s. Experiments B-F (other drone paths, plain EKF, pose stress, model-truth NEES, width x2) are in experiments/accel_grid/results_sim.md with the same row names as experiments/results_sim.md, camera swapped.
