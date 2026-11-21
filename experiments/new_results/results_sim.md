# Simulation results (200 Monte Carlo runs each; SIMULATION ONLY)

Detector noise model: `detector_noise.json`.

rel err = (estimated range - true range) / true range, over t >= 10 s. coverage = fraction of time steps (t >= 2 s) with |range error| <= 2 sigma (ideal 95%). ANEES range: mean of the 1-dof range NEES (ideal 1.0). diverged = final error outside +/-3 sigma.

| config | RMS rel err | bias | reported sigma/r | 2-sigma coverage | ANEES range | ANEES pos (3) | mean NIS | final outside 3sigma | final err > 50% | S final (true) |
|---|---|---|---|---|---|---|---|---|---|---|
| A_approach_straight_bearing_only | 133.9% | +104.8% | 9.3% | 21.6% | 26.69 | 27.37 | 2.07 | 99.5% | 99.5% | 0.439 (0.371) |
| A_approach_straight_size_prior | 42.7% | +1.8% | 6.7% | 33.4% | 23.71 | 19.27 | 2.85 | 58.0% | 15.0% | 0.381 (0.371) |
| A_approach_straight_tight_prior | 11.3% | -1.6% | 6.0% | 70.9% | 3.56 | 5.29 | 2.86 | 20.5% | 0.0% | 0.376 (0.354) |
| A_approach_straight_wrong_prior | 28.9% | -27.3% | 5.9% | 2.9% | 48.78 | 34.85 | 2.88 | 74.0% | 0.5% | 0.479 (0.600) |
| A_approach_straight_wrong_tight | 35.1% | -34.8% | 5.3% | 0.0% | 96.15 | 62.42 | 2.88 | 100.0% | 0.0% | 0.422 (0.600) |
| A_approach_lateral_bearing_only | 160.2% | +121.2% | 9.2% | 17.8% | 29.76 | 29.86 | 2.09 | 100.0% | 99.5% | 0.454 (0.371) |
| A_approach_lateral_size_prior | 44.6% | +3.9% | 6.7% | 35.8% | 23.37 | 19.03 | 2.86 | 58.5% | 16.0% | 0.381 (0.371) |
| A_approach_lateral_tight_prior | 11.3% | -2.0% | 6.0% | 71.5% | 3.54 | 5.26 | 2.86 | 21.5% | 0.0% | 0.374 (0.354) |
| A_approach_lateral_wrong_prior | 29.8% | -28.0% | 5.9% | 2.2% | 61.05 | 39.70 | 2.88 | 77.5% | 1.5% | 0.474 (0.600) |
| A_approach_lateral_wrong_tight | 35.3% | -35.0% | 5.3% | 0.0% | 96.79 | 62.58 | 2.89 | 100.0% | 0.0% | 0.421 (0.600) |
| A_approach_weaving_bearing_only | 297.1% | +28.0% | 6.3% | 89.9% | 35.75 | 47.24 | 3.59 | 2.0% | 0.5% | 0.393 (0.371) |
| A_approach_weaving_size_prior | 13.1% | -1.2% | 3.7% | 77.1% | 22.53 | 7.30 | 2.95 | 4.5% | 0.0% | 0.386 (0.371) |
| A_approach_weaving_tight_prior | 3.9% | +0.1% | 3.7% | 94.9% | 1.06 | 3.44 | 2.88 | 0.0% | 0.0% | 0.359 (0.354) |
| A_approach_weaving_wrong_prior | 9.8% | -7.9% | 3.2% | 38.7% | 13.29 | 11.40 | 3.04 | 11.5% | 0.0% | 0.577 (0.600) |
| A_approach_weaving_wrong_tight | 14.5% | -13.8% | 3.1% | 0.3% | 30.03 | 23.20 | 3.15 | 67.5% | 0.0% | 0.552 (0.600) |
| B_constant_velocity_straight_bearing_only | 190.4% | +23.5% | 7.9% | 69.7% | 35.02 | 43.86 | 2.79 | 5.0% | 2.0% | 0.401 (0.371) |
| B_constant_velocity_straight_size_prior | 19.0% | -6.0% | 6.9% | 54.8% | 10.32 | 12.50 | 2.85 | 40.5% | 2.0% | 0.338 (0.371) |
| B_constant_velocity_lateral_bearing_only | 98.5% | +88.3% | 8.8% | 18.4% | 23.76 | 26.72 | 1.94 | 100.0% | 100.0% | 0.422 (0.371) |
| B_constant_velocity_lateral_size_prior | 34.0% | +6.9% | 6.4% | 48.7% | 10.07 | 11.62 | 2.87 | 40.5% | 10.0% | 0.388 (0.371) |
| B_constant_velocity_weaving_bearing_only | 159.3% | +9.0% | 7.2% | 92.8% | 29.63 | 71.68 | 2.72 | 3.0% | 2.5% | 0.387 (0.371) |
| B_constant_velocity_weaving_size_prior | 10.6% | -7.9% | 4.8% | 61.3% | 6.49 | 8.98 | 2.92 | 37.0% | 0.5% | 0.352 (0.371) |
| B_gentle_turn_straight_bearing_only | 117.7% | +18.3% | 8.5% | 69.2% | 18.46 | 31.73 | 2.64 | 10.0% | 2.5% | 0.351 (0.371) |
| B_gentle_turn_straight_size_prior | 22.7% | -15.6% | 7.2% | 45.2% | 22.79 | 20.07 | 2.93 | 47.5% | 5.0% | 0.300 (0.371) |
| B_gentle_turn_lateral_bearing_only | 36.3% | -1.0% | 11.1% | 55.3% | 7.94 | 12.12 | 2.01 | 37.0% | 29.0% | 0.387 (0.371) |
| B_gentle_turn_lateral_size_prior | 29.8% | -20.7% | 7.5% | 30.6% | 25.05 | 27.40 | 2.94 | 75.5% | 9.0% | 0.299 (0.371) |
| B_gentle_turn_weaving_bearing_only | 136.0% | +5.1% | 6.7% | 78.9% | 183.12 | 111.98 | 3.01 | 5.0% | 2.0% | 0.353 (0.371) |
| B_gentle_turn_weaving_size_prior | 13.4% | -11.1% | 4.9% | 42.0% | 10.77 | 13.25 | 3.12 | 31.5% | 0.5% | 0.354 (0.371) |
| C_plainEKF_approach_straight_size_prior | 145.2% | +111.7% | 3.8% | 13.6% | 132.33 | 100.68 | 2.99 | 97.5% | 84.5% | 0.801 (0.371) |
| C_plainEKF_approach_lateral_size_prior | 146.4% | +101.3% | 3.7% | 14.5% | 127.07 | 95.50 | 3.15 | 97.5% | 80.0% | 0.735 (0.371) |
| C_plainEKF_approach_weaving_size_prior | 59.9% | +23.4% | 3.1% | 24.8% | 53.56 | 44.35 | 3.85 | 91.0% | 1.5% | 0.458 (0.371) |
| D_whitepose_straight | 74.4% | -67.5% | 20.2% | 15.1% | 507.38 | 471.08 | 2.61 | 21.0% | 21.5% | 0.118 (0.371) |
| D_drift_filter_assumes_white_straight | 84.8% | -79.8% | 18.8% | 11.7% | 1415.53 | 1096.35 | 0.52 | 90.0% | 91.0% | 0.050 (0.371) |
| D_precise_pose_straight | 78.1% | -74.8% | 9.3% | 5.8% | 2034.77 | 2133.80 | 2.28 | 93.0% | 91.0% | 0.092 (0.371) |
| D_whitepose_weaving | 15.8% | -2.5% | 4.9% | 77.8% | 16.93 | 21.91 | 3.03 | 5.5% | 0.5% | 0.369 (0.371) |
| D_drift_filter_assumes_white_weaving | 16.1% | -1.4% | 5.0% | 89.6% | 22.45 | 43.84 | 0.98 | 3.0% | 1.0% | 0.371 (0.371) |
| D_precise_pose_weaving | 13.4% | -5.5% | 3.5% | 65.0% | 14.93 | 133.89 | 2.61 | 17.5% | 0.0% | 0.371 (0.371) |
| E_model_truth_weaving_40m | 5.6% | -1.0% | 3.9% | 87.0% | 1.80 | 3.86 | 3.02 | 5.5% | 0.0% | 0.373 (0.371) |
| E_model_truth_lateral_40m | 39.0% | +27.1% | 6.6% | 33.6% | 13.59 | 15.01 | 3.08 | 42.5% | 10.0% | 0.440 (0.371) |
| E_model_truth_weaving_80m | 6.7% | -3.2% | 4.1% | 80.4% | 2.58 | 4.80 | 3.02 | 8.0% | 0.0% | 0.371 (0.371) |
| E_model_truth_lateral_80m | 23.5% | +4.1% | 6.8% | 53.7% | 7.66 | 9.99 | 3.00 | 31.5% | 3.5% | 0.371 (0.371) |
| E_model_truth_weaving_150m | 11.3% | -7.4% | 4.8% | 59.5% | 9.71 | 11.05 | 3.10 | 20.0% | 0.0% | 0.366 (0.371) |
| E_model_truth_lateral_150m | 30.9% | -23.5% | 8.2% | 34.3% | 28.13 | 30.44 | 2.94 | 55.0% | 10.5% | 0.291 (0.371) |
| F_widthx2_approach_straight_size_prior | 52.0% | +22.7% | 7.1% | 44.2% | 14.30 | 13.71 | 2.13 | 63.0% | 28.5% | 0.451 (0.371) |
| F_widthx2_approach_straight_wrong_prior | 19.0% | -16.0% | 6.1% | 27.3% | 15.34 | 13.78 | 2.15 | 17.0% | 0.0% | 0.555 (0.600) |
| F_widthx2_approach_lateral_size_prior | 57.0% | +27.1% | 7.0% | 43.1% | 21.55 | 15.63 | 2.14 | 65.0% | 29.0% | 0.456 (0.371) |
| F_widthx2_approach_lateral_wrong_prior | 18.8% | -15.9% | 6.1% | 27.8% | 14.91 | 13.14 | 2.16 | 14.5% | 0.0% | 0.556 (0.600) |
| F_widthx2_approach_weaving_size_prior | 11.4% | +1.2% | 4.7% | 91.7% | 1.54 | 3.85 | 2.17 | 0.5% | 0.0% | 0.398 (0.371) |
| F_widthx2_approach_weaving_wrong_prior | 5.8% | -4.0% | 4.2% | 81.1% | 2.94 | 5.15 | 2.23 | 1.0% | 0.0% | 0.593 (0.600) |

## RMS relative range error by true range (main grid A)

| config | 10-20 m | 20-40 m | 40-80 m | 80-120 m | 120-160 m | 160-200 m |
|---|---|---|---|---|---|---|
| A_approach_straight_bearing_only | 93.7% | 134.0% | 140.5% | 146.6% | 134.4% | 138.2% |
| A_approach_straight_size_prior | 37.4% | 39.5% | 42.9% | 45.2% | 50.8% | 73.9% |
| A_approach_straight_tight_prior | 10.1% | 10.3% | 11.2% | 12.7% | 14.7% | 15.0% |
| A_approach_straight_wrong_prior | 27.2% | 27.2% | 27.7% | 32.1% | 36.4% | 39.0% |
| A_approach_straight_wrong_tight | 33.9% | 33.9% | 34.3% | 37.4% | 40.3% | 41.5% |
| A_approach_lateral_bearing_only | 102.9% | 145.1% | 158.7% | 190.7% | 245.2% | 246.8% |
| A_approach_lateral_size_prior | 37.8% | 40.5% | 44.8% | 47.8% | 54.3% | 76.3% |
| A_approach_lateral_tight_prior | 10.0% | 10.3% | 11.1% | 12.8% | 14.7% | 14.3% |
| A_approach_lateral_wrong_prior | 28.1% | 28.1% | 28.7% | 32.9% | 36.9% | 38.4% |
| A_approach_lateral_wrong_tight | 34.1% | 34.1% | 34.5% | 37.6% | 40.3% | 40.8% |
| A_approach_weaving_bearing_only | 57.8% | 80.1% | 191.1% | 524.5% | 322.9% | 130.7% |
| A_approach_weaving_size_prior | 6.2% | 6.8% | 7.6% | 14.0% | 31.3% | 65.4% |
| A_approach_weaving_tight_prior | 3.1% | 3.4% | 3.4% | 4.9% | 6.5% | 11.5% |
| A_approach_weaving_wrong_prior | 7.6% | 8.2% | 9.0% | 12.1% | 16.2% | 25.5% |
| A_approach_weaving_wrong_tight | 11.4% | 12.2% | 13.3% | 17.8% | 23.7% | 33.1% |
