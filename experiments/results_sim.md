# Simulation results (200 Monte Carlo runs each; SIMULATION ONLY)

Detector noise model: `defaults`.

rel err = (estimated range - true range) / true range, over t >= 10 s. coverage = fraction of time steps (t >= 2 s) with |range error| <= 2 sigma (ideal 95%). ANEES range: mean of the 1-dof range NEES (ideal 1.0). diverged = final error outside +/-3 sigma.

| config | RMS rel err | bias | reported sigma/r | 2-sigma coverage | ANEES range | ANEES pos (3) | mean NIS | final outside 3sigma | final err > 50% | S final (true) |
|---|---|---|---|---|---|---|---|---|---|---|
| A_approach_straight_bearing_only | 126.1% | +95.7% | 9.7% | 22.6% | 22.03 | 23.29 | 2.03 | 99.5% | 91.0% | 0.432 (0.371) |
| A_approach_straight_size_prior | 42.5% | +0.5% | 7.0% | 34.5% | 26.26 | 20.11 | 2.84 | 58.5% | 12.5% | 0.373 (0.371) |
| A_approach_straight_tight_prior | 11.3% | -2.2% | 6.3% | 71.6% | 3.43 | 5.26 | 2.85 | 17.5% | 0.0% | 0.371 (0.354) |
| A_approach_straight_wrong_prior | 29.9% | -28.5% | 6.4% | 1.9% | 47.82 | 35.60 | 2.86 | 79.0% | 1.0% | 0.465 (0.600) |
| A_approach_straight_wrong_tight | 35.8% | -35.5% | 5.9% | 0.0% | 91.49 | 62.87 | 2.86 | 100.0% | 0.0% | 0.413 (0.600) |
| A_approach_lateral_bearing_only | 168.8% | +113.9% | 9.6% | 18.8% | 24.73 | 25.33 | 2.05 | 99.5% | 92.0% | 0.446 (0.371) |
| A_approach_lateral_size_prior | 45.8% | +2.6% | 7.0% | 35.7% | 37.31 | 24.18 | 2.85 | 58.0% | 15.0% | 0.371 (0.371) |
| A_approach_lateral_tight_prior | 11.3% | -2.7% | 6.3% | 72.1% | 3.44 | 5.25 | 2.86 | 17.0% | 0.0% | 0.369 (0.354) |
| A_approach_lateral_wrong_prior | 30.9% | -29.3% | 6.4% | 1.2% | 59.95 | 40.86 | 2.86 | 86.0% | 1.5% | 0.460 (0.600) |
| A_approach_lateral_wrong_tight | 36.0% | -35.7% | 5.9% | 0.0% | 92.11 | 63.06 | 2.87 | 100.0% | 0.0% | 0.412 (0.600) |
| A_approach_weaving_bearing_only | 307.0% | +32.8% | 6.3% | 90.5% | 37.35 | 55.22 | 3.86 | 2.5% | 1.5% | 0.392 (0.371) |
| A_approach_weaving_size_prior | 14.9% | -0.7% | 4.1% | 78.2% | 4.24 | 6.04 | 2.94 | 5.5% | 0.0% | 0.385 (0.371) |
| A_approach_weaving_tight_prior | 4.0% | +0.1% | 4.1% | 95.3% | 1.03 | 3.44 | 2.88 | 0.0% | 0.0% | 0.358 (0.354) |
| A_approach_weaving_wrong_prior | 9.7% | -7.8% | 3.7% | 43.2% | 11.61 | 10.95 | 3.01 | 11.5% | 0.0% | 0.573 (0.600) |
| A_approach_weaving_wrong_tight | 14.5% | -13.8% | 3.6% | 1.5% | 27.18 | 22.71 | 3.12 | 68.0% | 0.0% | 0.547 (0.600) |
| B_constant_velocity_straight_bearing_only | 71.3% | +15.7% | 8.0% | 72.8% | 23.41 | 34.47 | 2.37 | 3.5% | 2.0% | 0.395 (0.371) |
| B_constant_velocity_straight_size_prior | 19.8% | -7.9% | 7.1% | 52.9% | 12.30 | 14.45 | 2.85 | 44.0% | 2.5% | 0.331 (0.371) |
| B_constant_velocity_lateral_bearing_only | 91.1% | +81.8% | 9.1% | 19.9% | 21.09 | 25.07 | 1.93 | 100.0% | 99.5% | 0.420 (0.371) |
| B_constant_velocity_lateral_size_prior | 34.8% | +5.4% | 6.6% | 47.1% | 10.95 | 12.37 | 2.85 | 39.0% | 7.5% | 0.379 (0.371) |
| B_constant_velocity_weaving_bearing_only | 153.9% | +7.1% | 7.3% | 92.5% | 27.94 | 69.77 | 2.96 | 2.5% | 1.5% | 0.386 (0.371) |
| B_constant_velocity_weaving_size_prior | 11.9% | -8.5% | 4.8% | 59.3% | 9.07 | 11.10 | 2.94 | 39.0% | 1.0% | 0.351 (0.371) |
| B_gentle_turn_straight_bearing_only | 163.3% | +18.0% | 8.7% | 73.7% | 9.59 | 19.63 | 2.58 | 6.0% | 2.0% | 0.347 (0.371) |
| B_gentle_turn_straight_size_prior | 23.9% | -16.7% | 7.4% | 42.5% | 31.95 | 22.63 | 2.93 | 49.5% | 5.5% | 0.295 (0.371) |
| B_gentle_turn_lateral_bearing_only | 36.7% | -2.8% | 11.3% | 54.7% | 8.63 | 12.79 | 2.02 | 39.5% | 33.5% | 0.386 (0.371) |
| B_gentle_turn_lateral_size_prior | 30.4% | -21.8% | 7.6% | 30.5% | 27.69 | 29.98 | 2.93 | 75.5% | 10.0% | 0.294 (0.371) |
| B_gentle_turn_weaving_bearing_only | 216.7% | +9.1% | 6.7% | 78.9% | 38.18 | 53.84 | 2.85 | 3.0% | 1.5% | 0.356 (0.371) |
| B_gentle_turn_weaving_size_prior | 13.6% | -11.7% | 4.9% | 41.9% | 12.00 | 14.25 | 3.11 | 32.5% | 0.0% | 0.355 (0.371) |
| C_plainEKF_approach_straight_size_prior | 142.5% | +109.4% | 4.2% | 13.7% | 115.66 | 92.58 | 2.99 | 97.5% | 83.5% | 0.784 (0.371) |
| C_plainEKF_approach_lateral_size_prior | 144.8% | +99.4% | 4.1% | 14.9% | 125.93 | 87.59 | 3.09 | 97.0% | 78.0% | 0.721 (0.371) |
| C_plainEKF_approach_weaving_size_prior | 60.1% | +23.1% | 3.5% | 25.9% | 52.19 | 41.65 | 3.82 | 90.0% | 2.0% | 0.454 (0.371) |
| D_whitepose_straight | 74.6% | -66.9% | 20.0% | 15.3% | 477.79 | 471.44 | 2.62 | 23.0% | 23.5% | 0.117 (0.371) |
| D_drift_filter_assumes_white_straight | 85.1% | -79.4% | 19.3% | 12.0% | 1303.96 | 1104.97 | 0.52 | 87.0% | 89.0% | 0.050 (0.371) |
| D_precise_pose_straight | 80.6% | -76.4% | 9.9% | 6.1% | 2089.80 | 2103.56 | 2.27 | 89.5% | 89.0% | 0.079 (0.371) |
| D_whitepose_weaving | 17.9% | -2.2% | 5.5% | 79.4% | 14.23 | 20.75 | 3.02 | 5.5% | 1.0% | 0.369 (0.371) |
| D_drift_filter_assumes_white_weaving | 17.7% | -1.3% | 5.5% | 90.3% | 19.45 | 43.36 | 1.02 | 4.0% | 1.0% | 0.370 (0.371) |
| D_precise_pose_weaving | 16.8% | -6.7% | 3.8% | 60.6% | 23.77 | 57.48 | 2.71 | 29.0% | 0.5% | 0.361 (0.371) |
| E_model_truth_weaving_40m | 5.8% | -1.6% | 4.0% | 86.1% | 1.90 | 3.95 | 3.00 | 5.0% | 0.0% | 0.371 (0.371) |
| E_model_truth_lateral_40m | 36.6% | +23.5% | 6.8% | 38.6% | 11.84 | 13.42 | 3.05 | 41.0% | 8.0% | 0.425 (0.371) |
| E_model_truth_weaving_80m | 6.8% | -3.5% | 4.2% | 79.9% | 2.64 | 4.87 | 3.01 | 8.5% | 0.0% | 0.369 (0.371) |
| E_model_truth_lateral_80m | 22.7% | +2.5% | 7.0% | 53.4% | 7.77 | 9.99 | 2.98 | 30.0% | 3.5% | 0.363 (0.371) |
| E_model_truth_weaving_150m | 12.4% | -7.6% | 4.9% | 59.4% | 10.48 | 12.09 | 3.10 | 22.0% | 0.5% | 0.364 (0.371) |
| E_model_truth_lateral_150m | 31.6% | -24.1% | 8.3% | 33.6% | 27.64 | 30.24 | 2.94 | 55.5% | 11.5% | 0.288 (0.371) |
| F_widthx2_approach_straight_size_prior | 51.5% | +21.2% | 7.7% | 45.3% | 12.71 | 12.96 | 2.12 | 59.5% | 27.0% | 0.439 (0.371) |
| F_widthx2_approach_straight_wrong_prior | 19.9% | -17.5% | 7.2% | 24.5% | 14.34 | 13.85 | 2.13 | 25.5% | 0.0% | 0.535 (0.600) |
| F_widthx2_approach_lateral_size_prior | 54.6% | +25.2% | 7.6% | 44.4% | 12.81 | 13.17 | 2.13 | 63.0% | 29.0% | 0.445 (0.371) |
| F_widthx2_approach_lateral_wrong_prior | 20.0% | -17.7% | 7.2% | 23.7% | 14.73 | 13.80 | 2.14 | 18.5% | 0.0% | 0.533 (0.600) |
| F_widthx2_approach_weaving_size_prior | 14.0% | +1.2% | 5.2% | 93.1% | 1.40 | 3.81 | 2.16 | 0.0% | 0.0% | 0.395 (0.371) |
| F_widthx2_approach_weaving_wrong_prior | 5.8% | -4.0% | 4.7% | 83.3% | 2.61 | 5.01 | 2.22 | 0.5% | 0.0% | 0.587 (0.600) |

## RMS relative range error by true range (main grid A)

| config | 10-20 m | 20-40 m | 40-80 m | 80-120 m | 120-160 m | 160-200 m |
|---|---|---|---|---|---|---|
| A_approach_straight_bearing_only | 86.0% | 122.7% | 132.8% | 140.9% | 129.6% | 137.7% |
| A_approach_straight_size_prior | 36.6% | 38.6% | 41.1% | 47.4% | 53.1% | 74.6% |
| A_approach_straight_tight_prior | 10.2% | 10.2% | 10.9% | 12.8% | 14.9% | 15.2% |
| A_approach_straight_wrong_prior | 28.5% | 28.4% | 28.6% | 33.0% | 37.0% | 39.2% |
| A_approach_straight_wrong_tight | 34.8% | 34.7% | 34.9% | 38.1% | 40.8% | 41.7% |
| A_approach_lateral_bearing_only | 95.2% | 134.1% | 160.0% | 214.5% | 279.1% | 252.0% |
| A_approach_lateral_size_prior | 38.1% | 40.9% | 44.6% | 51.4% | 57.5% | 77.6% |
| A_approach_lateral_tight_prior | 10.2% | 10.2% | 10.9% | 13.0% | 14.9% | 14.5% |
| A_approach_lateral_wrong_prior | 29.5% | 29.4% | 29.6% | 33.9% | 37.7% | 38.8% |
| A_approach_lateral_wrong_tight | 35.0% | 34.9% | 35.0% | 38.3% | 40.8% | 41.0% |
| A_approach_weaving_bearing_only | 77.7% | 127.5% | 223.2% | 520.5% | 311.6% | 131.5% |
| A_approach_weaving_size_prior | 6.1% | 6.5% | 7.0% | 19.0% | 35.9% | 66.5% |
| A_approach_weaving_tight_prior | 3.9% | 3.6% | 3.3% | 4.9% | 6.5% | 11.5% |
| A_approach_weaving_wrong_prior | 8.0% | 8.4% | 8.7% | 12.0% | 16.0% | 25.4% |
| A_approach_weaving_wrong_tight | 11.8% | 12.6% | 13.1% | 18.0% | 23.8% | 33.3% |
