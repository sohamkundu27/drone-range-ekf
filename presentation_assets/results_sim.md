# Simulation results (200 Monte Carlo runs each; SIMULATION ONLY)

Detector noise model: `detector_noise.json`.

rel err = (estimated range - true range) / true range, over t >= 10 s. coverage = fraction of time steps (t >= 2 s) with |range error| <= 2 sigma (ideal 95%). ANEES range: mean of the 1-dof range NEES (ideal 1.0). diverged = final error outside +/-3 sigma.

| config | RMS rel err | bias | reported sigma/r | 2-sigma coverage | ANEES range | ANEES pos (3) | mean NIS | final outside 3sigma | final err > 50% | S final (true) |
|---|---|---|---|---|---|---|---|---|---|---|
| A_approach_straight_bearing_only | 134.8% | +105.6% | 9.3% | 21.5% | 26.76 | 27.38 | 2.07 | 100.0% | 99.5% | 0.439 (0.371) |
| A_approach_straight_size_prior | 42.5% | +2.1% | 6.7% | 33.3% | 23.43 | 19.22 | 2.85 | 57.0% | 15.5% | 0.382 (0.371) |
| A_approach_straight_tight_prior | 11.3% | -1.4% | 6.0% | 71.0% | 3.53 | 5.28 | 2.86 | 23.0% | 0.0% | 0.377 (0.354) |
| A_approach_straight_wrong_prior | 28.7% | -27.1% | 5.9% | 3.2% | 47.74 | 34.30 | 2.88 | 73.0% | 0.5% | 0.481 (0.600) |
| A_approach_straight_wrong_tight | 35.0% | -34.7% | 5.4% | 0.0% | 94.65 | 61.85 | 2.88 | 100.0% | 0.0% | 0.423 (0.600) |
| A_approach_lateral_bearing_only | 161.2% | +121.5% | 9.2% | 17.8% | 29.83 | 29.94 | 2.09 | 100.0% | 99.5% | 0.454 (0.371) |
| A_approach_lateral_size_prior | 44.9% | +3.9% | 6.8% | 35.9% | 41.61 | 23.07 | 2.86 | 59.5% | 16.0% | 0.381 (0.371) |
| A_approach_lateral_tight_prior | 11.3% | -1.8% | 6.0% | 71.7% | 3.49 | 5.24 | 2.86 | 22.5% | 0.0% | 0.375 (0.354) |
| A_approach_lateral_wrong_prior | 29.5% | -27.7% | 5.9% | 2.6% | 59.93 | 39.03 | 2.88 | 76.0% | 1.5% | 0.477 (0.600) |
| A_approach_lateral_wrong_tight | 35.2% | -34.9% | 5.4% | 0.0% | 95.22 | 61.96 | 2.89 | 100.0% | 0.0% | 0.422 (0.600) |
| A_approach_weaving_bearing_only | 321.4% | +31.6% | 6.2% | 89.8% | 51.95 | 36.52 | 3.43 | 2.0% | 0.5% | 0.393 (0.371) |
| A_approach_weaving_size_prior | 12.5% | -0.8% | 3.7% | 77.7% | 3.92 | 5.46 | 2.93 | 4.5% | 0.0% | 0.387 (0.371) |
| A_approach_weaving_tight_prior | 4.0% | +0.3% | 3.7% | 94.9% | 1.07 | 3.46 | 2.88 | 0.0% | 0.0% | 0.360 (0.354) |
| A_approach_weaving_wrong_prior | 9.7% | -7.8% | 3.3% | 40.3% | 13.00 | 11.25 | 3.04 | 11.5% | 0.0% | 0.578 (0.600) |
| A_approach_weaving_wrong_tight | 14.5% | -13.7% | 3.2% | 0.3% | 29.34 | 22.91 | 3.15 | 66.5% | 0.0% | 0.552 (0.600) |
| B_constant_velocity_straight_bearing_only | 72.1% | +15.8% | 7.8% | 68.6% | 22.09 | 33.92 | 2.39 | 5.5% | 2.5% | 0.400 (0.371) |
| B_constant_velocity_straight_size_prior | 19.3% | -5.9% | 6.9% | 55.3% | 10.63 | 12.78 | 2.85 | 40.0% | 2.0% | 0.340 (0.371) |
| B_constant_velocity_lateral_bearing_only | 93.6% | +86.8% | 8.8% | 18.6% | 23.83 | 26.37 | 1.94 | 99.5% | 99.5% | 0.423 (0.371) |
| B_constant_velocity_lateral_size_prior | 33.9% | +7.2% | 6.4% | 49.0% | 9.92 | 11.51 | 2.87 | 41.5% | 10.5% | 0.390 (0.371) |
| B_constant_velocity_weaving_bearing_only | 688.5% | +33.1% | 7.2% | 93.0% | 55.43 | 90.04 | 3.21 | 2.5% | 2.5% | 0.395 (0.371) |
| B_constant_velocity_weaving_size_prior | 10.8% | -7.9% | 4.8% | 61.6% | 6.43 | 8.92 | 2.92 | 36.5% | 0.5% | 0.353 (0.371) |
| B_gentle_turn_straight_bearing_only | 205.6% | +20.6% | 8.5% | 68.4% | 7.40 | 16.15 | 2.32 | 10.0% | 2.5% | 0.350 (0.371) |
| B_gentle_turn_straight_size_prior | 23.1% | -15.6% | 7.2% | 45.6% | 39.65 | 23.02 | 2.94 | 47.0% | 5.5% | 0.302 (0.371) |
| B_gentle_turn_lateral_bearing_only | 35.9% | -0.4% | 11.0% | 55.2% | 8.14 | 12.29 | 2.01 | 36.5% | 28.5% | 0.388 (0.371) |
| B_gentle_turn_lateral_size_prior | 29.7% | -20.7% | 7.5% | 30.8% | 24.80 | 27.17 | 2.94 | 75.0% | 9.0% | 0.301 (0.371) |
| B_gentle_turn_weaving_bearing_only | 150.8% | +11.5% | 6.6% | 79.2% | 59.76 | 95.76 | 3.61 | 5.0% | 2.5% | 0.357 (0.371) |
| B_gentle_turn_weaving_size_prior | 13.3% | -11.3% | 4.9% | 42.4% | 10.61 | 13.20 | 3.11 | 32.5% | 0.0% | 0.356 (0.371) |
| C_plainEKF_approach_straight_size_prior | 144.9% | +111.9% | 3.9% | 13.7% | 130.97 | 99.92 | 2.97 | 97.0% | 85.0% | 0.804 (0.371) |
| C_plainEKF_approach_lateral_size_prior | 146.3% | +101.3% | 3.8% | 14.5% | 118.96 | 90.83 | 3.03 | 97.0% | 79.5% | 0.738 (0.371) |
| C_plainEKF_approach_weaving_size_prior | 60.6% | +23.3% | 3.1% | 24.6% | 86.76 | 45.09 | 3.86 | 90.5% | 2.0% | 0.458 (0.371) |
| D_whitepose_straight | 74.4% | -67.4% | 20.2% | 15.2% | 503.53 | 468.24 | 2.61 | 20.5% | 20.5% | 0.119 (0.371) |
| D_drift_filter_assumes_white_straight | 84.8% | -79.6% | 18.9% | 12.0% | 1409.07 | 1099.06 | 0.52 | 87.0% | 89.0% | 0.050 (0.371) |
| D_precise_pose_straight | 77.8% | -74.4% | 9.4% | 6.0% | 1815.94 | 2574.77 | 2.27 | 93.5% | 90.5% | 0.094 (0.371) |
| D_whitepose_weaving | 16.1% | -2.6% | 5.0% | 77.7% | 16.81 | 22.25 | 3.04 | 5.5% | 1.0% | 0.369 (0.371) |
| D_drift_filter_assumes_white_weaving | 16.2% | -1.4% | 5.0% | 89.8% | 22.40 | 44.20 | 0.99 | 4.0% | 1.0% | 0.370 (0.371) |
| D_precise_pose_weaving | 13.4% | -5.3% | 3.5% | 65.8% | 13.62 | 48.63 | 2.61 | 17.5% | 0.0% | 0.371 (0.371) |
| E_model_truth_weaving_40m | 5.7% | -0.9% | 3.9% | 87.0% | 1.80 | 3.86 | 3.02 | 5.0% | 0.0% | 0.374 (0.371) |
| E_model_truth_lateral_40m | 39.2% | +27.3% | 6.6% | 33.7% | 13.52 | 15.02 | 3.08 | 42.0% | 10.5% | 0.442 (0.371) |
| E_model_truth_weaving_80m | 6.8% | -3.2% | 4.1% | 80.5% | 2.62 | 4.83 | 3.02 | 8.5% | 0.0% | 0.372 (0.371) |
| E_model_truth_lateral_80m | 23.3% | +4.7% | 6.8% | 54.1% | 7.38 | 9.73 | 3.00 | 31.0% | 3.0% | 0.374 (0.371) |
| E_model_truth_weaving_150m | 12.1% | -7.9% | 4.9% | 59.7% | 7.24 | 10.01 | 3.09 | 19.5% | 0.0% | 0.367 (0.371) |
| E_model_truth_lateral_150m | 31.0% | -23.3% | 8.2% | 34.6% | 29.49 | 31.16 | 2.94 | 55.0% | 10.5% | 0.292 (0.371) |
| F_widthx2_approach_straight_size_prior | 52.1% | +23.0% | 7.1% | 43.8% | 14.39 | 13.88 | 2.13 | 63.5% | 28.5% | 0.452 (0.371) |
| F_widthx2_approach_straight_wrong_prior | 18.7% | -15.6% | 6.2% | 29.1% | 14.61 | 13.33 | 2.15 | 16.5% | 0.0% | 0.558 (0.600) |
| F_widthx2_approach_lateral_size_prior | 57.1% | +27.8% | 7.1% | 43.4% | 14.04 | 13.81 | 2.14 | 65.0% | 29.5% | 0.459 (0.371) |
| F_widthx2_approach_lateral_wrong_prior | 18.3% | -15.2% | 6.2% | 30.6% | 13.85 | 12.46 | 2.16 | 14.0% | 0.0% | 0.559 (0.600) |
| F_widthx2_approach_weaving_size_prior | 11.5% | +1.4% | 4.8% | 92.0% | 1.52 | 3.84 | 2.17 | 0.5% | 0.0% | 0.398 (0.371) |
| F_widthx2_approach_weaving_wrong_prior | 5.6% | -3.8% | 4.4% | 82.4% | 2.74 | 5.01 | 2.23 | 1.0% | 0.0% | 0.594 (0.600) |

## RMS relative range error by true range (main grid A)

| config | 10-20 m | 20-40 m | 40-80 m | 80-120 m | 120-160 m | 160-200 m |
|---|---|---|---|---|---|---|
| A_approach_straight_bearing_only | 94.4% | 134.3% | 141.2% | 148.1% | 139.0% | 143.0% |
| A_approach_straight_size_prior | 37.3% | 39.4% | 42.7% | 44.5% | 50.6% | 75.2% |
| A_approach_straight_tight_prior | 10.1% | 10.3% | 11.2% | 12.6% | 14.8% | 15.2% |
| A_approach_straight_wrong_prior | 26.9% | 26.8% | 27.6% | 31.9% | 36.2% | 38.9% |
| A_approach_straight_wrong_tight | 33.8% | 33.7% | 34.3% | 37.3% | 40.2% | 41.5% |
| A_approach_lateral_bearing_only | 102.9% | 145.1% | 158.8% | 192.1% | 256.5% | 257.7% |
| A_approach_lateral_size_prior | 38.3% | 41.0% | 45.2% | 47.6% | 54.4% | 77.4% |
| A_approach_lateral_tight_prior | 10.1% | 10.3% | 11.1% | 12.8% | 14.8% | 14.4% |
| A_approach_lateral_wrong_prior | 27.8% | 27.7% | 28.4% | 32.6% | 36.7% | 38.2% |
| A_approach_lateral_wrong_tight | 33.9% | 33.8% | 34.4% | 37.5% | 40.3% | 40.8% |
| A_approach_weaving_bearing_only | 32.1% | 40.8% | 183.2% | 574.2% | 410.1% | 137.9% |
| A_approach_weaving_size_prior | 5.5% | 6.2% | 6.1% | 13.1% | 31.4% | 66.9% |
| A_approach_weaving_tight_prior | 3.3% | 3.4% | 3.4% | 4.9% | 6.5% | 11.6% |
| A_approach_weaving_wrong_prior | 7.6% | 8.0% | 9.0% | 12.0% | 16.0% | 25.4% |
| A_approach_weaving_wrong_tight | 11.4% | 12.1% | 13.3% | 17.8% | 23.7% | 33.2% |
