# Runs whose filter state became NaN (diagnostic, SIMULATION ONLY) - accelerating-camera grid

Cause: an iterated-update step put the drone estimate behind the camera (Z < 0), so log(Z) = NaN, and the NaN persists. The metrics in results_sim.md leave those time steps out (nanmean), and its 'final outside 3sigma' column counts a NaN run as not diverged. Not fixed; reported here.

Total: 272 runs over 48 configurations of 200 runs.

| config | NaN runs / 200 | time steps excluded | median first NaN [s] | final outside 3σ: reported | NaN counted as diverged |
|---|---|---|---|---|---|
| A_approach_lat_surge_15_size_prior | 14 | 6.3% | 3.4 | 3.5% | 10.5% |
| A_approach_surge_15_size_prior | 7 | 3.3% | 3.0 | 12.5% | 16.0% |
| A_approach_weaving_size_prior | 14 | 6.3% | 3.1 | 4.5% | 11.5% |
| B_constant_velocity_lat_surge_15_size_prior | 7 | 3.0% | 5.0 | 17.5% | 21.0% |
| B_constant_velocity_surge_15_size_prior | 2 | 0.9% | 5.4 | 17.5% | 18.5% |
| B_constant_velocity_weaving_size_prior | 5 | 2.2% | 5.0 | 36.5% | 39.0% |
| B_gentle_turn_lat_surge_15_size_prior | 8 | 3.4% | 5.1 | 48.0% | 52.0% |
| B_gentle_turn_surge_15_size_prior | 1 | 0.5% | 1.8 | 25.5% | 26.0% |
| B_gentle_turn_weaving_size_prior | 5 | 2.2% | 4.0 | 32.5% | 35.0% |
| C_plainEKF_approach_lat_surge_15_size_prior | 13 | 5.7% | 4.8 | 89.5% | 96.0% |
| C_plainEKF_approach_surge_15_size_prior | 5 | 2.2% | 4.7 | 97.0% | 99.5% |
| C_plainEKF_approach_weaving_size_prior | 14 | 6.2% | 4.9 | 90.5% | 97.5% |
| D_drift_filter_assumes_white_surge_15 | 29 | 10.1% | 7.2 | 60.0% | 74.5% |
| D_drift_filter_assumes_white_weaving | 9 | 4.1% | 3.0 | 4.0% | 8.5% |
| D_precise_pose_surge_15 | 9 | 4.2% | 3.0 | 84.0% | 88.5% |
| D_precise_pose_weaving | 16 | 7.3% | 3.5 | 17.5% | 25.5% |
| D_whitepose_surge_15 | 41 | 9.2% | 27.7 | 37.5% | 58.0% |
| D_whitepose_weaving | 9 | 4.1% | 3.0 | 5.5% | 10.0% |
| E_model_truth_lat_surge_15_150m | 8 | 3.8% | 2.0 | 17.5% | 21.5% |
| E_model_truth_lat_surge_15_40m | 2 | 1.0% | 0.2 | 6.0% | 7.0% |
| E_model_truth_lat_surge_15_80m | 5 | 2.4% | 0.8 | 12.0% | 14.5% |
| E_model_truth_weaving_150m | 7 | 3.3% | 2.0 | 19.5% | 23.0% |
| E_model_truth_weaving_40m | 1 | 0.5% | 0.1 | 5.0% | 5.5% |
| E_model_truth_weaving_80m | 3 | 1.4% | 1.5 | 8.5% | 10.0% |
| F_widthx2_approach_lat_surge_15_size_prior | 15 | 6.7% | 3.4 | 0.0% | 7.5% |
| F_widthx2_approach_surge_15_size_prior | 6 | 2.8% | 3.0 | 6.5% | 9.5% |
| F_widthx2_approach_weaving_size_prior | 17 | 7.7% | 3.2 | 0.5% | 9.0% |

Configurations not listed had no NaN runs (21 of 48).
