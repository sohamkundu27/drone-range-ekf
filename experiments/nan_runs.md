# Runs whose filter state became NaN (diagnostic, SIMULATION ONLY)

Cause: an iterated-update step put the drone estimate behind the camera (Z < 0), so log(Z) = NaN, and the NaN persists. The metrics in results_sim.md leave those time steps out (nanmean), and its 'final outside 3sigma' column counts a NaN run as not diverged. Not fixed; reported here.

Total: 321 runs over 48 configurations of 200 runs.

| config | NaN runs / 200 | time steps excluded | median first NaN [s] | final outside 3σ: reported | NaN counted as diverged |
|---|---|---|---|---|---|
| A_approach_lateral_size_prior | 4 | 1.9% | 2.5 | 59.5% | 61.5% |
| A_approach_straight_size_prior | 7 | 3.3% | 3.0 | 57.0% | 60.5% |
| A_approach_weaving_size_prior | 14 | 6.3% | 3.1 | 4.5% | 11.5% |
| B_constant_velocity_lateral_size_prior | 1 | 0.5% | 2.7 | 41.5% | 42.0% |
| B_constant_velocity_straight_size_prior | 1 | 0.5% | 1.8 | 40.0% | 40.5% |
| B_constant_velocity_weaving_size_prior | 5 | 2.2% | 5.0 | 36.5% | 39.0% |
| B_gentle_turn_lateral_size_prior | 2 | 0.9% | 2.7 | 75.0% | 76.0% |
| B_gentle_turn_straight_size_prior | 1 | 0.5% | 1.8 | 47.0% | 47.5% |
| B_gentle_turn_weaving_size_prior | 5 | 2.2% | 4.0 | 32.5% | 35.0% |
| C_plainEKF_approach_lateral_size_prior | 5 | 2.2% | 4.7 | 97.0% | 99.5% |
| C_plainEKF_approach_straight_size_prior | 6 | 2.7% | 4.8 | 97.0% | 100.0% |
| C_plainEKF_approach_weaving_size_prior | 14 | 6.2% | 4.9 | 90.5% | 97.5% |
| D_drift_filter_assumes_white_straight | 22 | 9.3% | 5.5 | 87.0% | 98.0% |
| D_drift_filter_assumes_white_weaving | 9 | 4.1% | 3.0 | 4.0% | 8.5% |
| D_precise_pose_straight | 10 | 4.7% | 3.0 | 93.5% | 98.5% |
| D_precise_pose_weaving | 16 | 7.3% | 3.5 | 17.5% | 25.5% |
| D_whitepose_straight | 147 | 27.6% | 26.6 | 20.5% | 94.0% |
| D_whitepose_weaving | 9 | 4.1% | 3.0 | 5.5% | 10.0% |
| E_model_truth_lateral_150m | 4 | 1.9% | 2.2 | 55.0% | 57.0% |
| E_model_truth_lateral_80m | 1 | 0.5% | 1.5 | 31.0% | 31.5% |
| E_model_truth_weaving_150m | 7 | 3.3% | 2.0 | 19.5% | 23.0% |
| E_model_truth_weaving_40m | 1 | 0.5% | 0.1 | 5.0% | 5.5% |
| E_model_truth_weaving_80m | 3 | 1.4% | 1.5 | 8.5% | 10.0% |
| F_widthx2_approach_lateral_size_prior | 4 | 1.7% | 5.8 | 65.0% | 67.0% |
| F_widthx2_approach_straight_size_prior | 6 | 2.8% | 3.0 | 63.5% | 66.5% |
| F_widthx2_approach_weaving_size_prior | 17 | 7.7% | 3.2 | 0.5% | 9.0% |

Configurations not listed had no NaN runs (all bearing-only, tight-prior and wrong-prior rows).
