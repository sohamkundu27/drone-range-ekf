# Runs whose filter state became NaN (diagnostic, SIMULATION ONLY)

Cause: an iterated-update step put the drone estimate behind the camera (Z < 0), so log(Z) = NaN, and the NaN persists. The metrics in results_sim.md leave those time steps out (nanmean), and its 'final outside 3sigma' column counts a NaN run as not diverged. Not fixed; reported here.

Total: 315 runs over 48 configurations of 200 runs.

| config | NaN runs / 200 | time steps excluded | median first NaN [s] | final outside 3σ: reported | NaN counted as diverged |
|---|---|---|---|---|---|
| A_approach_lateral_size_prior | 4 | 1.9% | 2.5 | 58.5% | 60.5% |
| A_approach_straight_size_prior | 7 | 3.3% | 3.0 | 58.0% | 61.5% |
| A_approach_weaving_size_prior | 14 | 6.3% | 3.1 | 4.5% | 11.5% |
| B_constant_velocity_lateral_size_prior | 1 | 0.5% | 2.7 | 40.5% | 41.0% |
| B_constant_velocity_straight_size_prior | 1 | 0.5% | 1.8 | 40.5% | 41.0% |
| B_constant_velocity_weaving_size_prior | 5 | 2.2% | 5.0 | 37.0% | 39.5% |
| B_gentle_turn_lateral_size_prior | 2 | 0.9% | 2.7 | 75.5% | 76.5% |
| B_gentle_turn_straight_size_prior | 1 | 0.5% | 1.8 | 47.5% | 48.0% |
| B_gentle_turn_weaving_size_prior | 5 | 2.2% | 4.0 | 31.5% | 34.0% |
| C_plainEKF_approach_lateral_size_prior | 4 | 1.8% | 4.2 | 97.5% | 99.5% |
| C_plainEKF_approach_straight_size_prior | 5 | 2.2% | 4.7 | 97.5% | 100.0% |
| C_plainEKF_approach_weaving_size_prior | 14 | 6.2% | 4.9 | 91.0% | 98.0% |
| D_drift_filter_assumes_white_straight | 18 | 7.7% | 4.8 | 90.0% | 99.0% |
| D_drift_filter_assumes_white_weaving | 9 | 4.1% | 3.0 | 3.0% | 7.5% |
| D_precise_pose_straight | 12 | 5.5% | 3.2 | 93.0% | 99.0% |
| D_precise_pose_weaving | 17 | 7.6% | 3.7 | 17.5% | 26.0% |
| D_whitepose_straight | 146 | 27.4% | 26.6 | 21.0% | 94.0% |
| D_whitepose_weaving | 9 | 4.1% | 3.0 | 5.5% | 10.0% |
| E_model_truth_lateral_150m | 4 | 1.9% | 2.2 | 55.0% | 57.0% |
| E_model_truth_weaving_150m | 7 | 3.1% | 2.2 | 20.0% | 23.5% |
| E_model_truth_weaving_40m | 1 | 0.5% | 0.1 | 5.5% | 6.0% |
| E_model_truth_weaving_80m | 3 | 1.4% | 1.5 | 8.0% | 9.5% |
| F_widthx2_approach_lateral_size_prior | 3 | 1.3% | 6.7 | 65.0% | 66.5% |
| F_widthx2_approach_straight_size_prior | 6 | 2.8% | 3.0 | 63.0% | 66.0% |
| F_widthx2_approach_weaving_size_prior | 17 | 7.7% | 3.2 | 0.5% | 9.0% |

Configurations not listed had no NaN runs (all bearing-only, tight-prior and wrong-prior rows).
