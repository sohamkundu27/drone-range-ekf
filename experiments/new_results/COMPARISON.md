# Simulation with the NEW detector noise vs the OLD (SIMULATION ONLY)

Same sim/EKF code (byte-identical copies, see `detfly_retrain/sim_shadow_check.md`), same seeds; only `detector_noise.json` differs (new = box noise of the retrained s on held-out DUT test drones, `detector/measure_noise.py`). The known NaN behaviour is unchanged and not fixed.

## Did the measured noise change?

Matched DUT test drones: 2196 → 2202 of 2245.

| true width px | n matched | recall | centre σ px | log-width σ | log-width bias |
|---|---|---|---|---|---|
| 16-24 | 126 → 126 | 0.962 → 0.962 | 0.88 → 0.92 | 0.094 → 0.092 | +0.073 → +0.086 |
| 24-32 | 230 → 232 | 0.979 → 0.987 | 0.89 → 0.84 | 0.073 → 0.071 | +0.034 → +0.034 |
| 32-48 | 543 → 541 | 0.989 → 0.985 | 1.06 → 0.99 | 0.047 → 0.051 | +0.007 → +0.007 |
| 48-64 | 228 → 228 | 0.966 → 0.966 | 1.02 → 1.02 | 0.047 → 0.041 | +0.004 → +0.008 |
| 64-96 | 193 → 195 | 0.970 → 0.980 | 1.18 → 1.07 | 0.045 → 0.034 | +0.009 → +0.004 |
| 96-160 | 236 → 238 | 0.959 → 0.967 | 1.42 → 1.51 | 0.030 → 0.030 | +0.012 → +0.009 |
| 160-inf | 625 → 627 | 0.986 → 0.989 | 2.90 → 2.81 | 0.029 → 0.030 | +0.000 → +0.002 |

**The noise barely changed**: every robust σ moved by ≤ ~0.1 px (centre) or ≤ 0.011 (log width), recall by ≤ 0.01. Differences in the sim below are therefore mostly Monte-Carlo sensitivity, not a better or worse detector model.

## NaN runs: 321 → 315 (of 9,600)

## Per configuration (old → new)

| config | RMS rel err | 2σ coverage | ANEES range | ANEES w/o worst run (old → new) | NaN runs |
|---|---|---|---|---|---|
| A_approach_straight_bearing_only | 134.8% → 133.9% | 21.5% → 21.6% | 26.76 → 26.69 | 26.55 → 26.48 (worst run 67 → 68) | 0 → 0 |
| A_approach_straight_size_prior | 42.5% → 42.7% | 33.3% → 33.4% | 23.43 → 23.71 | 21.85 → 22.15 (worst run 222 → 218) | 7 → 7 |
| A_approach_straight_tight_prior | 11.3% → 11.3% | 71.0% → 70.9% | 3.53 → 3.56 | 3.43 → 3.46 (worst run 23 → 24) | 0 → 0 |
| A_approach_straight_wrong_prior | 28.7% → 28.9% | 3.2% → 2.9% | 47.74 → 48.78 | 45.34 → 46.41 (worst run 525 → 521) | 0 → 0 |
| A_approach_straight_wrong_tight | 35.0% → 35.1% | 0.0% → 0.0% | 94.65 → 96.15 | 94.22 → 95.72 (worst run 180 → 183) | 0 → 0 |
| A_approach_lateral_bearing_only | 161.2% → 160.2% | 17.8% → 17.8% | 29.83 → 29.76 | 29.60 → 29.54 (worst run 79 → 78) | 0 → 0 |
| A_approach_lateral_size_prior | 44.9% → 44.6% | 35.9% → 35.8% | 41.61 → 23.37 | 22.20 → 22.24 (worst run 3774 → 201) | 4 → 4 |
| A_approach_lateral_tight_prior | 11.3% → 11.3% | 71.7% → 71.5% | 3.49 → 3.54 | 3.40 → 3.44 (worst run 22 → 23) | 0 → 0 |
| A_approach_lateral_wrong_prior | 29.5% → 29.8% | 2.6% → 2.2% | 59.93 → 61.05 | 51.35 → 52.76 (worst run 1767 → 1712) | 0 → 0 |
| A_approach_lateral_wrong_tight | 35.2% → 35.3% | 0.0% → 0.0% | 95.22 → 96.79 | 94.80 → 96.37 (worst run 178 → 181) | 0 → 0 |
| A_approach_weaving_bearing_only | 321.4% → 297.1% | 89.8% → 89.9% | 51.95 → 35.75 | 34.33 → 23.25 (worst run 4076 → 2627) | 0 → 0 |
| A_approach_weaving_size_prior | 12.5% → 13.1% | 77.7% → 77.1% | 3.92 → 22.53 ⚠ | 3.61 → 3.76 (worst run 38 → 3478) | 14 → 14 |
| A_approach_weaving_tight_prior | 4.0% → 3.9% | 94.9% → 94.9% | 1.07 → 1.06 | 1.06 → 1.05 (worst run 4 → 4) | 0 → 0 |
| A_approach_weaving_wrong_prior | 9.7% → 9.8% | 40.3% → 38.7% | 13.00 → 13.29 | 10.63 → 10.95 (worst run 484 → 480) | 0 → 0 |
| A_approach_weaving_wrong_tight | 14.5% → 14.5% | 0.3% → 0.3% | 29.34 → 30.03 | 29.15 → 29.83 (worst run 68 → 70) | 0 → 0 |
| B_constant_velocity_straight_bearing_only | 72.1% → 190.4% | 68.6% → 69.7% | 22.09 → 35.02 ⚠ | 9.03 → 20.66 (worst run 2620 → 2891) | 0 → 0 |
| B_constant_velocity_straight_size_prior | 19.3% → 19.0% | 55.3% → 54.8% | 10.63 → 10.32 | 9.70 → 9.65 (worst run 196 → 141) | 1 → 1 |
| B_constant_velocity_lateral_bearing_only | 93.6% → 98.5% | 18.6% → 18.4% | 23.83 → 23.76 | 23.73 → 23.33 (worst run 44 → 111) | 0 → 0 |
| B_constant_velocity_lateral_size_prior | 33.9% → 34.0% | 49.0% → 48.7% | 9.92 → 10.07 | 9.05 → 9.15 (worst run 174 → 184) | 1 → 1 |
| B_constant_velocity_weaving_bearing_only | 688.5% → 159.3% | 93.0% → 92.8% | 55.43 → 29.63 | 22.70 → 9.52 (worst run 6574 → 4077) | 0 → 0 |
| B_constant_velocity_weaving_size_prior | 10.8% → 10.6% | 61.6% → 61.3% | 6.43 → 6.49 | 5.97 → 6.05 (worst run 76 → 74) | 5 → 5 |
| B_gentle_turn_straight_bearing_only | 205.6% → 117.7% | 68.4% → 69.2% | 7.40 → 18.46 ⚠ | 5.33 → 10.76 (worst run 419 → 1551) | 0 → 0 |
| B_gentle_turn_straight_size_prior | 23.1% → 22.7% | 45.6% → 45.2% | 39.65 → 22.79 | 20.18 → 14.35 (worst run 3893 → 1692) | 1 → 1 |
| B_gentle_turn_lateral_bearing_only | 35.9% → 36.3% | 55.2% → 55.3% | 8.14 → 7.94 | 7.29 → 7.28 (worst run 176 → 139) | 0 → 0 |
| B_gentle_turn_lateral_size_prior | 29.7% → 29.8% | 30.8% → 30.6% | 24.80 → 25.05 | 23.51 → 23.76 (worst run 233 → 233) | 2 → 2 |
| B_gentle_turn_weaving_bearing_only | 150.8% → 136.0% | 79.2% → 78.9% | 59.76 → 183.12 ⚠ | 32.66 → 114.40 (worst run 5481 → 13882) | 0 → 0 |
| B_gentle_turn_weaving_size_prior | 13.3% → 13.4% | 42.4% → 42.0% | 10.61 → 10.77 | 9.81 → 10.02 (worst run 125 → 115) | 5 → 5 |
| C_plainEKF_approach_straight_size_prior | 144.9% → 145.2% | 13.7% → 13.6% | 130.97 → 132.33 | 127.03 → 128.90 (worst run 240 → 239) | 6 → 5 |
| C_plainEKF_approach_lateral_size_prior | 146.3% → 146.4% | 14.5% → 14.5% | 118.96 → 127.07 | 114.33 → 117.79 (worst run 558 → 1548) | 5 → 4 |
| C_plainEKF_approach_weaving_size_prior | 60.6% → 59.9% | 24.6% → 24.8% | 86.76 → 53.56 | 43.41 → 40.96 (worst run 7807 → 2080) | 14 → 14 |
| D_whitepose_straight | 74.4% → 74.4% | 15.2% → 15.1% | 503.53 → 507.38 | 548.03 → 549.13 (worst run 4538 → 4594) | 147 → 146 |
| D_drift_filter_assumes_white_straight | 84.8% → 84.8% | 12.0% → 11.7% | 1409.07 → 1415.53 | 1259.42 → 1281.84 (worst run 7033 → 7303) | 22 → 18 |
| D_precise_pose_straight | 77.8% → 78.1% | 6.0% → 5.8% | 1815.94 → 2034.77 | 1671.05 → 1761.57 (worst run 17157 → 37444) | 10 → 12 |
| D_whitepose_weaving | 16.1% → 15.8% | 77.7% → 77.8% | 16.81 → 16.93 | 10.56 → 9.59 (worst run 1141 → 1357) | 9 → 9 |
| D_drift_filter_assumes_white_weaving | 16.2% → 16.1% | 89.8% → 89.6% | 22.40 → 22.45 | 13.95 → 13.60 (worst run 1531 → 1609) | 9 → 9 |
| D_precise_pose_weaving | 13.4% → 13.4% | 65.8% → 65.0% | 13.62 → 14.93 | 16.29 → 23.18 (worst run 963 → 1783) | 16 → 17 |
| E_model_truth_weaving_40m | 5.7% → 5.6% | 87.0% → 87.0% | 1.80 → 1.80 | 1.77 → 1.77 (worst run 8 → 8) | 1 → 1 |
| E_model_truth_lateral_40m | 39.2% → 39.0% | 33.7% → 33.6% | 13.52 → 13.59 | 13.30 → 13.21 (worst run 57 → 89) | 0 → 0 |
| E_model_truth_weaving_80m | 6.8% → 6.7% | 80.5% → 80.4% | 2.62 → 2.58 | 2.48 → 2.48 (worst run 27 → 21) | 3 → 3 |
| E_model_truth_lateral_80m | 23.3% → 23.5% | 54.1% → 53.7% | 7.38 → 7.66 | 7.18 → 7.39 (worst run 47 → 61) | 1 → 0 |
| E_model_truth_weaving_150m | 12.1% → 11.3% | 59.7% → 59.5% | 7.24 → 9.71 | 6.65 → 7.45 (worst run 102 → 519) | 7 → 7 |
| E_model_truth_lateral_150m | 31.0% → 30.9% | 34.6% → 34.3% | 29.49 → 28.13 | 24.03 → 23.58 (worst run 1019 → 842) | 4 → 4 |
| F_widthx2_approach_straight_size_prior | 52.1% → 52.0% | 43.8% → 44.2% | 14.39 → 14.30 | 13.39 → 13.27 (worst run 164 → 171) | 6 → 6 |
| F_widthx2_approach_straight_wrong_prior | 18.7% → 19.0% | 29.1% → 27.3% | 14.61 → 15.34 | 13.97 → 14.66 (worst run 142 → 152) | 0 → 0 |
| F_widthx2_approach_lateral_size_prior | 57.1% → 57.0% | 43.4% → 43.1% | 14.04 → 21.55 ⚠ | 13.35 → 14.05 (worst run 116 → 1470) | 4 → 3 |
| F_widthx2_approach_lateral_wrong_prior | 18.3% → 18.8% | 30.6% → 27.8% | 13.85 → 14.91 | 12.91 → 13.93 (worst run 200 → 211) | 0 → 0 |
| F_widthx2_approach_weaving_size_prior | 11.5% → 11.4% | 92.0% → 91.7% | 1.52 → 1.54 | 1.35 → 1.37 (worst run 21 → 22) | 17 → 17 |
| F_widthx2_approach_weaving_wrong_prior | 5.6% → 5.8% | 82.4% → 81.1% | 2.74 → 2.94 | 2.64 → 2.83 (worst run 23 → 25) | 0 → 0 |

⚠ = ANEES changed by more than max(1, 50 %). Check the 'w/o worst run' column: where it barely moves, the change comes from a single run whose NEES exploded (finite, so it is not counted as a NaN run), not from a shift of the typical run. 'w/o worst run' is the mean of per-run mean NEES over the finite runs minus the worst one (an approximation of ANEES: results_sim.json averages over time steps).
