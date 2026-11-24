# Runs whose filter state became NaN (diagnostic, SIMULATION ONLY) - sweep study

Cause: an iterated-update step put the drone estimate behind the camera (Z < 0), so log(Z) = NaN, and the NaN persists. The metrics in results_sweep.md leave those time steps out (nanmean), and its 'final outside 3sigma' column counts a NaN run as not diverged. Not fixed; reported here. The last two columns are added for this study: 2-sigma coverage as reported (NaN steps skipped) and with every NaN step after initialisation counted as a miss (t >= 2 s, true range <= 200 m).

Total: 188 runs over 20 configurations of 200 runs.

| config | NaN runs / 200 | time steps excluded | median first NaN [s] | final outside 3σ: reported | NaN counted as diverged | 2σ coverage: reported | 2σ coverage, NaN = miss |
|---|---|---|---|---|---|---|---|
| G_approach_sweep15_bearing_only | 0 | 0.0% | - | 2.0% | 2.0% | 89.4% | 89.4% |
| G_approach_sweep15_size_prior | 15 | 6.7% | 3.2 | 4.5% | 12.0% | 77.8% | 72.4% |
| G_approach_sweep15_tight_prior | 0 | 0.0% | - | 0.0% | 0.0% | 95.1% | 95.1% |
| G_approach_sweep15_wrong_prior | 0 | 0.0% | - | 11.5% | 11.5% | 40.5% | 40.5% |
| G_approach_sweep15_wrong_tight | 0 | 0.0% | - | 67.0% | 67.0% | 0.3% | 0.3% |
| H_approach_hover_size_prior | 7 | 3.3% | 3.0 | 56.0% | 59.5% | 33.8% | 32.6% |
| H_approach_hover_tight_prior | 0 | 0.0% | - | 21.5% | 21.5% | 71.3% | 71.3% |
| I_approach_sweep5_size_prior | 9 | 4.1% | 3.0 | 4.5% | 9.0% | 75.1% | 71.9% |
| I_approach_sweep10_size_prior | 13 | 6.0% | 3.0 | 3.0% | 9.5% | 76.6% | 71.9% |
| I_approach_sweep30_size_prior | 18 | 8.3% | 2.5 | 4.0% | 13.0% | 78.8% | 72.1% |
| J_constant_velocity_sweep15_size_prior | 6 | 2.7% | 4.0 | 37.0% | 40.0% | 57.9% | 56.3% |
| J_gentle_turn_sweep15_size_prior | 6 | 2.7% | 4.0 | 46.5% | 49.5% | 50.2% | 48.9% |
| J_dash_sweep15_size_prior | 12 | 5.4% | 3.1 | 28.0% | 34.0% | 57.3% | 54.1% |
| J_jink_sweep15_size_prior | 24 | 10.7% | 3.7 | 88.0% | 100.0% | 4.6% | 4.1% |
| J_orbit_sweep15_size_prior | 20 | 9.9% | 0.4 | 7.5% | 17.5% | 76.1% | 68.5% |
| K_constant_velocity_sweep15_size_prior | 5 | 2.2% | 4.0 | 39.5% | 42.0% | 52.0% | 50.8% |
| K_gentle_turn_sweep15_size_prior | 5 | 2.2% | 4.0 | 35.0% | 37.5% | 56.2% | 55.0% |
| K_dash_sweep15_size_prior | 12 | 5.4% | 3.1 | 36.5% | 42.5% | 56.8% | 53.6% |
| K_jink_sweep15_size_prior | 16 | 7.3% | 3.3 | 91.0% | 99.0% | 5.0% | 4.6% |
| K_orbit_sweep15_size_prior | 20 | 9.9% | 0.4 | 7.5% | 17.5% | 77.0% | 69.3% |
