# Sweep-in-place study: slide tables (SIMULATION ONLY)

200 runs per row, seeds 1000..1199, size_prior (0.35 ± 0.15 m), samples with true range <= 200 m. RMS / 68th pct / reported σ over t >= 10 s; coverage over t >= 2 s (ideal 95%). '-' = the target never enters that range bucket.

## Behaviour table (sweep ±15 m)

| behaviour | RMS err | 10-20 m | 20-40 m | 40-80 m | 80-120 m | 120-160 m | 160-200 m | 2σ coverage | actual (68th pct) vs reported σ | NaN runs | min range | max bearing rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inbound | 12.7% | 5.6% | 6.2% | 6.1% | 13.3% | 32.1% | 67.1% | 77.8% | 6.0% vs 3.7% | 15 | 9.6 m | 47.9°/s |
| crossing | 11.3% | - | - | - | 11.1% | 22.5% | 13.2% | 57.9% | 11.7% vs 4.8% | 6 | 116.1 m | 6.2°/s |
| gentle turn | 13.4% | - | - | - | - | 27.8% | 10.5% | 50.2% | 15.8% vs 4.9% | 6 | 127.5 m | 6.0°/s |
| dash | 18.7% | - | 11.4% | 11.5% | 18.2% | 31.3% | 46.2% | 57.3% | 9.8% vs 3.9% | 12 | 21.4 m | 41.3°/s |
| jink | 46.9% | 43.1% | 44.6% | 47.0% | 46.1% | 54.2% | 79.9% | 4.6% | 48.1% vs 4.4% | 24 | 8.0 m | 71.9°/s |
| orbit | 9.0% | - | - | 10.8% | 11.5% | - | - | 76.1% | 6.8% vs 4.1% | 20 | 70.3 m | 11.0°/s |

## Width table (inbound, size_prior)

| camera | RMS err | 10-20 m | 20-40 m | 40-80 m | 80-120 m | 120-160 m | 160-200 m | 2σ coverage | NaN runs |
|---|---|---|---|---|---|---|---|---|---|
| hover | 43.8% | 37.1% | 39.3% | 43.5% | 48.1% | 53.4% | 75.9% | 33.8% | 7 |
| sweep ±5 | 15.4% | 6.9% | 7.0% | 10.2% | 19.4% | 33.8% | 67.3% | 75.1% | 9 |
| sweep ±10 | 13.1% | 5.5% | 5.9% | 7.6% | 14.8% | 33.4% | 67.3% | 76.6% | 13 |
| sweep ±15 | 12.7% | 5.6% | 6.2% | 6.1% | 13.3% | 32.1% | 67.1% | 77.8% | 15 |
| sweep ±30 | 14.6% | 5.5% | 5.4% | 6.4% | 15.9% | 37.0% | 66.2% | 78.8% | 18 |

## G headline (sweep ±15 m, inbound, all variants)

| variant | RMS err | bias | actual (68th pct) vs reported σ | 2σ coverage | ANEES range | NaN runs |
|---|---|---|---|---|---|---|
| bearing_only | 307.2% | +27.1% | 9.0% vs 6.3% | 89.4% | 41.90 | 0 |
| size_prior | 12.7% | -0.7% | 6.0% vs 3.7% | 77.8% | 4.28 | 15 |
| tight_prior | 4.0% | +0.3% | 3.7% vs 3.7% | 95.1% | 1.06 | 0 |
| wrong_prior | 9.8% | -7.8% | 9.0% vs 3.3% | 40.5% | 13.01 | 0 |
| wrong_tight | 14.5% | -13.8% | 15.8% vs 3.2% | 0.3% | 29.31 | 0 |

## Full grid: target behaviour x size variant (sweep ±15 m, q = 0.1; groups G, J, L)

Each cell is one config of 200 runs, seeds 1000..1199 (paired across cells; jink phases identical in every jink cell). Variants: bearing_only = no size; size_prior = 0.35 ± 0.15 m; tight_prior = 0.35 ± 0.03 m; wrong_prior = true size 0.60 m, prior 0.35 ± 0.15 m; wrong_tight = true 0.60 m, prior 0.35 ± 0.03 m.

### RMS range error / true range, t ≥ 10 s

| behaviour | bearing_only | size_prior | tight_prior | wrong_prior | wrong_tight |
|---|---|---|---|---|---|
| inbound | 307.2% | 12.7% | 4.0% | 9.8% | 14.5% |
| crossing | 708.7% | 11.3% | 8.8% | 14.1% | 19.1% |
| gentle turn | 343.1% | 13.4% | 10.6% | 16.3% | 21.6% |
| dash | 387.3% | 18.7% | 7.1% | 12.2% | 16.8% |
| jink | 338.1% | 46.9% | 40.9% | 47.1% | 46.5% |
| orbit | 110.6% | 9.0% | 4.8% | 7.4% | 13.4% |

### 2σ coverage, t ≥ 2 s (ideal 95%)

| behaviour | bearing_only | size_prior | tight_prior | wrong_prior | wrong_tight |
|---|---|---|---|---|---|
| inbound | 89.4% | 77.8% | 95.1% | 40.5% | 0.3% |
| crossing | 90.6% | 57.9% | 69.8% | 17.4% | 0.0% |
| gentle turn | 82.1% | 50.2% | 57.7% | 8.3% | 0.0% |
| dash | 79.6% | 57.3% | 72.1% | 22.2% | 1.5% |
| jink | 10.2% | 4.6% | 4.5% | 0.4% | 0.0% |
| orbit | 55.8% | 76.1% | 89.5% | 59.5% | 0.9% |

### 2σ coverage with NaN steps counted as misses

| behaviour | bearing_only | size_prior | tight_prior | wrong_prior | wrong_tight |
|---|---|---|---|---|---|
| inbound | 89.4% | 72.4% | 95.1% | 40.5% | 0.3% |
| crossing | 90.6% | 56.3% | 69.8% | 17.4% | 0.0% |
| gentle turn | 82.1% | 48.9% | 57.7% | 8.3% | 0.0% |
| dash | 79.6% | 54.1% | 72.1% | 22.2% | 1.5% |
| jink | 10.2% | 4.1% | 4.5% | 0.4% | 0.0% |
| orbit | 55.8% | 68.5% | 89.5% | 59.5% | 0.9% |

### Range ANEES (1 dof; ideal 1)

| behaviour | bearing_only | size_prior | tight_prior | wrong_prior | wrong_tight |
|---|---|---|---|---|---|
| inbound | 41.90 | 4.28 | 1.06 | 13.01 | 29.31 |
| crossing | 50.70 | 7.16 | 3.42 | 19.50 | 38.76 |
| gentle turn | 77.47 | 8.92 | 4.59 | 22.78 | 47.65 |
| dash | 135.42 | 16.66 | 3.08 | 19.52 | 40.91 |
| jink | 299.23 | 343.90 | 233.43 | 470.14 | 488.50 |
| orbit | 230.92 | 9.25 | 1.53 | 4.41 | 21.81 |

### Bias (mean signed relative error), t ≥ 10 s

| behaviour | bearing_only | size_prior | tight_prior | wrong_prior | wrong_tight |
|---|---|---|---|---|---|
| inbound | +27.1% | -0.7% | +0.3% | -7.8% | -13.8% |
| crossing | +52.7% | -8.6% | -7.3% | -12.8% | -18.7% |
| gentle turn | +38.4% | -11.1% | -9.7% | -15.3% | -21.3% |
| dash | +35.8% | -3.5% | -2.5% | -10.5% | -16.3% |
| jink | -4.2% | -44.1% | -40.3% | -46.9% | -46.4% |
| orbit | +20.3% | -4.4% | -1.3% | -6.1% | -12.9% |

### NaN runs / 200

| behaviour | bearing_only | size_prior | tight_prior | wrong_prior | wrong_tight |
|---|---|---|---|---|---|
| inbound | 0 | 15 | 0 | 0 | 0 |
| crossing | 0 | 6 | 0 | 0 | 0 |
| gentle turn | 0 | 6 | 0 | 0 | 0 |
| dash | 0 | 12 | 0 | 0 | 0 |
| jink | 0 | 24 | 0 | 0 | 0 |
| orbit | 0 | 20 | 0 | 0 | 0 |

## Filter process noise q = 1.0 vs 0.1 (sweep ±15 m, size_prior)

| behaviour | RMS err q=0.1 | RMS err q=1.0 | 2σ coverage q=0.1 | 2σ coverage q=1.0 | NaN runs q=0.1 / 1.0 |
|---|---|---|---|---|---|
| crossing | 11.3% | 19.1% | 57.9% | 52.0% | 6 / 5 |
| gentle turn | 13.4% | 19.2% | 50.2% | 56.2% | 6 / 5 |
| dash | 18.7% | 24.0% | 57.3% | 56.8% | 12 / 12 |
| jink | 46.9% | 46.5% | 4.6% | 5.0% | 24 / 16 |
| orbit | 9.0% | 13.9% | 76.1% | 77.0% | 20 / 20 |

## 200 m scoring cut applied to the existing main-grid runs

Unchanged: 34 of 48 configs (groups fully unchanged: A, C, D, F). Changed (these targets fly beyond 200 m; results_sim.md is NOT modified, the cut is applied only to the sweep study):

* B_constant_velocity_straight_bearing_only: 17.0% of samples beyond 200 m; RMS 72.1% -> 78.1%
* B_constant_velocity_straight_size_prior: 17.0% of samples beyond 200 m; RMS 19.3% -> 18.8%
* B_constant_velocity_weaving_bearing_only: 3.2% of samples beyond 200 m; RMS 688.5% -> 703.8%
* B_constant_velocity_weaving_size_prior: 3.2% of samples beyond 200 m; RMS 10.8% -> 10.4%
* B_gentle_turn_lateral_bearing_only: 20.0% of samples beyond 200 m; RMS 35.9% -> 35.3%
* B_gentle_turn_lateral_size_prior: 20.0% of samples beyond 200 m; RMS 29.7% -> 27.5%
* B_gentle_turn_weaving_bearing_only: 24.9% of samples beyond 200 m; RMS 150.8% -> 161.7%
* B_gentle_turn_weaving_size_prior: 24.9% of samples beyond 200 m; RMS 13.3% -> 13.2%
* E_model_truth_weaving_40m: 3.0% of samples beyond 200 m; RMS 5.7% -> 5.3%
* E_model_truth_lateral_40m: 6.9% of samples beyond 200 m; RMS 39.2% -> 40.8%
* E_model_truth_weaving_80m: 4.0% of samples beyond 200 m; RMS 6.8% -> 6.3%
* E_model_truth_lateral_80m: 7.8% of samples beyond 200 m; RMS 23.3% -> 23.2%
* E_model_truth_weaving_150m: 11.6% of samples beyond 200 m; RMS 12.1% -> 11.8%
* E_model_truth_lateral_150m: 16.8% of samples beyond 200 m; RMS 31.0% -> 27.0%

<!-- random_paths:start -->
## Random target paths (sweep ±15 m observer; SIMULATION ONLY)

Question: how often does the filter work when the other drone can be anywhere and do anything? Each seed 1000..1199 draws one random path (sim/random_path.py: start 30-200 m from the sweep centre in any direction, -25..+50 m height, 0-15 m/s; constant random accelerations up to 3 m/s² for 4-10 s each; 30% chance per segment of a ±5 m sidestep in 1 s, peak 25 m/s²; 5-150 m altitude). The same 200 paths in every row. Sweep ±15 m / 12 s, q = 0.1, samples with true range <= 200 m (53% of all samples; the drones fly away). Per-run RMS = RMS relative error of one run over t >= 10 s.

Regression check before the runs: G_approach_sweep15_size_prior rerun = 12.71% RMS, 77.8% coverage, 15 NaN runs; every field matches the saved result.

| config | RMS err | 68th pct err | reported σ | 2σ coverage | 2σ cov. (NaN = miss) | range ANEES | bias | NaN runs | runs > 25% | runs > 50% | median run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| broad prior 0.35 ± 0.15 m | 26.6% | 34.4% | 4.5% | 21.0% | 20.2% | 63.57 | -22.4% | 11 | 54% | 3% | 26.3% |
| tight prior 0.35 ± 0.03 m | 23.4% | 31.9% | 4.3% | 25.5% | 25.5% | 45.44 | -19.8% | 0 | 38% | 0% | 23.0% |
| wrong prior (true 0.60 m) | 27.9% | 34.6% | 3.9% | 11.0% | 11.0% | 175.02 | -25.3% | 0 | 58% | 1% | 27.2% |

Runs > 25% / > 50%: share of the runs (with scored samples after 10 s) whose per-run RMS exceeds it. Counting NaN runs as failures too: broad prior 0.35 ± 0.15 m 54% / 3%; tight prior 0.35 ± 0.03 m 38% / 0%; wrong prior 58% / 1%.

### Breakdown by path property (per-run RMS; pooled = all scored samples of the group)

**sidesteps**

| group | runs | broad prior: pooled RMS / median run / runs > 25% / coverage | tight prior: pooled RMS / median run / runs > 25% / coverage | wrong prior: pooled RMS / median run / runs > 25% / coverage |
|---|---|---|---|---|
| 0 sidesteps | 29 | 24.0% / 21.4% / 45% / 27% | 20.1% / 20.6% / 24% / 33% | 22.4% / 22.9% / 36% / 18% |
| 1+ sidesteps | 171 | 27.0% / 26.4% / 55% / 20% | 24.0% / 23.1% / 41% / 24% | 28.8% / 27.7% / 62% / 10% |

**max segment accel**

| group | runs | broad prior: pooled RMS / median run / runs > 25% / coverage | tight prior: pooled RMS / median run / runs > 25% / coverage | wrong prior: pooled RMS / median run / runs > 25% / coverage |
|---|---|---|---|---|
| 0–2.5 m/s² | 57 | 24.4% / 21.7% / 39% / 22% | 22.2% / 21.9% / 31% / 28% | 24.9% / 23.4% / 45% / 14% |
| 2.5–2.75 m/s² | 48 | 30.2% / 31.9% / 74% / 16% | 25.6% / 25.0% / 54% / 22% | 29.9% / 31.5% / 68% / 10% |
| 2.75–2.9 m/s² | 49 | 26.8% / 28.3% / 62% / 26% | 23.4% / 24.6% / 42% / 29% | 30.6% / 27.9% / 68% / 11% |
| 2.9–3 m/s² | 46 | 26.2% / 23.9% / 46% / 18% | 23.3% / 22.9% / 28% / 22% | 27.2% / 26.1% / 54% / 9% |

**start range**

| group | runs | broad prior: pooled RMS / median run / runs > 25% / coverage | tight prior: pooled RMS / median run / runs > 25% / coverage | wrong prior: pooled RMS / median run / runs > 25% / coverage |
|---|---|---|---|---|
| 30–80 m | 49 | 28.0% / 28.7% / 54% / 20% | 25.0% / 25.6% / 52% / 25% | 30.3% / 25.9% / 52% / 19% |
| 80–130 m | 67 | 26.6% / 27.0% / 60% / 19% | 23.6% / 23.6% / 42% / 21% | 27.1% / 27.8% / 58% / 10% |
| 130–200 m | 84 | 24.9% / 23.9% / 45% / 25% | 21.6% / 20.0% / 22% / 31% | 26.4% / 26.5% / 64% / 6% |

**closest approach to camera**

| group | runs | broad prior: pooled RMS / median run / runs > 25% / coverage | tight prior: pooled RMS / median run / runs > 25% / coverage | wrong prior: pooled RMS / median run / runs > 25% / coverage |
|---|---|---|---|---|
| 0–50 m | 37 | 25.7% / 24.3% / 50% / 21% | 22.5% / 20.6% / 41% / 28% | 28.5% / 22.9% / 43% / 15% |
| 50–100 m | 69 | 26.6% / 26.7% / 58% / 22% | 23.5% / 23.5% / 38% / 25% | 27.5% / 27.6% / 60% / 14% |
| 100–200 m | 94 | 27.3% / 25.2% / 52% / 20% | 24.2% / 22.9% / 37% / 25% | 27.9% / 27.5% / 65% / 5% |

### Diagnostic controls (not results; broad prior, same seeds and starts, parts of the motion switched off)

| paths | RMS err | 2σ coverage | bias | median run | runs > 25% | NaN runs |
|---|---|---|---|---|---|---|
| as reported (accelerations + sidesteps) | 26.6% | 21.0% | -22.4% | 26.3% | 54% | 11 |
| no sidesteps (accelerations kept) | 23.4% | 24.7% | -18.7% | 20.9% | 34% | 11 |
| constant velocity (no random accelerations or sidesteps; altitude-limit braking kept) | 8.2% | 72.3% | -4.0% | 5.7% | 1% | 10 |

**What breaks it:** the target's sustained acceleration (turns and speed changes up to 3 m/s² for 4-10 s): the same 200 random starts with no random accelerations or sidesteps (only the altitude-limit braking left) give 8.2% RMS and 72% coverage, removing only the sidesteps still leaves 23.4% RMS and 25% coverage (vs 26.6% / 21% with them), the 0-sidestep runs alone have 21% median error, and every variant underestimates range (-25% to -20% bias), i.e. the constant-velocity filter (q = 0.1) reads unmodelled target acceleration as a nearer target; start range, closest approach and the size of the largest acceleration matter much less (breakdown above).

Spearman rank correlation of per-run RMS (broad prior) with each path property: first_sidestep_t_s -0.30, max_total_accel_m_s2 +0.17, n_sidesteps +0.17, start_altitude_m +0.15, frac_time_within_200m -0.14, max_speed_m_s +0.14, start_elevation_deg +0.13, mean_speed_m_s +0.12, final_range_from_centre_m +0.11, max_altitude_m +0.10, min_range_to_camera_m +0.08, max_segment_accel_m_s2 +0.07, max_base_accel_m_s2 +0.07, frac_time_altitude_brake +0.07, initial_vz_clipped +0.06, start_range_m -0.05, n_segments -0.03, initial_speed_m_s -0.01, min_altitude_m +0.01, start_azimuth_deg -0.01.
<!-- random_paths:end -->
