# Check: the shadow sim runner reproduces the committed simulation results

`scripts/new_sim_shadow.sh` runs byte-identical copies of `ekf/`, `sim/` and `experiments/*.py` in
`runs/sim_shadow_<tag>/`, so new results never overwrite `experiments/` or `figures/`.

Check run on Sat 2026-11-21 ~02:30 CST with the OLD noise (`experiments/detector_noise.json`):
`results_sim.json`, `results_sim.md`, `nan_runs.json` and all five figures (`rmse_vs_range`, `nees_time`,
`range_error_envelopes`, `uncertainty_vs_motion`, `consistency_model_truth`) were **byte-identical** to
the committed files. Any difference in `experiments/new_results/` therefore comes only from the new
detector-noise JSON.
