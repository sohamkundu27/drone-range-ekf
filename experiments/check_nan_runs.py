"""Count Monte Carlo runs whose filter state became NaN, and show how much that affects the metrics.

Why this exists: in a few runs, early on (range still very uncertain), a Gauss-Newton step of the
iterated update moves the drone estimate BEHIND the camera (Z < 0). The measurement model then
takes log(Z), which is NaN, and the NaN stays in the state for the rest of the run.
experiments/run_experiments.py computes its metrics with nanmean / nanmedian, so those time steps
are silently left out, and the final-step fractions count a NaN run as "not outside 3 sigma".
This script does not change any experiment; it reads runs/sim/*.npz and reports the effect.

    python experiments/check_nan_runs.py        # after experiments/run_experiments.py

Writes experiments/nan_runs.json and experiments/nan_runs.md.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "runs" / "sim"


def check(path):
    z = np.load(path)
    r, sig, t = z["r_est"], z["r_sigma"], z["t"]
    n_runs = r.shape[0]
    started = np.isfinite(r).any(axis=1)                       # filter was initialised at some point
    first = np.argmax(np.isfinite(r), axis=1)                  # first estimate of each run
    after_init = np.arange(r.shape[1])[None, :] >= first[:, None]
    bad_step = after_init & started[:, None] & ~(np.isfinite(r) & np.isfinite(sig))
    bad_run = bad_step.any(axis=1)
    first_bad_t = [float(t[np.argmax(row)]) for row in bad_step[bad_run]]

    # final-step divergence as reported (NaN counts as "not outside 3 sigma") and with NaN = diverged
    fin_rel = (r[:, -1] - z["r_true"][:, -1]) / z["r_true"][:, -1]
    fin_sig = sig[:, -1] / r[:, -1]
    outside_reported = float(np.nanmean(np.abs(fin_rel) > 3 * fin_sig))
    outside_nan_as_fail = float(np.mean((np.abs(fin_rel) > 3 * fin_sig) | ~np.isfinite(fin_rel)))
    return {"n_runs": n_runs, "nan_runs": int(bad_run.sum()),
            "frac_steps_after_init_excluded": float(bad_step.sum() / (after_init & started[:, None]).sum()),
            "median_time_of_first_nan_s": float(np.median(first_bad_t)) if first_bad_t else None,
            "final_outside_3sigma_as_reported": outside_reported,
            "final_outside_3sigma_nan_counted_as_diverged": outside_nan_as_fail}


def main():
    res = {p.stem: check(p) for p in sorted(SIM.glob("*.npz"))}
    total = sum(v["nan_runs"] for v in res.values())
    (ROOT / "experiments" / "nan_runs.json").write_text(json.dumps(res, indent=1))
    L = ["# Runs whose filter state became NaN (diagnostic, SIMULATION ONLY)", "",
         "Cause: an iterated-update step put the drone estimate behind the camera (Z < 0), so log(Z) = NaN, "
         "and the NaN persists. The metrics in results_sim.md leave those time steps out (nanmean), and its "
         "'final outside 3sigma' column counts a NaN run as not diverged. Not fixed; reported here.", "",
         f"Total: {total} runs over {len(res)} configurations of 200 runs.", "",
         "| config | NaN runs / 200 | time steps excluded | median first NaN [s] | final outside 3σ: reported | NaN counted as diverged |",
         "|---|---|---|---|---|---|"]
    for k, v in res.items():
        if v["nan_runs"] == 0:
            continue
        L.append(f"| {k} | {v['nan_runs']} | {100 * v['frac_steps_after_init_excluded']:.1f}% | "
                 f"{v['median_time_of_first_nan_s']:.1f} | {100 * v['final_outside_3sigma_as_reported']:.1f}% | "
                 f"{100 * v['final_outside_3sigma_nan_counted_as_diverged']:.1f}% |")
    L += ["", "Configurations not listed had no NaN runs (all bearing-only, tight-prior and wrong-prior rows)."]
    (ROOT / "experiments" / "nan_runs.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
