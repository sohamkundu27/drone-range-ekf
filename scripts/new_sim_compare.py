"""Compare the simulation with the OLD vs NEW detector noise (writes experiments/new_results/COMPARISON.md/.json).

    python scripts/new_sim_compare.py

Old: experiments/results_sim.json, nan_runs.json, runs/sim/*.npz (committed / original run).
New: experiments/new_results/ and runs/sim_shadow_newnoise/runs/sim/*.npz (scripts/new_sim_shadow.sh).
Also reports, per config, the ANEES-range without its single worst run, because ANEES is a mean over
runs and one run whose NEES explodes (but does not become NaN) can move it by an order of magnitude.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "experiments" / "new_results"
NPZ = {"old": ROOT / "runs" / "sim", "new": ROOT / "runs" / "sim_shadow_newnoise" / "runs" / "sim"}


def worst_run_check(tag, cfg):
    p = NPZ[tag] / f"{cfg}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    ne = d["nees_range"][:, d["t"] >= 2]
    with np.errstate(all="ignore"):
        m = np.nanmean(ne, axis=1)
    m = m[np.isfinite(m)]
    s = np.sort(m)
    return {"median_run_mean_nees": float(np.median(m)), "worst_run_mean_nees": float(s[-1]),
            "mean_of_run_means_without_worst": float(s[:-1].mean())}


def noise_table():
    o = json.loads((ROOT / "experiments/detector_noise.json").read_text())
    n = json.loads((NEW / "detector_noise_used.json").read_text())
    ob = {b["w_lo"]: b for b in o["all_bins"]}
    rows = []
    for b in n["all_bins"]:
        a = ob.get(b["w_lo"], {})
        if "sigma_center_px" in b and "sigma_center_px" in a:
            hi = "inf" if b["w_hi"] >= 10_000 else b["w_hi"]
            rows.append(f"| {b['w_lo']}-{hi} | {a['n_matched']} → {b['n_matched']} | {a['recall']:.3f} → {b['recall']:.3f} | "
                        f"{a['sigma_center_px']:.2f} → {b['sigma_center_px']:.2f} | {a['sigma_logw']:.3f} → {b['sigma_logw']:.3f} | "
                        f"{a['bias_logw']:+.3f} → {b['bias_logw']:+.3f} |")
    return o, n, rows


def main():
    old = json.loads((ROOT / "experiments/results_sim.json").read_text())["results"]
    new = json.loads((NEW / "results_sim.json").read_text())["results"]
    onan = json.loads((ROOT / "experiments/nan_runs.json").read_text())
    nnan = json.loads((NEW / "nan_runs.json").read_text())
    o, n, nrows = noise_table()
    L = ["# Simulation with the NEW detector noise vs the OLD (SIMULATION ONLY)", "",
         "Same sim/EKF code (byte-identical copies, see `detfly_retrain/sim_shadow_check.md`), same seeds; only "
         "`detector_noise.json` differs (new = box noise of the retrained s on held-out DUT test drones, "
         "`detector/measure_noise.py`). The known NaN behaviour is unchanged and not fixed.", "",
         "## Did the measured noise change?", "",
         f"Matched DUT test drones: {o['n_matched']} → {n['n_matched']} of {n['n_gt']}.", "",
         "| true width px | n matched | recall | centre σ px | log-width σ | log-width bias |", "|---|---|---|---|---|---|"] + nrows
    L += ["", "**The noise barely changed**: every robust σ moved by ≤ ~0.1 px (centre) or ≤ 0.011 (log width), "
          "recall by ≤ 0.01. Differences in the sim below are therefore mostly Monte-Carlo sensitivity, not a "
          "better or worse detector model.", "",
          f"## NaN runs: {sum(v['nan_runs'] for v in onan.values())} → {sum(v['nan_runs'] for v in nnan.values())} "
          "(of 9,600)", "",
          "## Per configuration (old → new)", "",
          "| config | RMS rel err | 2σ coverage | ANEES range | ANEES w/o worst run (old → new) | NaN runs |",
          "|---|---|---|---|---|---|"]
    js = {}
    for k in old:
        a, b = old[k], new.get(k, {})
        wo, wn = worst_run_check("old", k), worst_run_check("new", k)
        flag = ""
        if b and abs(b["anees_range"] - a["anees_range"]) > max(1.0, 0.5 * a["anees_range"]):
            flag = " ⚠"
        L.append(f"| {k} | {100 * a['rms_rel_err_after10s']:.1f}% → {100 * b.get('rms_rel_err_after10s', float('nan')):.1f}% | "
                 f"{100 * a['coverage_2sigma']:.1f}% → {100 * b.get('coverage_2sigma', float('nan')):.1f}% | "
                 f"{a['anees_range']:.2f} → {b.get('anees_range', float('nan')):.2f}{flag} | "
                 + (f"{wo['mean_of_run_means_without_worst']:.2f} → {wn['mean_of_run_means_without_worst']:.2f} "
                    f"(worst run {wo['worst_run_mean_nees']:.0f} → {wn['worst_run_mean_nees']:.0f})" if wo and wn else "n/a")
                 + f" | {onan.get(k, {}).get('nan_runs', 0)} → {nnan.get(k, {}).get('nan_runs', 0)} |")
        js[k] = {"old": {m: a[m] for m in ("rms_rel_err_after10s", "coverage_2sigma", "anees_range")},
                 "new": {m: b.get(m) for m in ("rms_rel_err_after10s", "coverage_2sigma", "anees_range")},
                 "worst_run_old": wo, "worst_run_new": wn}
    L += ["", "⚠ = ANEES changed by more than max(1, 50 %). Check the 'w/o worst run' column: where it barely moves, "
          "the change comes from a single run whose NEES exploded (finite, so it is not counted as a NaN run), "
          "not from a shift of the typical run. 'w/o worst run' is the mean of per-run mean NEES over the finite "
          "runs minus the worst one (an approximation of ANEES: results_sim.json averages over time steps)."]
    (NEW / "COMPARISON.md").write_text("\n".join(L) + "\n")
    (NEW / "COMPARISON.json").write_text(json.dumps(js, indent=1) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
