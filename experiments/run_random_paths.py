"""Random target paths vs the sweep-in-place observer: an add-on to experiments/run_sweep.py. SIMULATION ONLY.

    .venv/bin/python -I experiments/run_random_paths.py                  # regression check, runs, reports, figures
    .venv/bin/python -I experiments/run_random_paths.py --report-only    # reports and figures from saved runs

Question: how often does the filter work when the other drone can be anywhere and do anything?
Configs (200 runs each, seeds 1000..1199; each seed = one random path from sim/random_path.py, the same path in
all three configs, so they are paired):
  R_random_path_sweep15_size_prior    broad prior 0.35 +/- 0.15 m (true size drawn from it)
  R_random_path_sweep15_tight_prior   tight prior 0.35 +/- 0.03 m
  R_random_path_sweep15_wrong_prior   true size 0.60 m, prior 0.35 +/- 0.15 m
Everything else exactly as the sweep study: sweep +/-15 m / 12 s camera, q = 0.1, 3 iterations, measured detector
noise, default GPS/INS model, 10 Hz, 40 s; scored with run_experiments.summarize on samples with true range <= 200 m
(run_sweep.range_masked), NaN runs from check_nan_runs.check, extras from run_sweep.extra_metrics.

Before running, the script reruns G_approach_sweep15_size_prior and requires every field to match the saved result
(and sweep_summary.json: 12.71% RMS, 77.8% coverage, 15 NaN runs); otherwise it exits without running anything.

Outputs (the 40-config sweep_summary.* is not touched):
  runs/sim_sweep/R_*.npz / .json                       per-config runs and summary (gitignored)
  experiments/random_paths_summary.csv / .json         one row per config
  experiments/random_paths_runs.csv                    one row per seed: path parameters + per-run error per config
  experiments/sweep_tables.md, results_sweep.md, nan_runs_sweep.md : a "Random paths" section appended between
      markers (re-running this script replaces only that section; re-run it after run_sweep.py, which rewrites
      those files)
  figures/sweep/random_paths_examples.png, random_paths_breakdown.png
"""
import csv
import json
import math
import sys
import tempfile
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.run_sweep as rs  # noqa: E402
from experiments.check_nan_runs import check as nan_check  # noqa: E402
from experiments.mc import Q_FILTER, VARIANTS, run_one  # noqa: E402
from experiments.run_experiments import KEEP, N_RUNS, summarize, write_markdown  # noqa: E402
from sim.noise import DetectorNoise  # noqa: E402
from sim.random_path import make_random_scenario  # noqa: E402

EXP = ROOT / "experiments"
FIG = ROOT / "figures" / "sweep"
OUT = rs.OUT_NPZ
RP_VARIANTS = ("size_prior", "tight_prior", "wrong_prior")
CONFIGS = {f"R_random_path_sweep15_{v}": dict(group="R", target="random_path", camera="sweep", width=15.0,
                                               variant=v, q=Q_FILTER) for v in RP_VARIANTS}
# Diagnostic controls (NOT results configs): the same generator and seeds with parts of the motion switched off,
# to see which part of the motion breaks the filter. Reported separately, broad prior only.
CONTROLS = {"Rctrl_random_path_nosidestep_sweep15_size_prior": ("no sidesteps (accelerations kept)", {"p_sidestep": 0.0}),
            "Rctrl_random_path_constvel_sweep15_size_prior": ("constant velocity (no random accelerations or sidesteps; altitude-limit braking kept)",
                                                              {"a_max": 0.0, "p_sidestep": 0.0})}
VLABEL = {"size_prior": "broad prior 0.35 ± 0.15 m", "tight_prior": "tight prior 0.35 ± 0.03 m",
          "wrong_prior": "wrong prior (true 0.60 m)"}
PATH_KEYS = ("start_range_m", "start_azimuth_deg", "start_elevation_deg", "start_altitude_m", "initial_speed_m_s",
             "initial_vz_clipped", "n_segments", "n_sidesteps", "first_sidestep_t_s", "max_segment_accel_m_s2",
             "max_base_accel_m_s2", "max_total_accel_m_s2", "mean_speed_m_s", "max_speed_m_s", "min_altitude_m",
             "max_altitude_m", "frac_time_altitude_brake", "final_range_from_centre_m", "n_redraws",
             "min_range_to_camera_m", "frac_time_within_200m", "visible_at_t0")
ACC_EDGES = (0.0, 2.5, 2.75, 2.9, 3.0001)            # max segment acceleration buckets [m/s^2]
MARK = ("<!-- random_paths:start -->", "<!-- random_paths:end -->")
TITLE = "Random target paths (sweep ±15 m observer; SIMULATION ONLY)"


# ------------------------------------------------------------------ regression check
def nan_aware_diffs(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k} missing" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += nan_aware_diffs(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} length"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in nan_aware_diffs(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def regression_check(pool):
    name = "G_approach_sweep15_size_prior"
    saved = json.loads((OUT / f"{name}.json").read_text())
    row = {r["config"]: r for r in json.loads((EXP / "sweep_summary.json").read_text())["rows"]}[name]
    with tempfile.TemporaryDirectory() as tmp:
        rs.OUT_NPZ = Path(tmp)
        try:
            rs.run_config(name, rs.sweep_configs()[name], pool)
            new = json.loads((Path(tmp) / f"{name}.json").read_text())
        finally:
            rs.OUT_NPZ = OUT
    d = nan_aware_diffs(new, saved)
    headline = (round(100 * new["rms_rel_err_after10s"], 2), round(100 * new["coverage_2sigma"], 1),
                new["nan"]["nan_runs"])
    expected = (row["rms_pct_after10s"], row["coverage_2sigma_pct"], row["nan_runs"])
    ok = not d and headline == expected == (12.71, 77.8, 15)
    return ok, {"config": name, "fields_differing": d, "rerun_rms_cov_nan": headline,
                "sweep_summary_rms_cov_nan": expected, "match": ok}


# ------------------------------------------------------------------ runs
def _job(args):
    c, i = args
    seed = 1000 + i                                    # same seeds as every other config
    scn = make_random_scenario(seed, **c.get("path_kw", {}))   # path: default_rng([seed, 2]); noise: default_rng(seed)
    out = run_one(scn, VARIANTS[c["variant"]], seed, q=c["q"], det_noise=DetectorNoise())
    keep = {k: out[k] for k in KEEP}
    keep["S_est"] = out["x"][:, 6]
    keep["S_sigma"] = np.sqrt(out["P_diag"][:, 6])
    keep["S_true"] = out["S_true"]
    keep["t"] = out["t"]
    keep["min_range"] = float(np.min(np.linalg.norm(scn["drone_p"] - scn["cam_p"], axis=1)))
    keep["max_bearing_rate_deg_s"] = float(rs.bearing_rate_deg(scn).max())
    for k in PATH_KEYS:
        keep[f"path_{k}"] = float(scn["path"][k])
    return keep


def per_run(res):
    """Per-run error on the scored samples (true range <= 200 m), NaN steps left out as in summarize()."""
    t, r, sig, rt = res["t"], res["r_est"], res["r_sigma"], res["r_true"]
    rel = (r - rt) / rt
    in_range = rt <= rs.MAX_RANGE
    late = (t >= 10.0)[None, :] & in_range & np.isfinite(rel)
    n_late = late.sum(1)
    rms = np.sqrt(np.where(late, rel ** 2, 0).sum(1) / np.maximum(n_late, 1))
    rms[n_late == 0] = np.nan
    cov_m = (t >= 2.0)[None, :] & in_range & np.isfinite(r) & np.isfinite(sig)
    inside = cov_m & (np.abs(r - rt) <= 2 * sig)
    cov = inside.sum(1) / np.maximum(cov_m.sum(1), 1)
    cov[cov_m.sum(1) == 0] = np.nan
    started = np.isfinite(r).any(1)
    after_init = np.arange(r.shape[1])[None, :] >= np.argmax(np.isfinite(r), axis=1)[:, None]
    nan_run = (after_init & started[:, None] & ~(np.isfinite(r) & np.isfinite(sig))).any(1)
    return {"rms": rms, "coverage": cov, "n_scored_after10s": n_late, "nan_run": nan_run}


def run_config(name, c, pool):
    runs = pool.map(_job, [(c, i) for i in range(N_RUNS)])
    res = {k: np.array([r[k] for r in runs]) for k in runs[0] if k != "t"}
    res["t"] = runs[0]["t"]
    path = OUT / f"{name}.npz"
    np.savez_compressed(path, **res)
    return summarize_config(name, c, res, path)


def summarize_config(name, c, res, path):
    s = summarize(rs.range_masked(res))
    s["nis_dof"] = 3
    s["config"] = c
    s["extra"] = rs.extra_metrics(res)
    s["nan"] = nan_check(path)
    pr = per_run(res)
    ok = np.isfinite(pr["rms"])
    s["random_paths"] = {
        "runs_with_scored_samples_after10s": int(ok.sum()),
        "frac_scored_samples_within_200m": float(np.mean(res["r_true"] <= rs.MAX_RANGE)),
        "median_per_run_rms": float(np.nanmedian(pr["rms"])),
        "frac_runs_rms_over_25pct": float(np.mean(pr["rms"][ok] > 0.25)),
        "frac_runs_rms_over_50pct": float(np.mean(pr["rms"][ok] > 0.50)),
        "frac_runs_over_25pct_or_nan": float(np.mean((pr["rms"][ok] > 0.25) | pr["nan_run"][ok])),
        "frac_runs_over_50pct_or_nan": float(np.mean((pr["rms"][ok] > 0.50) | pr["nan_run"][ok])),
    }
    (OUT / f"{name}.json").write_text(json.dumps(s, indent=1))
    return s


def load_res(name):
    z = np.load(OUT / f"{name}.npz")
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ breakdown by path property
def groups(P):
    side = P["n_sidesteps"]
    acc = P["max_segment_accel_m_s2"]
    g = {"sidesteps": [("0 sidesteps", side == 0), ("1+ sidesteps", side >= 1)]}
    g["max segment accel"] = [(f"{lo:g}–{min(hi, 3):g} m/s²", (acc >= lo) & (acc < hi))
                              for lo, hi in zip(ACC_EDGES[:-1], ACC_EDGES[1:])]
    sr = P["start_range_m"]
    g["start range"] = [(f"{lo}–{hi} m", (sr >= lo) & (sr < hi)) for lo, hi in ((30, 80), (80, 130), (130, 200))]
    mr = P["min_range_to_camera_m"]
    g["closest approach to camera"] = [(f"{lo}–{hi} m", (mr >= lo) & (mr < hi))
                                       for lo, hi in ((0, 50), (50, 100), (100, 200))]
    return g


def group_stats(res, pr, mask):
    sub = {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == N_RUNS else v)
           for k, v in res.items()}
    sub["t"] = res["t"]
    s = summarize(rs.range_masked(sub))
    rms = pr["rms"][mask]
    ok = np.isfinite(rms)
    return {"n_runs": int(mask.sum()), "pooled_rms_pct": 100 * s["rms_rel_err_after10s"],
            "median_run_rms_pct": float(100 * np.nanmedian(rms)) if ok.any() else float("nan"),
            "frac_runs_over_25pct": float(np.mean(rms[ok] > 0.25)) if ok.any() else float("nan"),
            "coverage_pct": 100 * s["coverage_2sigma"], "nan_runs": int(pr["nan_run"][mask].sum())}


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ reports
def pct(x, d=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def replace_section(path, body):
    text = path.read_text() if path.exists() else ""
    block = MARK[0] + "\n" + body.rstrip("\n") + "\n" + MARK[1] + "\n"
    if MARK[0] in text and MARK[1] in text:
        a, b = text.index(MARK[0]), text.index(MARK[1]) + len(MARK[1])
        text = text[:a] + block.rstrip("\n") + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n" + block
    path.write_text(text if text.endswith("\n") else text + "\n")


def verdict(summ, bd, controls):
    """One sentence, every number from the data."""
    b = summ["R_random_path_sweep15_size_prior"]
    ns = controls["Rctrl_random_path_nosidestep_sweep15_size_prior"]
    cv = controls["Rctrl_random_path_constvel_sweep15_size_prior"]
    g0 = bd["sidesteps"]["size_prior"]["0 sidesteps"]
    biases = [100 * s["mean_rel_err_after10s"] for s in summ.values()]
    return (f"the target's sustained acceleration (turns and speed changes up to 3 m/s² for 4-10 s): the same 200 "
            f"random starts with no random accelerations or sidesteps (only the altitude-limit braking left) give "
            f"{pct(cv['rms_rel_err_after10s'])} RMS and {pct(cv['coverage_2sigma'], 0)} coverage, removing only the "
            f"sidesteps still leaves {pct(ns['rms_rel_err_after10s'])} RMS and {pct(ns['coverage_2sigma'], 0)} "
            f"coverage (vs {pct(b['rms_rel_err_after10s'])} / {pct(b['coverage_2sigma'], 0)} with them), the 0-sidestep "
            f"runs alone have {g0['median_run_rms_pct']:.0f}% median error, and every variant underestimates range "
            f"({min(biases):+.0f}% to {max(biases):+.0f}% bias), i.e. the constant-velocity filter (q = 0.1) reads "
            "unmodelled target acceleration as a nearer target; start range, closest approach and the size of the "
            "largest acceleration matter much less (breakdown above).")


def write_reports(summ, bd, corr, P, prs, regress, timing, controls):
    rows = []
    for n, s in summ.items():
        c, e, rp = s["config"], s["extra"], s["random_paths"]
        row = {"config": n, "group": "R", "target": "random path", "camera": "sweep", "width_m": 15.0,
               "variant": c["variant"], "q": c["q"]}
        row.update({f"rms_pct_{b}": (round(100 * s["by_range"][b]["rms_rel_err"], 2) if b in s["by_range"] else None)
                    for b in rs.BUCKETS})
        row.update({"rms_pct_after10s": round(100 * s["rms_rel_err_after10s"], 2),
                    "p68_pct_after10s": round(100 * e["p68_abs_rel_err_after10s"], 2),
                    "bias_pct_after10s": round(100 * s["mean_rel_err_after10s"], 2),
                    "reported_sigma_pct": round(100 * s["median_reported_rel_sigma_after10s"], 2),
                    "coverage_2sigma_pct": round(100 * s["coverage_2sigma"], 1),
                    "coverage_2sigma_nan_as_miss_pct": round(100 * e["coverage_2sigma_nan_as_miss"], 1),
                    "anees_range": round(s["anees_range"], 2), "anees_pos": round(s["anees_pos"], 2),
                    "mean_nis": round(s["mean_nis"], 2), "nis_dof": s["nis_dof"], "nan_runs": s["nan"]["nan_runs"],
                    "min_true_range_m": round(e["min_true_range_m"], 1),
                    "max_bearing_rate_deg_s": round(e["max_bearing_rate_deg_s"], 1),
                    "frac_samples_beyond_200m": round(e["frac_samples_beyond_200m"], 3),
                    "median_per_run_rms_pct": round(100 * rp["median_per_run_rms"], 2),
                    "frac_runs_rms_over_25pct": round(rp["frac_runs_rms_over_25pct"], 3),
                    "frac_runs_rms_over_50pct": round(rp["frac_runs_rms_over_50pct"], 3),
                    "frac_runs_over_25pct_or_nan": round(rp["frac_runs_over_25pct_or_nan"], 3),
                    "frac_runs_over_50pct_or_nan": round(rp["frac_runs_over_50pct_or_nan"], 3),
                    "runs_with_scored_samples_after10s": rp["runs_with_scored_samples_after10s"]})
        rows.append(row)
    with open(EXP / "random_paths_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (EXP / "random_paths_summary.json").write_text(json.dumps({
        "meta": {"note": "SIMULATION ONLY. Random target paths (sim/random_path.py) vs the sweep +/-15 m observer.",
                 "n_runs": N_RUNS, "seeds": "1000..1199 (each seed = one random path, same path in every config)",
                 "q": Q_FILTER, "max_scored_range_m": 200.0, "regression_check": regress, "timing": timing,
                 "per_run_rms": "RMS of (est - true) / true over t >= 10 s and true range <= 200 m, NaN steps left out",
                 "spearman_per_run_rms_vs_path_property": corr},
        "rows": rows, "breakdown": bd, "diagnostic_controls": controls}, indent=1, default=float) + "\n")

    # per-run CSV: what breaks the filter
    names = list(summ)
    with open(EXP / "random_paths_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + list(PATH_KEYS) + [f"{v}_{m}" for v in RP_VARIANTS for m in
                                                  ("rms_pct", "coverage_pct", "nan_run", "scored_samples_after10s")])
        for i in range(N_RUNS):
            out = [1000 + i] + [P[k][i] if k not in ("start_range_m",) else round(P[k][i], 2) for k in PATH_KEYS]
            for n in names:
                pr = prs[n]
                out += [round(100 * pr["rms"][i], 2) if np.isfinite(pr["rms"][i]) else "",
                        round(100 * pr["coverage"][i], 1) if np.isfinite(pr["coverage"][i]) else "",
                        int(pr["nan_run"][i]), int(pr["n_scored_after10s"][i])]
            w.writerow([round(x, 4) if isinstance(x, float) else x for x in out])

    # sweep_tables.md section
    T = [f"## {TITLE}", "",
         "Question: how often does the filter work when the other drone can be anywhere and do anything? Each seed "
         "1000..1199 draws one random path (sim/random_path.py: start 30-200 m from the sweep centre in any direction, "
         "-25..+50 m height, 0-15 m/s; constant random accelerations up to 3 m/s² for 4-10 s each; 30% chance per "
         "segment of a ±5 m sidestep in 1 s, peak 25 m/s²; 5-150 m altitude). The same 200 paths in every row. "
         "Sweep ±15 m / 12 s, q = 0.1, samples with true range <= 200 m "
         f"({100 * np.mean([s['random_paths']['frac_scored_samples_within_200m'] for s in summ.values()]):.0f}% of "
         "all samples; the drones fly away). Per-run RMS = RMS relative error of one run over t >= 10 s.", "",
         f"Regression check before the runs: G_approach_sweep15_size_prior rerun = {regress['rerun_rms_cov_nan'][0]}% RMS, "
         f"{regress['rerun_rms_cov_nan'][1]}% coverage, {regress['rerun_rms_cov_nan'][2]} NaN runs; every field "
         f"{'matches' if regress['match'] else 'DOES NOT match'} the saved result.", "",
         "| config | RMS err | 68th pct err | reported σ | 2σ coverage | 2σ cov. (NaN = miss) | range ANEES | bias | "
         "NaN runs | runs > 25% | runs > 50% | median run |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summ.items():
        rp = s["random_paths"]
        T.append(f"| {VLABEL[s['config']['variant']]} | {pct(s['rms_rel_err_after10s'])} | "
                 f"{pct(s['extra']['p68_abs_rel_err_after10s'])} | {pct(s['median_reported_rel_sigma_after10s'])} | "
                 f"{pct(s['coverage_2sigma'])} | {pct(s['extra']['coverage_2sigma_nan_as_miss'])} | "
                 f"{s['anees_range']:.2f} | {100 * s['mean_rel_err_after10s']:+.1f}% | {s['nan']['nan_runs']} | "
                 f"{pct(rp['frac_runs_rms_over_25pct'], 0)} | {pct(rp['frac_runs_rms_over_50pct'], 0)} | "
                 f"{pct(rp['median_per_run_rms'])} |")
    T += ["", "Runs > 25% / > 50%: share of the runs (with scored samples after 10 s) whose per-run RMS exceeds it. "
              "Counting NaN runs as failures too: " + "; ".join(
                  f"{VLABEL[s['config']['variant']].split(' (')[0]} {pct(s['random_paths']['frac_runs_over_25pct_or_nan'], 0)} / "
                  f"{pct(s['random_paths']['frac_runs_over_50pct_or_nan'], 0)}" for s in summ.values()) + ".",
          "", "### Breakdown by path property (per-run RMS; pooled = all scored samples of the group)", ""]
    for prop, gl in bd.items():
        T += [f"**{prop}**", "", "| group | runs | " + " | ".join(
            f"{VLABEL[v].split(' (')[0].split(' 0')[0]}: pooled RMS / median run / runs > 25% / coverage"
            for v in RP_VARIANTS) + " |", "|---|---|" + "---|" * len(RP_VARIANTS)]
        for lab in gl[RP_VARIANTS[0]]:
            cells = []
            for v in RP_VARIANTS:
                g = gl[v][lab]
                cells.append(f"{g['pooled_rms_pct']:.1f}% / {g['median_run_rms_pct']:.1f}% / "
                             f"{100 * g['frac_runs_over_25pct']:.0f}% / {g['coverage_pct']:.0f}%")
            T.append(f"| {lab} | {gl[RP_VARIANTS[0]][lab]['n_runs']} | " + " | ".join(cells) + " |")
        T.append("")
    T += ["### Diagnostic controls (not results; broad prior, same seeds and starts, parts of the motion switched off)",
          "", "| paths | RMS err | 2σ coverage | bias | median run | runs > 25% | NaN runs |", "|---|---|---|---|---|---|---|"]
    base_s = summ["R_random_path_sweep15_size_prior"]
    for lab, s in [("as reported (accelerations + sidesteps)", base_s)] + [(c["label"], c) for c in controls.values()]:
        rp_ = s["random_paths"]
        T.append(f"| {lab} | {pct(s['rms_rel_err_after10s'])} | {pct(s['coverage_2sigma'])} | "
                 f"{100 * s['mean_rel_err_after10s']:+.1f}% | {pct(rp_['median_per_run_rms'])} | "
                 f"{pct(rp_['frac_runs_rms_over_25pct'], 0)} | {s['nan']['nan_runs']} |")
    T += ["", "**What breaks it:** " + verdict(summ, bd, controls), ""]
    T += ["Spearman rank correlation of per-run RMS (broad prior) with each path property: " +
          ", ".join(f"{k} {v:+.2f}" for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]))) + "."]
    replace_section(EXP / "sweep_tables.md", "\n".join(T))

    # results_sweep.md section: the standard results table for the three configs
    with tempfile.TemporaryDirectory() as tmp:
        write_markdown(summ, Path(tmp) / "r.md")
        main_table = (Path(tmp) / "r.md").read_text().split("## RMS relative range error by true range")[0]
    body = main_table.splitlines()
    body[0] = f"## {TITLE}"
    body.insert(1, "\nSame columns as above; 200 random paths (sim/random_path.py), the same in each row. "
                   "Details and breakdown: experiments/sweep_tables.md, experiments/random_paths_summary.csv, "
                   "experiments/random_paths_runs.csv.")
    replace_section(EXP / "results_sweep.md", "\n".join(body))

    # nan_runs_sweep.md section
    L = [f"## {TITLE}", "", f"Total: {sum(s['nan']['nan_runs'] for s in summ.values())} runs over {len(summ)} "
         "configurations of 200 runs.", "",
         "| config | NaN runs / 200 | time steps excluded | median first NaN [s] | final outside 3σ: reported | "
         "NaN counted as diverged | 2σ coverage: reported | 2σ coverage, NaN = miss |",
         "|---|---|---|---|---|---|---|---|"]
    for n, s in summ.items():
        v = s["nan"]
        mt = "-" if v["median_time_of_first_nan_s"] is None else f"{v['median_time_of_first_nan_s']:.1f}"
        L.append(f"| {n} | {v['nan_runs']} | {100 * v['frac_steps_after_init_excluded']:.1f}% | {mt} | "
                 f"{100 * v['final_outside_3sigma_as_reported']:.1f}% | "
                 f"{100 * v['final_outside_3sigma_nan_counted_as_diverged']:.1f}% | "
                 f"{100 * s['coverage_2sigma']:.1f}% | {100 * s['extra']['coverage_2sigma_nan_as_miss']:.1f}% |")
    replace_section(EXP / "nan_runs_sweep.md", "\n".join(L))
    return "\n".join(T)


# ------------------------------------------------------------------ figures
def fig_examples(seeds=(1000, 1001, 1002, 1003, 1004)):
    from experiments.make_figures import COLOR, INK2, MUTED, VARIANTS as MV, plt
    cols = [COLOR[v] for v in MV]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(200 * np.cos(th), 200 * np.sin(th), color=MUTED, lw=0.8, ls="--")
    ax.text(0, 205, "200 m from the sweep centre", ha="center", fontsize=8, color=INK2)
    for s, c in zip(seeds, cols):
        scn = make_random_scenario(s)
        p, cp = scn["drone_p"], scn["cam_p"]
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1.8)
        ax.plot(*p[0, :2], "o", color=c, ms=7, mec="white", mew=1.2)
        ax.plot(*p[-1, :2], ">", color=c, ms=6)
        ax.plot(cp[:, 0], cp[:, 1], color=c, lw=4, alpha=0.5, solid_capstyle="butt")
        ax.annotate(f"seed {s}", p[0, :2], xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK2)
        r = np.linalg.norm(p - cp, axis=1)
        pth = scn["path"]
        ax2.plot(scn["t"], r, color=c, lw=1.8,
                 label=f"seed {s}: {pth['n_sidesteps']} sidestep{'s' * (pth['n_sidesteps'] != 1)}, "
                       f"start {pth['start_range_m']:.0f} m, mean {pth['mean_speed_m_s']:.1f} m/s")
    ax.plot([], [], "o", color=INK2, ms=6, label="start (in view at t = 0)")
    ax.plot([], [], ">", color=INK2, ms=6, label="end (t = 40 s)")
    ax.plot([], [], color=INK2, lw=4, alpha=0.5, label="observer sweep ±15 m (axis ⊥ initial line of sight)".replace("⊥", "perp. to"))
    ax.set_aspect("equal")
    lim = 420
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title("Top-down view (sweep centre at the origin)")
    ax.legend(fontsize=7.5, loc="lower left")
    ax2.axhline(200, color=MUTED, lw=1, ls="--")
    ax2.text(0.5, 204, "200 m: samples beyond are not scored", fontsize=8, color=INK2)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("true range observer → drone [m]")
    ax2.set_title("Range vs time")
    ax2.set_ylim(0, None)
    ax2.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Five random target paths (seeds 1000-1004, not hand-picked). SIMULATION ONLY.", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "random_paths_examples.png")
    plt.close(fig)


def fig_breakdown(bd, props=("sidesteps", "max segment accel", "start range", "closest approach to camera")):
    from experiments.make_figures import COLOR, INK2, plt
    fig, axes = plt.subplots(2, len(props), figsize=(15, 7.2), sharey="row")
    width = 0.26
    for j, prop in enumerate(props):
        labs = list(bd[prop][RP_VARIANTS[0]])
        x = np.arange(len(labs))
        for i, v in enumerate(RP_VARIANTS):
            g = [bd[prop][v][lab] for lab in labs]
            xs = x + (i - 1) * width
            med = [d["median_run_rms_pct"] for d in g]
            over = [100 * d["frac_runs_over_25pct"] for d in g]
            axes[0, j].bar(xs, med, width * 0.92, color=COLOR[v], label=VLABEL[v])
            axes[1, j].bar(xs, over, width * 0.92, color=COLOR[v], label=VLABEL[v])
            for xx, m, o in zip(xs, med, over):
                axes[0, j].text(xx, m + 0.4, f"{m:.0f}", ha="center", fontsize=7, color=INK2)
                axes[1, j].text(xx, o + 1, f"{o:.0f}", ha="center", fontsize=7, color=INK2)
        n = [bd[prop][RP_VARIANTS[0]][lab]["n_runs"] for lab in labs]
        for a in axes[:, j]:
            a.set_xticks(x, [f"{lab}\n(n = {k})" for lab, k in zip(labs, n)], fontsize=8)
            a.grid(axis="x", visible=False)
        axes[0, j].set_title(prop)
    axes[0, 0].set_ylabel("median per-run RMS error [%]")
    axes[1, 0].set_ylabel("runs with RMS error > 25% [%]")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("What breaks the filter on random paths: per-run range error by path property (sweep ±15 m, "
                 "q = 0.1, 200 paths, t ≥ 10 s, true range ≤ 200 m). SIMULATION ONLY.", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "random_paths_breakdown.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    report_only = "--report-only" in sys.argv
    timing = {"processes": cpu_count()}
    meta_path = EXP / "random_paths_summary.json"
    regress = json.loads(meta_path.read_text())["meta"]["regression_check"] if report_only else None
    summ = {}
    if not report_only:
        with Pool(cpu_count()) as pool:
            t0 = time.time()
            ok, regress = regression_check(pool)
            timing["regression_s"] = round(time.time() - t0, 1)
            print("REGRESSION", "MATCH" if ok else "MISMATCH", regress, flush=True)
            if not ok:
                sys.exit("regression check failed: nothing was run")
            todo = dict(CONFIGS)
            todo.update({n: dict(CONFIGS["R_random_path_sweep15_size_prior"], path_kw=kw) for n, (_, kw) in CONTROLS.items()})
            for n, c in todo.items():
                t0 = time.time()
                s_ = run_config(n, c, pool)
                if n in CONFIGS:
                    summ[n] = s_
                timing[n] = round(time.time() - t0, 1)
                print(f"{n:48s} {timing[n]:5.1f} s  RMS {100 * s_['rms_rel_err_after10s']:6.1f}%  cov "
                      f"{100 * s_['coverage_2sigma']:5.1f}%  NaN runs {s_['nan']['nan_runs']}", flush=True)
    else:
        timing = json.loads(meta_path.read_text())["meta"]["timing"]
        summ = {n: summarize_config(n, c, load_res(n), OUT / f"{n}.npz") for n, c in CONFIGS.items()}
    res = {n: load_res(n) for n in CONFIGS}
    first = res[next(iter(CONFIGS))]
    P = {k: first[f"path_{k}"] for k in PATH_KEYS}
    for n in CONFIGS:                                      # the same path for every config of a seed
        assert all(np.array_equal(res[n][f"path_{k}"], P[k], equal_nan=True) for k in PATH_KEYS), n
    prs = {n: per_run(res[n]) for n in CONFIGS}
    bd = {}
    for prop, gl in groups(P).items():
        bd[prop] = {c["variant"]: {lab: group_stats(res[n], prs[n], m) for lab, m in gl} for n, c in CONFIGS.items()}
    base = prs["R_random_path_sweep15_size_prior"]["rms"]
    corr = {k: round(spearman(P[k].astype(float), base), 3) for k in PATH_KEYS
            if k not in ("visible_at_t0", "n_redraws") and np.nanstd(P[k].astype(float)) > 0}
    controls = {}
    for n, (lab, kw) in CONTROLS.items():
        cs = summarize_config(n, dict(CONFIGS["R_random_path_sweep15_size_prior"], path_kw=kw), load_res(n), OUT / f"{n}.npz")
        controls[n] = {"label": lab, "path_kw": kw, **{k: cs[k] for k in ("rms_rel_err_after10s", "coverage_2sigma",
                       "mean_rel_err_after10s", "anees_range", "nan", "random_paths")}}
    print(write_reports(summ, bd, corr, P, prs, regress, timing, controls))
    fig_examples()
    fig_breakdown(bd)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
