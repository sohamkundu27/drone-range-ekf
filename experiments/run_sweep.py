"""Sweep-in-place study: an observer that sweeps side to side around a fixed centre (or hovers), against
the existing and four harder target behaviours. SIMULATION ONLY.

    nohup .venv/bin/python -I experiments/run_sweep.py > logs/run_sweep.log 2>&1 &     # resumable
    .venv/bin/python -I experiments/run_sweep.py --report-only                         # tables from saved runs

Configs (<group>_<target>_<camera>_<variant>; 200 runs each, seeds 1000..1199 as in run_experiments.py):
  G  sweep +/-15 x approach_recede ("inbound") x all 5 variants
  H  hover x inbound x size_prior, tight_prior
  I  sweep +/-5, +/-10, +/-30 x inbound x size_prior
  J  sweep +/-15 x constant_velocity ("crossing"), gentle_turn, dash, jink, orbit x size_prior
  K  the J configs with filter q = 1.0 instead of 0.1 (per-config q; Q_FILTER is not edited)
  L  sweep +/-15 x the 5 J targets x the 4 other variants (bearing_only, tight_prior, wrong_prior, wrong_tight),
     q = 0.1; with G and J this completes the 6 targets x 5 variants grid (see grid_name)
Everything else as run_experiments.py: run_one (experiments/mc.py) with its default GPS/INS model,
the measured detector noise, 3 iterations, 10 Hz, 40 s. The jink phase comes from a SEPARATE generator,
default_rng([seed, 1]), so detector noise, pose drift and true-size draws are identical to the other configs.

Scoring: run_experiments.summarize (unchanged) on the runs with every sample whose true range is above
200 m set to NaN (dash, crossing and gentle turn go beyond 200 m). NaN-run diagnostics use the unmasked
runs, with experiments/check_nan_runs.check. Extra per config: 68th percentile of |range error| (t >= 10 s),
2-sigma coverage with NaN steps counted as misses, minimum true observer-target range, maximum bearing rate.

Outputs: runs/sim_sweep/<config>.npz and <config>.json (one result per config; existing ones are skipped),
experiments/results_sweep.md, experiments/nan_runs_sweep.md, experiments/sweep_summary.{csv,json},
experiments/sweep_tables.md (behaviour and width tables).
"""
import csv
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.check_nan_runs import check as nan_check  # noqa: E402
from experiments.mc import Q_FILTER, VARIANTS, run_one  # noqa: E402
from experiments.run_experiments import KEEP, N_RUNS, RANGE_BINS, configs as main_configs, summarize, write_markdown  # noqa: E402
from sim.noise import DetectorNoise  # noqa: E402
from sim.trajectories import make_scenario  # noqa: E402

OUT_NPZ = ROOT / "runs" / "sim_sweep"
EXP = ROOT / "experiments"
MAX_RANGE = 200.0 + 1e-6          # samples beyond 200 m are not scored (1 um tolerance: inbound starts at 200 + 3e-14 m)
BUCKETS = [f"{a}-{b}m" for a, b in zip(RANGE_BINS[:-1], RANGE_BINS[1:])]
TARGET_LABEL = {"approach_recede": "inbound", "constant_velocity": "crossing", "gentle_turn": "gentle turn",
                "dash": "dash", "jink": "jink", "orbit": "orbit"}
TOKEN = {"approach_recede": "approach"}      # existing naming: A_approach_..., B_constant_velocity_...


def cam_token(camera, A):
    return "hover" if camera == "hover" else f"sweep{A:g}"


def cfg(group, target, camera, variant, A=15.0, q=Q_FILTER):
    name = f"{group}_{TOKEN.get(target, target)}_{cam_token(camera, A)}_{variant}"
    return name, dict(group=group, target=target, camera=camera, width=(0.0 if camera == "hover" else A),
                      variant=variant, q=q)


def sweep_configs():
    c = [cfg("G", "approach_recede", "sweep", v) for v in VARIANTS]
    c += [cfg("H", "approach_recede", "hover", v) for v in ("size_prior", "tight_prior")]
    c += [cfg("I", "approach_recede", "sweep", "size_prior", A=A) for A in (5.0, 10.0, 30.0)]
    targets = ("constant_velocity", "gentle_turn", "dash", "jink", "orbit")
    c += [cfg("J", t, "sweep", "size_prior") for t in targets]
    c += [cfg("K", t, "sweep", "size_prior", q=1.0) for t in targets]
    c += [cfg("L", t, "sweep", v) for t in targets for v in VARIANTS if v != "size_prior"]
    out = dict(c)
    assert len(out) == len(c) == 40
    return out


GRID_TARGETS = ("approach_recede", "constant_velocity", "gentle_turn", "dash", "jink", "orbit")


def grid_name(target, variant):
    """Config of the 6 targets x 5 variants grid on the sweep +/-15 m camera, q = 0.1 (groups G, J and L)."""
    if target == "approach_recede":
        return f"G_approach_sweep15_{variant}"
    return f"{'J' if variant == 'size_prior' else 'L'}_{target}_sweep15_{variant}"


# ------------------------------------------------------------------ one run
def bearing_rate_deg(scn):
    """Instantaneous line-of-sight rotation rate |r x v_rel| / |r|^2 [deg/s], r = drone - camera."""
    r = scn["drone_p"] - scn["cam_p"]
    v = scn["drone_v"] - scn["cam_v"]
    return np.degrees(np.linalg.norm(np.cross(r, v), axis=1) / np.sum(r * r, axis=1))


def _job(args):
    c, i = args
    seed = 1000 + i                                   # same seeds as run_experiments.py -> paired comparisons
    phi = float(np.random.default_rng([seed, 1]).uniform(0.0, 2 * np.pi)) if c["target"] == "jink" else np.nan
    scn = make_scenario(c["target"], c["camera"], sweep_A=c["width"] or 15.0,
                        jink_phase=0.0 if np.isnan(phi) else phi)
    out = run_one(scn, VARIANTS[c["variant"]], seed, q=c["q"], det_noise=DetectorNoise())
    keep = {k: out[k] for k in KEEP}
    keep["S_est"] = out["x"][:, 6]
    keep["S_sigma"] = np.sqrt(out["P_diag"][:, 6])
    keep["S_true"] = out["S_true"]
    keep["t"] = out["t"]
    keep["phi"] = phi
    keep["min_range"] = float(np.min(np.linalg.norm(scn["drone_p"] - scn["cam_p"], axis=1)))
    keep["max_bearing_rate_deg_s"] = float(bearing_rate_deg(scn).max())
    return keep


# ------------------------------------------------------------------ scoring
def range_masked(res):
    """Copy of res with every sample whose TRUE range exceeds 200 m set to NaN (not scored)."""
    far = res["r_true"] > MAX_RANGE
    m = dict(res)
    for k in ("r_true", "r_est", "r_sigma", "nees_range", "nees_pos", "nis"):
        a = np.array(res[k], dtype=float)
        a[far] = np.nan
        m[k] = a
    return m


def extra_metrics(res):
    """Metrics not in summarize(): 68th pct |rel err|, coverage with NaN steps as misses, geometry."""
    t, r, sig, rt = res["t"], res["r_est"], res["r_sigma"], res["r_true"]
    rel = (r - rt) / rt
    after10, after2 = t >= 10.0, t >= 2.0
    in_range = np.isfinite(rt) & (rt <= MAX_RANGE)
    started = np.isfinite(r).any(axis=1)
    after_init = np.arange(r.shape[1])[None, :] >= np.argmax(np.isfinite(r), axis=1)[:, None]
    scored = after2[None, :] & in_range & after_init & started[:, None]
    inside = np.isfinite(r) & np.isfinite(sig) & (np.abs(r - rt) <= 2 * sig)
    return {"p68_abs_rel_err_after10s": float(np.nanpercentile(np.abs(rel[:, after10]), 68)),
            "coverage_2sigma_nan_as_miss": float((inside & scored).sum() / scored.sum()),
            "min_true_range_m": float(np.min(res["min_range"])),
            "max_bearing_rate_deg_s": float(np.max(res["max_bearing_rate_deg_s"])),
            "frac_samples_beyond_200m": float(np.mean(res["r_true"] > MAX_RANGE))}


def run_config(name, c, pool):
    runs = pool.map(_job, [(c, i) for i in range(N_RUNS)])
    res = {k: np.array([r[k] for r in runs]) for k in runs[0] if k != "t"}
    res["t"] = runs[0]["t"]
    path = OUT_NPZ / f"{name}.npz"
    np.savez_compressed(path, **res)
    s = summarize(range_masked(res))
    s["nis_dof"] = 2 if c["variant"] == "bearing_only" else 3
    s["config"] = c
    s["extra"] = extra_metrics(res)
    s["nan"] = nan_check(path)
    if c["target"] == "jink":
        s["jink_phase_per_run"] = res["phi"].tolist()
    (OUT_NPZ / f"{name}.json").write_text(json.dumps(s, indent=1))
    return s


# ------------------------------------------------------------------ check: the 200 m cut on existing configs
def range_cut_effect_on_existing():
    """Apply the same 200 m cut to the saved main-grid runs (runs/sim) and list configs whose summary changes."""
    old = json.loads((EXP / "results_sim.json").read_text())["results"]
    changed, unchanged = {}, []
    for name in main_configs():
        p = ROOT / "runs" / "sim" / f"{name}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        res = {k: z[k] for k in z.files}
        s = summarize(range_masked(res))
        s["nis_dof"] = old[name]["nis_dof"]
        diff = {k: (old[name][k], s[k]) for k in ("rms_rel_err_after10s", "coverage_2sigma", "anees_range")
                if old[name][k] != s[k]}
        all_diff = [k for k in old[name] if k != "config" and old[name][k] != s[k]]
        if all_diff:
            changed[name] = {"fields": all_diff, "headline": diff,
                             "frac_samples_beyond_200m": float(np.mean(res["r_true"] > MAX_RANGE))}
        else:
            unchanged.append(name)
    return {"unchanged": unchanged, "changed": changed}


# ------------------------------------------------------------------ reports
def fmt_pct(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def bucket_rms(s, b):
    return s["by_range"][b]["rms_rel_err"] if b in s["by_range"] else None


def write_reports(summ, cut_check, timing):
    # results_sweep.md: the results_sim.md tables, produced by write_markdown itself
    md = EXP / "results_sweep.md"
    write_markdown(summ, md)
    lines = md.read_text().splitlines()
    lines[0] = "# Sweep-in-place study (200 Monte Carlo runs each; SIMULATION ONLY)"
    lines[1:1] = ["", "Generated by experiments/run_sweep.py. Same columns and definitions as experiments/results_sim.md, "
                      "but only samples with true range <= 200 m are scored. The range-bin table below lists grid A of the "
                      "main study only, so it is empty here; per-bucket RMS for these configs is in "
                      "experiments/sweep_summary.csv and experiments/sweep_tables.md."]
    md.write_text("\n".join(lines) + "\n")

    # nan_runs_sweep.md: nan_runs.md format + coverage with NaN steps counted as misses
    total = sum(s["nan"]["nan_runs"] for s in summ.values())
    L = ["# Runs whose filter state became NaN (diagnostic, SIMULATION ONLY) - sweep study", "",
         "Cause: an iterated-update step put the drone estimate behind the camera (Z < 0), so log(Z) = NaN, and the "
         "NaN persists. The metrics in results_sweep.md leave those time steps out (nanmean), and its 'final outside "
         "3sigma' column counts a NaN run as not diverged. Not fixed; reported here. The last two columns are added "
         "for this study: 2-sigma coverage as reported (NaN steps skipped) and with every NaN step after "
         "initialisation counted as a miss (t >= 2 s, true range <= 200 m).", "",
         f"Total: {total} runs over {len(summ)} configurations of 200 runs.", "",
         "| config | NaN runs / 200 | time steps excluded | median first NaN [s] | final outside 3σ: reported | "
         "NaN counted as diverged | 2σ coverage: reported | 2σ coverage, NaN = miss |",
         "|---|---|---|---|---|---|---|---|"]
    for k, s in summ.items():
        v = s["nan"]
        mt = "-" if v["median_time_of_first_nan_s"] is None else f"{v['median_time_of_first_nan_s']:.1f}"
        L.append(f"| {k} | {v['nan_runs']} | {100 * v['frac_steps_after_init_excluded']:.1f}% | {mt} | "
                 f"{100 * v['final_outside_3sigma_as_reported']:.1f}% | {100 * v['final_outside_3sigma_nan_counted_as_diverged']:.1f}% | "
                 f"{100 * s['coverage_2sigma']:.1f}% | {100 * s['extra']['coverage_2sigma_nan_as_miss']:.1f}% |")
    (EXP / "nan_runs_sweep.md").write_text("\n".join(L) + "\n")

    # sweep_summary.csv / .json
    rows = []
    for k, s in summ.items():
        c, e = s["config"], s["extra"]
        row = {"config": k, "group": c["group"], "target": TARGET_LABEL[c["target"]], "camera": c["camera"],
               "width_m": c["width"], "variant": c["variant"], "q": c["q"]}
        row.update({f"rms_pct_{b}": (None if bucket_rms(s, b) is None else round(100 * bucket_rms(s, b), 2)) for b in BUCKETS})
        row.update({"rms_pct_after10s": round(100 * s["rms_rel_err_after10s"], 2),
                    "p68_pct_after10s": round(100 * e["p68_abs_rel_err_after10s"], 2),
                    "bias_pct_after10s": round(100 * s["mean_rel_err_after10s"], 2),
                    "reported_sigma_pct": round(100 * s["median_reported_rel_sigma_after10s"], 2),
                    "coverage_2sigma_pct": round(100 * s["coverage_2sigma"], 1),
                    "coverage_2sigma_nan_as_miss_pct": round(100 * e["coverage_2sigma_nan_as_miss"], 1),
                    "anees_range": round(s["anees_range"], 2), "nan_runs": s["nan"]["nan_runs"],
                    "min_true_range_m": round(e["min_true_range_m"], 1),
                    "max_bearing_rate_deg_s": round(e["max_bearing_rate_deg_s"], 1),
                    "frac_samples_beyond_200m": round(e["frac_samples_beyond_200m"], 3),
                    "anees_pos": round(s["anees_pos"], 2), "mean_nis": round(s["mean_nis"], 2),
                    "nis_dof": s["nis_dof"]})
        rows.append(row)
    with open(EXP / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (EXP / "sweep_summary.json").write_text(json.dumps({
        "meta": {"n_runs": N_RUNS, "seeds": "1000..1199", "q_default": Q_FILTER, "max_scored_range_m": 200.0,
                 "detector_noise_source": DetectorNoise().source, "timing": timing,
                 "range_cut_check_on_existing_configs": cut_check,
                 "note": "Simulation only."},
        "rows": rows,
        "jink_phase_rad_per_seed": {n: dict(zip(range(1000, 1000 + N_RUNS), np.round(s["jink_phase_per_run"], 6).tolist()))
                                    for n, s in summ.items() if "jink_phase_per_run" in s}}, indent=1) + "\n")

    # behaviour and width tables
    T = ["# Sweep-in-place study: slide tables (SIMULATION ONLY)", "",
         "200 runs per row, seeds 1000..1199, size_prior (0.35 ± 0.15 m), samples with true range <= 200 m. "
         "RMS / 68th pct / reported σ over t >= 10 s; coverage over t >= 2 s (ideal 95%). '-' = the target never "
         "enters that range bucket.", "", "## Behaviour table (sweep ±15 m)", "",
         "| behaviour | RMS err | " + " | ".join(f"{b[:-1]} m" for b in BUCKETS) +
         " | 2σ coverage | actual (68th pct) vs reported σ | NaN runs | min range | max bearing rate |",
         "|---|---|" + "---|" * len(BUCKETS) + "---|---|---|---|---|"]
    beh = [("inbound", "G_approach_sweep15_size_prior")] + [
        (TARGET_LABEL[t], f"J_{t}_sweep15_size_prior") for t in ("constant_velocity", "gentle_turn", "dash", "jink", "orbit")]
    for lab, n in beh:
        s = summ[n]
        e = s["extra"]
        T.append(f"| {lab} | {fmt_pct(s['rms_rel_err_after10s'])} | " + " | ".join(fmt_pct(bucket_rms(s, b)) for b in BUCKETS) +
                 f" | {fmt_pct(s['coverage_2sigma'])} | {fmt_pct(e['p68_abs_rel_err_after10s'])} vs "
                 f"{fmt_pct(s['median_reported_rel_sigma_after10s'])} | {s['nan']['nan_runs']} | "
                 f"{e['min_true_range_m']:.1f} m | {e['max_bearing_rate_deg_s']:.1f}°/s |")
    T += ["", "## Width table (inbound, size_prior)", "",
          "| camera | RMS err | " + " | ".join(f"{b[:-1]} m" for b in BUCKETS) + " | 2σ coverage | NaN runs |",
          "|---|---|" + "---|" * len(BUCKETS) + "---|---|"]
    wid = [("hover", "H_approach_hover_size_prior")] + [
        (f"sweep ±{A}", f"{'G' if A == 15 else 'I'}_approach_sweep{A}_size_prior") for A in (5, 10, 15, 30)]
    for lab, n in wid:
        s = summ[n]
        T.append(f"| {lab} | {fmt_pct(s['rms_rel_err_after10s'])} | " + " | ".join(fmt_pct(bucket_rms(s, b)) for b in BUCKETS) +
                 f" | {fmt_pct(s['coverage_2sigma'])} | {s['nan']['nan_runs']} |")
    T += ["", "## G headline (sweep ±15 m, inbound, all variants)", "",
          "| variant | RMS err | bias | actual (68th pct) vs reported σ | 2σ coverage | ANEES range | NaN runs |",
          "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        s = summ[f"G_approach_sweep15_{v}"]
        T.append(f"| {v} | {fmt_pct(s['rms_rel_err_after10s'])} | {100 * s['mean_rel_err_after10s']:+.1f}% | "
                 f"{fmt_pct(s['extra']['p68_abs_rel_err_after10s'])} vs {fmt_pct(s['median_reported_rel_sigma_after10s'])} | "
                 f"{fmt_pct(s['coverage_2sigma'])} | {s['anees_range']:.2f} | {s['nan']['nan_runs']} |")
    if all(grid_name(t, v) in summ for t in GRID_TARGETS for v in VARIANTS):
        T += ["", "## Full grid: target behaviour x size variant (sweep ±15 m, q = 0.1; groups G, J, L)", "",
              "Each cell is one config of 200 runs, seeds 1000..1199 (paired across cells; jink phases identical in "
              "every jink cell). Variants: bearing_only = no size; size_prior = 0.35 ± 0.15 m; tight_prior = "
              "0.35 ± 0.03 m; wrong_prior = true size 0.60 m, prior 0.35 ± 0.15 m; wrong_tight = true 0.60 m, prior "
              "0.35 ± 0.03 m."]
        grids = [("RMS range error / true range, t ≥ 10 s", lambda s: fmt_pct(s["rms_rel_err_after10s"])),
                 ("2σ coverage, t ≥ 2 s (ideal 95%)", lambda s: fmt_pct(s["coverage_2sigma"])),
                 ("2σ coverage with NaN steps counted as misses", lambda s: fmt_pct(s["extra"]["coverage_2sigma_nan_as_miss"])),
                 ("Range ANEES (1 dof; ideal 1)", lambda s: f"{s['anees_range']:.2f}"),
                 ("Bias (mean signed relative error), t ≥ 10 s", lambda s: f"{100 * s['mean_rel_err_after10s']:+.1f}%"),
                 ("NaN runs / 200", lambda s: str(s["nan"]["nan_runs"]))]
        for title, f in grids:
            T += ["", f"### {title}", "", "| behaviour | " + " | ".join(VARIANTS) + " |",
                  "|---|" + "---|" * len(VARIANTS)]
            T += [f"| {TARGET_LABEL[t]} | " + " | ".join(f(summ[grid_name(t, v)]) for v in VARIANTS) + " |"
                  for t in GRID_TARGETS]
    k_rows = [n for n in summ if n.startswith("K_")]
    if k_rows:
        T += ["", "## Filter process noise q = 1.0 vs 0.1 (sweep ±15 m, size_prior)", "",
              "| behaviour | RMS err q=0.1 | RMS err q=1.0 | 2σ coverage q=0.1 | 2σ coverage q=1.0 | NaN runs q=0.1 / 1.0 |",
              "|---|---|---|---|---|---|"]
        for n in k_rows:
            j = summ["J" + n[1:]]
            k = summ[n]
            T.append(f"| {TARGET_LABEL[k['config']['target']]} | {fmt_pct(j['rms_rel_err_after10s'])} | "
                     f"{fmt_pct(k['rms_rel_err_after10s'])} | {fmt_pct(j['coverage_2sigma'])} | {fmt_pct(k['coverage_2sigma'])} | "
                     f"{j['nan']['nan_runs']} / {k['nan']['nan_runs']} |")
    T += ["", "## 200 m scoring cut applied to the existing main-grid runs", "",
          f"Unchanged: {len(cut_check['unchanged'])} of {len(cut_check['unchanged']) + len(cut_check['changed'])} configs "
          f"(groups fully unchanged: {', '.join(sorted({n[0] for n in cut_check['unchanged']} - {n[0] for n in cut_check['changed']}))}). "
          "Changed (these targets fly beyond 200 m; results_sim.md is NOT modified, the cut is applied only to the "
          "sweep study):", ""]
    T += [f"* {n}: {100 * v['frac_samples_beyond_200m']:.1f}% of samples beyond 200 m; RMS "
          f"{fmt_pct(v['headline'].get('rms_rel_err_after10s', (None, None))[0])} -> "
          f"{fmt_pct(v['headline'].get('rms_rel_err_after10s', (None, None))[1])}"
          for n, v in cut_check["changed"].items()] or ["* none"]
    (EXP / "sweep_tables.md").write_text("\n".join(T) + "\n")
    print("\n".join(T), flush=True)


def main():
    OUT_NPZ.mkdir(parents=True, exist_ok=True)
    report_only = "--report-only" in sys.argv
    confs = sweep_configs()
    n_proc = cpu_count()
    timing = {"processes": n_proc}
    t_all = time.time()
    if not report_only:
        todo = [n for n in confs if not (OUT_NPZ / f"{n}.json").exists()]
        print(f"{len(confs)} configs, {len(todo)} to run, {n_proc} processes", flush=True)
        with Pool(n_proc) as pool:
            for i, name in enumerate(todo):
                t0 = time.time()
                s = run_config(name, confs[name], pool)
                dt = time.time() - t0
                timing.setdefault("per_config_s", {})[name] = round(dt, 1)
                eta = dt * (len(todo) - i - 1)
                print(f"{name:40s} {dt:6.1f} s  RMS {100 * s['rms_rel_err_after10s']:6.1f}%  cov "
                      f"{100 * s['coverage_2sigma']:5.1f}%  NaN runs {s['nan']['nan_runs']:3d}  "
                      f"ETA rest {eta / 60:5.1f} min", flush=True)
    summ = {n: json.loads((OUT_NPZ / f"{n}.json").read_text()) for n in confs if (OUT_NPZ / f"{n}.json").exists()}
    timing["total_s_this_invocation"] = round(time.time() - t_all, 1)
    if len(summ) < len(confs):
        print(f"only {len(summ)} of {len(confs)} configs done; reports not written", flush=True)
        return
    cut = range_cut_effect_on_existing()
    write_reports(summ, cut, timing)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
