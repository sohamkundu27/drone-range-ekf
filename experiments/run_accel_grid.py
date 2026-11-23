"""The whole simulation grid (experiments A-F of run_experiments.py) with ACCELERATING cameras.

    python -I experiments/run_accel_grid.py            # ~2-3 min on 16 processes
    python -I experiments/make_figures_accel.py        # figures_accel/ from the saved runs

Every configuration of run_experiments.configs() is kept (48 configs, 200 runs each, seeds
1000..1199, same variants, detector noise and drone paths); only the camera is swapped
(sim/trajectories.py ACCEL_SWAP):
    straight -> surge_15      (2 t + 15 sin(2 pi t / 12) along the initial line of sight)
    lateral  -> lat_surge_15  (the same motion along the sideways axis)
    weaving  -> weaving       (unchanged: must reproduce experiments/results_sim.md exactly)
Config names get the new camera name, e.g. A_approach_straight_size_prior -> A_approach_surge_15_size_prior.

Reuses run_experiments.py (configs, run_config, summarize, write_markdown) and mc.py unchanged;
only the npz output folder is redirected. Nothing on main is overwritten.

Outputs
    runs/sim_accel/<config>.npz                 per-run time series (git-ignored)
    experiments/accel_grid/results_sim.{json,md} same tables and columns as experiments/results_sim.md
    experiments/accel_grid/nan_runs.{json,md}    same format as experiments/nan_runs.md
    experiments/accel_grid/comparison.md         constant-speed vs accelerating, per camera
"""
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.run_experiments as rx  # noqa: E402
from ekf.nees import anees_bounds  # noqa: E402
from experiments.check_nan_runs import check as nan_check  # noqa: E402
from sim.noise import DetectorNoise  # noqa: E402
from sim.trajectories import ACCEL_SWAP, DT  # noqa: E402

OUT_NPZ = ROOT / "runs" / "sim_accel"
OLD_NPZ = ROOT / "runs" / "sim"
OUT = ROOT / "experiments" / "accel_grid"
CMP_VARIANTS = ("size_prior", "tight_prior", "wrong_prior", "bearing_only")
CAM_LABEL = {"straight": "straight", "lateral": "lateral", "weaving": "weaving",
             "surge_15": "straight, accelerating", "lat_surge_15": "lateral, accelerating"}
PERFECT_CAMERAS = ("lateral", "lat_surge_15", "straight", "surge_15", "weaving")


def accel_name(name):
    """Config name with the camera token swapped (straight -> surge_15, lateral -> lat_surge_15)."""
    return re.sub(r"(?<=_)(straight|lateral)(?=_|$)", lambda m: ACCEL_SWAP[m.group(1)], name)


def accel_configs():
    out = {}
    for name, cfg in rx.configs().items():
        new = dict(cfg, camera=ACCEL_SWAP[cfg["camera"]])
        new_name = accel_name(name)
        assert (cfg["camera"] in name) and (new["camera"] in new_name), (name, new_name)
        out[new_name] = new
    assert len(out) == len(rx.configs()) == 48
    return out


def time_under(rel_err, thr=0.02):
    """First time after which |relative error| stays below thr until the end (None if never)."""
    ok = np.isfinite(rel_err) & (np.abs(rel_err) < thr)
    if not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    k = 0 if len(bad) == 0 else bad[-1] + 1
    return k * DT


def perfect_runs():
    from tests.test_ekf import run_noise_free
    res = {}
    for cam in PERFECT_CAMERAS:
        _, out, _ = run_noise_free(cam, 0.55)
        rel = (out["r_est"] - out["r_true"]) / out["r_true"]
        res[cam] = {"t": out["t"], "rel_err": rel, "S_final": float(out["x"][-1, 6]),
                    "t_under_2pct": time_under(rel)}
    return res


# ------------------------------------------------------------------ comparison
def cmp_stats(npz_path, summ):
    z = np.load(npz_path)
    rel = (z["r_est"] - z["r_true"]) / z["r_true"]
    late = z["t"] >= 10
    return {"rms": summ["rms_rel_err_after10s"], "cov": summ["coverage_2sigma"], "anees": summ["anees_range"],
            "nan": nan_check(npz_path)["nan_runs"],
            "sig": summ["median_reported_rel_sigma_after10s"],
            "p68": float(np.nanpercentile(np.abs(rel[:, late]), 68)),       # as in make_figures.py
            "S": summ["S_final_mean"], "S_true": summ["S_true_mean"]}


def write_comparison(new_summ, old_summ, weaving_check, perfect, runtime_s):
    L = ["# Constant-speed vs accelerating cameras (SIMULATION ONLY)", "",
         "Main grid A (approach_recede drone, 200 runs, seeds 1000..1199). Old = experiments/results_sim.md "
         "(constant-speed `straight` 4 m/s along the line of sight, `lateral` 4 m/s sideways); new = "
         "experiments/accel_grid/results_sim.md (`surge_15` / `lat_surge_15`: 2 t + 15 sin(2 pi t / 12) along "
         "the same axis, peak acceleration 4.1 m/s^2, same as the weave). `weaving` is in both and was rerun.", "",
         "Columns: RMS rel err over t >= 10 s; 2-sigma coverage over t >= 2 s (ideal 95%); ANEES range (ideal 1.0); "
         "NaN runs = runs whose state became NaN after initialisation (of 200); reported sigma = median sigma/r the "
         "filter reports, t >= 10 s; actual 68th pct = 68th percentile of |range error| / r over runs and t >= 10 s "
         "(about 1 sigma; an honest filter has reported sigma close to it); S final = mean final size estimate "
         "(mean true size).", ""]
    for v in CMP_VARIANTS:
        L += [f"## {v}", "",
              "| camera | RMS rel err | 2-sigma coverage | ANEES range | NaN runs | reported sigma/r | "
              "actual 68th pct | S final (true) |", "|---|---|---|---|---|---|---|---|"]
        rows = [("lateral", "old", "lateral"), ("lat_surge_15", "new", "lateral"),
                ("straight", "old", "straight"), ("surge_15", "new", "straight"),
                ("weaving", "old", "weaving"), ("weaving", "new", "weaving")]
        for cam, which, _ in rows:
            name = f"A_approach_{cam}_{v}"
            summ, folder = (old_summ, OLD_NPZ) if which == "old" else (new_summ, OUT_NPZ)
            s = cmp_stats(folder / f"{name}.npz", summ[name])
            label = CAM_LABEL[cam] + (" (rerun)" if which == "new" and cam == "weaving" else "")
            L.append(f"| {label} | {100 * s['rms']:.1f}% | {100 * s['cov']:.1f}% | {s['anees']:.2f} | {s['nan']} | "
                     f"{100 * s['sig']:.1f}% | {100 * s['p68']:.1f}% | {s['S']:.3f} ({s['S_true']:.3f}) |")
        L.append("")
    L += ["## Perfect data", "",
          "tests/test_ekf.py `run_noise_free` (true size 0.55 m, prior 0.35 m, size_prior, no pose noise, perfect "
          "detector). Time after which |range error| stays below 2% until the end of the 40 s run "
          "(see figures_accel/perfect_data_error_vs_time.png).", "",
          "| camera | final range error | S final (true 0.55) | under 2% from |", "|---|---|---|---|"]
    for cam, r in perfect.items():
        tu = "never" if r["t_under_2pct"] is None else f"t = {r['t_under_2pct']:.1f} s"
        L.append(f"| {CAM_LABEL[cam]} | {100 * r['rel_err'][-1]:+.1f}% | {r['S_final']:.3f} m | {tu} |")
    L += ["", "## Weaving reproduction check", "", weaving_check, "",
          f"Grid runtime: {runtime_s:.0f} s. Experiments B-F (other drone paths, plain EKF, pose stress, "
          "model-truth NEES, width x2) are in experiments/accel_grid/results_sim.md with the same row names "
          "as experiments/results_sim.md, camera swapped."]
    (OUT / "comparison.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ NaN report (format of nan_runs.md)
def write_nan_report():
    res = {p.stem: nan_check(p) for p in sorted(OUT_NPZ.glob("*.npz"))}
    total = sum(v["nan_runs"] for v in res.values())
    (OUT / "nan_runs.json").write_text(json.dumps(res, indent=1))
    L = ["# Runs whose filter state became NaN (diagnostic, SIMULATION ONLY) - accelerating-camera grid", "",
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
    n_clean = sum(v["nan_runs"] == 0 for v in res.values())
    L += ["", f"Configurations not listed had no NaN runs ({n_clean} of {len(res)})."]
    (OUT / "nan_runs.md").write_text("\n".join(L) + "\n")
    return total


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    OUT_NPZ.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rx.OUT_NPZ = OUT_NPZ            # run_config saves to rx.OUT_NPZ; redirect so runs/sim/ is untouched

    summ = {}
    with Pool(16) as pool:
        for name, cfg in accel_configs().items():
            summ[name] = rx.run_config(name, cfg, pool)
    for s in summ.values():
        s["nis_dof"] = 2 if "bearing_only" in s["config"]["variant"] else 3
    runtime = time.time() - t0

    meta = {"n_runs": rx.N_RUNS, "detector_noise_source": DetectorNoise().source,
            "anees_range_bounds_95": list(anees_bounds(1, rx.N_RUNS)),
            "anees_pos_bounds_95": list(anees_bounds(3, rx.N_RUNS)),
            "camera_swap": ACCEL_SWAP, "runtime_s": round(runtime, 1),
            "note": "Simulation only. Main grid with accelerating cameras; see experiments/run_accel_grid.py."}
    (OUT / "results_sim.json").write_text(json.dumps({"meta": meta, "results": summ}, indent=1))
    md = OUT / "results_sim.md"
    rx.write_markdown(summ, md)
    lines = md.read_text().splitlines()
    lines[0] = "# Simulation results, ACCELERATING cameras (200 Monte Carlo runs each; SIMULATION ONLY)"
    lines[1:1] = ["", "Same grid as experiments/results_sim.md with the camera swapped: straight -> surge_15 "
                      "(accelerating along the line of sight), lateral -> lat_surge_15 (accelerating sideways), "
                      "weaving unchanged. Generated by experiments/run_accel_grid.py."]
    md.write_text("\n".join(lines) + "\n")

    # weaving rows must reproduce the main grid exactly (same configs, same seeds)
    old = json.loads((ROOT / "experiments" / "results_sim.json").read_text())["results"]
    weave = [n for n in summ if "weaving" in n]
    diffs = [n for n in weave if {k: v for k, v in summ[n].items() if k != "config"}
             != {k: v for k, v in old[n].items() if k != "config"}]
    old_md = (ROOT / "experiments" / "results_sim.md").read_text().splitlines()
    md_diffs = [n for n in weave
                if [l for l in old_md if l.startswith(f"| {n} |")] != [l for l in lines if l.startswith(f"| {n} |")]]
    check = (f"All {len(weave)} weaving configurations (A, B, C, D, E, F) were rerun: "
             f"{len(weave) - len(diffs)} of {len(weave)} reproduce every metric in experiments/results_sim.json "
             f"exactly, and {len(weave) - len(md_diffs)} of {len(weave)} rows (main table and range-bin table) "
             f"are identical in experiments/results_sim.md." + (f" MISMATCH: {diffs + md_diffs}" if diffs or md_diffs else ""))
    print(check)

    n_nan = write_nan_report()
    write_comparison(summ, old, check, perfect_runs(), runtime)
    print(f"NaN runs: {n_nan}; wrote {OUT}; total {time.time() - t0:.0f} s")
    if diffs or md_diffs:
        sys.exit(1)


if __name__ == "__main__":
    main()
