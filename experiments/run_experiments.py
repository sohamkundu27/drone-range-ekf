"""Run every simulation experiment (200 Monte Carlo runs per configuration) and save results.

    python experiments/run_experiments.py            # ~5-10 min on 8 cores
    python experiments/make_figures.py               # figures/ from the saved results

Outputs
    runs/sim/<config>.npz           per-run time series (git-ignored, used by make_figures.py)
    experiments/results_sim.json    summary metrics for every configuration
    experiments/results_sim.md      the same as markdown tables

Experiments
    A  main grid      drone approach_recede x camera {straight, lateral, weaving}
                      x variant {bearing_only, size_prior, tight_prior, wrong_prior, wrong_tight}
    B  other paths    drone {constant_velocity, gentle_turn} x camera x {bearing_only, size_prior}
    C  plain EKF      as A/size_prior but n_iter = 1 (no iterations), to show why we iterate
    D  pose stress    as A/size_prior with: white 1.5 m / 0.5 deg GPS/INS noise; drift that the
                      filter ignores (treats as white); near-perfect pose (1 cm, 0.01 deg)
    E  consistency    truth drawn from the filter's own model (NEES/NIS check), start 40/80/150 m
    F  R inflation    as A but the filter assumes 2x the true box-width noise (a tuning mitigation
                      for the linearisation inconsistency described in STUDY.md)
All runs use the detector noise model in sim/noise.py (experiments/detector_noise.json if present).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ekf.nees import anees_bounds, consistency_report  # noqa: E402
from experiments.mc import VARIANTS, run_one  # noqa: E402
from sim.noise import DetectorNoise, PoseNoise, white_pose_noise  # noqa: E402
from sim.trajectories import CAMERA_PATHS, cv_with_process_noise, make_scenario  # noqa: E402

N_RUNS = 200
OUT_NPZ = ROOT / "runs" / "sim"
KEEP = ["r_true", "r_est", "r_sigma", "nees_pos", "nees_range", "nis", "detected", "w_true"]
RANGE_BINS = [10, 20, 40, 80, 120, 160, 200]

PRECISE_POSE = PoseNoise(sigma_pos=0.01, sigma_pos_white=0.01, sigma_att=np.deg2rad(0.01),
                         sigma_att_white=np.deg2rad(0.01))
# filter that treats the 1.5 m / 0.5 deg error as white noise although it really drifts
DRIFT_AS_WHITE = white_pose_noise(sigma_pos=1.5, sigma_att=np.deg2rad(0.5))


# ------------------------------------------------------------------ running
def _job(args):
    """One Monte Carlo run. args = (cfg dict, run index)."""
    cfg, i = args
    seed = 1000 + i      # same seeds across configurations -> paired comparisons
    if cfg.get("model_truth"):
        # E: drone follows the filter's own CV + white-acceleration model
        rng = np.random.default_rng(50_000 + i)
        t = np.arange(0, 30.0 + 1e-9, 0.1)
        p0 = np.array([0.0, cfg["start_range"], 45.0]) + rng.normal(0, 5, 3)
        dp, dv = cv_with_process_noise(p0, rng.normal(0, 3, 3), t, 0.1, rng)
        scn = make_scenario("constant_velocity", cfg["camera"], duration=30.0, drone_override=(dp, dv))
    else:
        scn = make_scenario(cfg["drone"], cfg["camera"])
    out = run_one(scn, VARIANTS[cfg["variant"]], seed, pose_noise=cfg.get("pose_noise"),
                  filter_pose_noise=cfg.get("filter_pose_noise"), n_iter=cfg.get("n_iter", 3),
                  width_sigma_scale=cfg.get("width_sigma_scale", 1.0),
                  det_noise=DetectorNoise())
    keep = {k: out[k] for k in KEEP}
    keep["S_est"] = out["x"][:, 6]
    keep["S_sigma"] = np.sqrt(out["P_diag"][:, 6])
    keep["S_true"] = out["S_true"]
    keep["t"] = out["t"]
    return keep


def run_config(name, cfg, pool):
    runs = pool.map(_job, [(cfg, i) for i in range(N_RUNS)])
    res = {k: np.array([r[k] for r in runs]) for k in runs[0] if k != "t"}
    res["t"] = runs[0]["t"]
    np.savez_compressed(OUT_NPZ / f"{name}.npz", **res)
    s = summarize(res)
    s["config"] = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in cfg.items()}
    print(f"{name:45s} RMS rel range err (t>=10s) {100 * s['rms_rel_err_after10s']:6.1f}%   "
          f"2-sigma coverage {100 * s['coverage_2sigma']:5.1f}%   ANEES range {s['anees_range']:7.2f}   "
          f"diverged {100 * s['frac_final_outside_3sigma']:4.1f}%", flush=True)
    return s


# ------------------------------------------------------------------ metrics
def summarize(res):
    t = res["t"]
    err = res["r_est"] - res["r_true"]
    rel = err / res["r_true"]
    after2 = t >= 2.0
    after10 = t >= 10.0
    ok = ~np.isnan(rel)
    inside2 = np.abs(err) <= 2 * res["r_sigma"]
    s = {
        "n_runs": int(rel.shape[0]),
        "rms_rel_err_after10s": float(np.sqrt(np.nanmean(rel[:, after10] ** 2))),
        "median_abs_rel_err_after10s": float(np.nanmedian(np.abs(rel[:, after10]))),
        "mean_rel_err_after10s": float(np.nanmean(rel[:, after10])),          # bias
        "median_reported_rel_sigma_after10s": float(np.nanmedian((res["r_sigma"] / res["r_est"])[:, after10])),
        "coverage_2sigma": float(np.sum(inside2[:, after2] & ok[:, after2]) / np.sum(ok[:, after2])),
        "anees_range": float(np.nanmean(res["nees_range"][:, after2])),
        "anees_pos": float(np.nanmean(res["nees_pos"][:, after2])),
        "mean_nis": float(np.nanmean(res["nis"][:, after2])),
        "nis_dof": None,
        "detection_rate": float(np.mean(res["detected"])),
    }
    # final time step
    fin_rel = rel[:, -1]
    fin_sig = (res["r_sigma"] / res["r_est"])[:, -1]
    s["final_rms_rel_err"] = float(np.sqrt(np.nanmean(fin_rel**2)))
    s["frac_final_outside_3sigma"] = float(np.nanmean(np.abs(fin_rel) > 3 * fin_sig))
    s["frac_final_err_over_50pct"] = float(np.nanmean(np.abs(fin_rel) > 0.5))
    s["S_true_mean"] = float(np.mean(res["S_true"]))
    s["S_final_mean"] = float(np.nanmean(res["S_est"][:, -1]))
    s["S_final_rms_err"] = float(np.sqrt(np.nanmean((res["S_est"][:, -1] - res["S_true"]) ** 2)))
    s["S_final_reported_sigma"] = float(np.nanmedian(res["S_sigma"][:, -1]))
    # RMS relative error per true-range bin (all times after 2 s)
    bins = {}
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        m = (res["r_true"] >= lo) & (res["r_true"] < hi) & ok & after2[None, :]
        if m.sum() > 0:
            bins[f"{lo}-{hi}m"] = {"rms_rel_err": float(np.sqrt(np.mean(rel[m] ** 2))),
                                   "median_reported_rel_sigma": float(np.median((res["r_sigma"] / res["r_est"])[m])),
                                   "n": int(m.sum())}
    s["by_range"] = bins
    # chi-square consistency on range NEES (1 dof)
    rep = consistency_report(res["nees_range"][:, after2], dof=1)
    s["range_nees_verdict"] = rep["verdict"]
    s["range_nees_bounds"] = rep["bounds"]
    s["range_nees_frac_steps_inside"] = rep["frac_steps_inside"]
    return s


# ------------------------------------------------------------------ configurations
def configs():
    c = {}
    for cam in CAMERA_PATHS:
        for v in VARIANTS:
            c[f"A_approach_{cam}_{v}"] = dict(drone="approach_recede", camera=cam, variant=v)
    for drone in ("constant_velocity", "gentle_turn"):
        for cam in CAMERA_PATHS:
            for v in ("bearing_only", "size_prior"):
                c[f"B_{drone}_{cam}_{v}"] = dict(drone=drone, camera=cam, variant=v)
    for cam in CAMERA_PATHS:
        c[f"C_plainEKF_approach_{cam}_size_prior"] = dict(drone="approach_recede", camera=cam,
                                                          variant="size_prior", n_iter=1)
    for cam in ("straight", "weaving"):
        base = dict(drone="approach_recede", camera=cam, variant="size_prior")
        c[f"D_whitepose_{cam}"] = dict(base, pose_noise=white_pose_noise())
        c[f"D_drift_filter_assumes_white_{cam}"] = dict(base, filter_pose_noise=DRIFT_AS_WHITE)
        c[f"D_precise_pose_{cam}"] = dict(base, pose_noise=PRECISE_POSE)
    for r0 in (40.0, 80.0, 150.0):
        for cam in ("weaving", "lateral"):
            c[f"E_model_truth_{cam}_{int(r0)}m"] = dict(model_truth=True, camera=cam, start_range=r0,
                                                         variant="size_prior")
    for cam in CAMERA_PATHS:
        for v in ("size_prior", "wrong_prior"):
            c[f"F_widthx2_approach_{cam}_{v}"] = dict(drone="approach_recede", camera=cam, variant=v,
                                                      width_sigma_scale=2.0)
    return c


def write_markdown(summ, path):
    lines = ["# Simulation results (200 Monte Carlo runs each; SIMULATION ONLY)", "",
             f"Detector noise model: `{DetectorNoise().source}`.", "",
             "rel err = (estimated range - true range) / true range, over t >= 10 s. "
             "coverage = fraction of time steps (t >= 2 s) with |range error| <= 2 sigma (ideal 95%). "
             "ANEES range: mean of the 1-dof range NEES (ideal 1.0). "
             "diverged = final error outside +/-3 sigma.", "",
             "| config | RMS rel err | bias | reported sigma/r | 2-sigma coverage | ANEES range | ANEES pos (3) | "
             "mean NIS | final outside 3sigma | final err > 50% | S final (true) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summ.items():
        lines.append(
            f"| {name} | {100 * s['rms_rel_err_after10s']:.1f}% | {100 * s['mean_rel_err_after10s']:+.1f}% | "
            f"{100 * s['median_reported_rel_sigma_after10s']:.1f}% | {100 * s['coverage_2sigma']:.1f}% | "
            f"{s['anees_range']:.2f} | {s['anees_pos']:.2f} | {s['mean_nis']:.2f} | "
            f"{100 * s['frac_final_outside_3sigma']:.1f}% | {100 * s['frac_final_err_over_50pct']:.1f}% | "
            f"{s['S_final_mean']:.3f} ({s['S_true_mean']:.3f}) |")
    lines += ["", "## RMS relative range error by true range (main grid A)", "",
              "| config | " + " | ".join(f"{a}-{b} m" for a, b in zip(RANGE_BINS[:-1], RANGE_BINS[1:])) + " |",
              "|---|" + "---|" * (len(RANGE_BINS) - 1)]
    for name, s in summ.items():
        if name.startswith("A_"):
            cells = [f"{100 * s['by_range'][k]['rms_rel_err']:.1f}%" if k in s["by_range"] else "-"
                     for k in (f"{a}-{b}m" for a, b in zip(RANGE_BINS[:-1], RANGE_BINS[1:]))]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    OUT_NPZ.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]          # optional: run only configs whose name starts with these prefixes
    summ = {}
    out_json = ROOT / "experiments" / "results_sim.json"
    if only and out_json.exists():
        summ = json.loads(out_json.read_text())["results"]
    with Pool(8) as pool:
        for name, cfg in configs().items():
            if only and not any(name.startswith(p) for p in only):
                continue
            summ[name] = run_config(name, cfg, pool)
    for s in summ.values():
        s["nis_dof"] = 2 if "bearing_only" in s["config"]["variant"] else 3
    meta = {"n_runs": N_RUNS, "detector_noise_source": DetectorNoise().source,
            "anees_range_bounds_95": list(anees_bounds(1, N_RUNS)),
            "anees_pos_bounds_95": list(anees_bounds(3, N_RUNS)),
            "note": "Simulation only. Range accuracy has not been validated on real flight data."}
    out_json.write_text(json.dumps({"meta": meta, "results": summ}, indent=1))
    write_markdown(summ, ROOT / "experiments" / "results_sim.md")
    print("wrote", out_json)


if __name__ == "__main__":
    main()
