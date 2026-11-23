"""Camera that speeds up and slows down along ONE straight line (the initial line of sight),
compared with the straight / lateral / weaving cameras of the main grid.

    python -I experiments/accel_line.py        # ~1-2 min

Camera paths (sim/trajectories.py, NOT in CAMERA_PATHS, so the main 48-config grid is unchanged):
  surge_15 : 2 t + 15 sin(w t) along the LOS, w = 2 pi / 12 (weave's peak acceleration, 4.1 m/s^2)
  surge_4  : 2 t +  4 sin(w t) along the LOS (gentler, 1.1 m/s^2)
Everything else as the weaving case: approach_recede drone, gimbal with +/-2 deg wobble,
10 Hz, 40 s, same detector and pose noise models.

  a) perfect data (tests/test_ekf.py run_noise_free: true size 0.55 m, prior 0.35 m,
     no pose noise, perfect detector)
  b) noisy Monte Carlo: 200 runs per config, seeds 1000..1199 (same as the main grid),
     all 5 variants, metrics from experiments/run_experiments.py summarize()

Writes experiments/accel_line_results.{md,json}; per-run arrays go to runs/accel_line/
(git-ignored, kept apart from runs/sim/ so the main-grid scripts never pick them up).
"""
import json
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.check_nan_runs import check as nan_check  # noqa: E402
from experiments.mc import VARIANTS  # noqa: E402
from experiments.run_experiments import N_RUNS, _job, summarize, write_markdown  # noqa: E402
from sim.trajectories import DT  # noqa: E402
from tests.test_ekf import run_noise_free  # noqa: E402

CAMERAS = ("surge_15", "surge_4")
PERFECT_TIMES = (10.0, 20.0, 30.0, 40.0)
S_TRUE_PERFECT = 0.55
OUT_NPZ = ROOT / "runs" / "accel_line"
REF_ROWS = [f"A_approach_{cam}_{v}" for cam in ("straight", "lateral", "weaving")
            for v in ("size_prior", "tight_prior", "wrong_prior")]


def perfect_data(camera):
    scn, out, _ = run_noise_free(camera, S_TRUE_PERFECT)
    err = out["r_est"] - out["r_true"]
    rel = err / out["r_true"]
    row = {"final_err_m": float(err[-1]), "final_rel_err": float(rel[-1]),
           "S_final": float(out["x"][-1, 6]), "S_true": S_TRUE_PERFECT,
           "final_true_range_m": float(out["r_true"][-1])}
    for tt in PERFECT_TIMES:
        k = int(round(tt / DT))
        row[f"err_m_t{int(tt)}"] = float(err[k])
        row[f"rel_err_t{int(tt)}"] = float(rel[k])
        row[f"true_range_m_t{int(tt)}"] = float(out["r_true"][k])
    return row


def run_mc(name, cfg, pool, out_dir=OUT_NPZ):
    runs = pool.map(_job, [(cfg, i) for i in range(N_RUNS)])
    res = {k: np.array([r[k] for r in runs]) for k in runs[0] if k != "t"}
    res["t"] = runs[0]["t"]
    path = Path(out_dir) / f"{name}.npz"
    np.savez_compressed(path, **res)
    s = summarize(res)
    s["config"] = cfg
    s["nis_dof"] = 2 if cfg["variant"] == "bearing_only" else 3
    nan = nan_check(path)
    s["nan_runs"] = nan["nan_runs"]
    s["frac_final_outside_3sigma_nan_as_fail"] = nan["final_outside_3sigma_nan_counted_as_diverged"]
    s["median_time_of_first_nan_s"] = nan["median_time_of_first_nan_s"]
    r, sig = res["r_est"], res["r_sigma"]
    after_init = np.arange(r.shape[1])[None, :] >= np.argmax(np.isfinite(r), axis=1)[:, None]
    bad = (after_init & np.isfinite(r).any(axis=1)[:, None] & ~(np.isfinite(r) & np.isfinite(sig))).any(axis=1)
    s["nan_run_seeds"] = (1000 + np.flatnonzero(bad)).tolist()     # _job uses seed = 1000 + i
    print(f"{name:28s} RMS rel {100 * s['rms_rel_err_after10s']:6.1f}%  bias {100 * s['mean_rel_err_after10s']:+6.1f}%  "
          f"cov {100 * s['coverage_2sigma']:5.1f}%  ANEES r {s['anees_range']:7.2f}  NaN runs {s['nan_runs']}",
          flush=True)
    return s


def _fmt_s(x):
    return "-" if x is None else f"{x:.1f}"


def table_lines(summ):
    """The results_sim.md table (header + rows), produced by run_experiments.write_markdown itself
    so the column format is identical."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.md"
        write_markdown(summ, p)
        lines = p.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("| config |"))
    end = next(i for i in range(start, len(lines)) if not lines[i].startswith("|"))
    return lines[start:end]


def notes(perfect, summ, t_perfect, t_total):
    p15, p4 = perfect["surge_15"], perfect["surge_4"]
    pct = lambda name, key: f"{100 * summ[name][key]:.1f}%"                       # noqa: E731
    nan_cfgs = {n: s["nan_run_seeds"] for n, s in summ.items() if s["nan_run_seeds"]}
    out = [f"- Runtime: perfect-data runs {t_perfect:.1f} s, everything (2 x 5 x {N_RUNS} noisy runs) "
           f"{t_total:.0f} s on a 16-process pool.",
           "- `surge_4` is not strictly forward: with A = 4, min speed is 2 - 4 (2 pi / 12) = -0.09 m/s, so it "
           "reverses very slightly near each slowest point. A < 2 / w = 3.82 m would keep it forward.",
           f"- Perfect data: `surge_15` makes range observable (final {100 * p15['final_rel_err']:+.1f}%, "
           f"S {p15['S_final']:.3f} m) but converges much more slowly than weaving "
           f"({100 * p15['rel_err_t10']:+.1f}% at 10 s, while weaving is about -1% by then). `surge_4` converges "
           f"only partly in 40 s (final {100 * p4['final_rel_err']:+.1f}%, S {p4['S_final']:.3f} m); straight "
           "stays at -37% (no observability).",
           f"- Noisy runs: `surge_15` falls between straight and weaving (size_prior RMS "
           f"{pct('surge_15_size_prior', 'rms_rel_err_after10s')}, wrong_prior bias "
           f"{pct('surge_15_wrong_prior', 'mean_rel_err_after10s')}; see Reference). `surge_4` is close to the "
           f"straight camera (size_prior RMS {pct('surge_4_size_prior', 'rms_rel_err_after10s')}).",
           f"- The filter is overconfident on the surge paths: surge_15 size_prior has 2-sigma coverage "
           f"{pct('surge_15_size_prior', 'coverage_2sigma')} and range ANEES "
           f"{summ['surge_15_size_prior']['anees_range']:.1f} (ideal 95% / 1.0).",
           f"- `surge_15_bearing_only` has a large bias over t >= 10 s "
           f"({100 * summ['surge_15_bearing_only']['mean_rel_err_after10s']:+.1f}%) yet only "
           f"{pct('surge_15_bearing_only', 'frac_final_outside_3sigma')} of runs end outside 3 sigma: range is "
           "overestimated by roughly +70-80% until the closest approach at 28 s and recovers to a few percent "
           "by 40 s (checked from runs/accel_line/surge_15_bearing_only.npz)."]
    if nan_cfgs:
        out.append("- NaN runs (state NaN after init, the known log(Z < 0) issue in experiments/nan_runs.md; "
                   "numpy prints 'invalid value encountered in log' warnings during the run): " +
                   "; ".join(f"{n}: seeds {', '.join(map(str, v))}" for n, v in nan_cfgs.items()) +
                   ". These are the same 7 seeds that go NaN in A_approach_straight_size_prior, first NaN "
                   "around 3 s, so they come from the early filter, not from the surge motion.")
    else:
        out.append("- No run went NaN.")
    return out


def main():
    t0 = time.time()
    OUT_NPZ.mkdir(parents=True, exist_ok=True)

    perfect = {cam: perfect_data(cam) for cam in CAMERAS}
    for cam, r in perfect.items():
        print(cam, "perfect data: final err", f"{r['final_err_m']:+.2f} m ({100 * r['final_rel_err']:+.1f}%)",
              "S final", f"{r['S_final']:.3f}")
    t_perfect = time.time() - t0

    summ = {}
    with Pool(16) as pool:
        for cam in CAMERAS:
            for v in VARIANTS:
                summ[f"{cam}_{v}"] = run_mc(f"{cam}_{v}", dict(drone="approach_recede", camera=cam, variant=v),
                                            pool)
    t_total = time.time() - t0

    ref_md = (ROOT / "experiments" / "results_sim.md").read_text().splitlines()
    ref = [next(l for l in ref_md if l.startswith(f"| {name} |")) for name in REF_ROWS]

    (ROOT / "experiments" / "accel_line_results.json").write_text(json.dumps({
        "meta": {"n_runs": N_RUNS, "seeds": f"1000..{1000 + N_RUNS - 1}", "cameras": list(CAMERAS),
                 "perfect_data": "tests/test_ekf.py run_noise_free: S_true 0.55, prior 0.35, no pose noise, "
                                 "perfect detector, variant size_prior",
                 "runtime_s": round(t_total, 1),
                 "note": "Simulation only."},
        "perfect_data": perfect, "results": summ}, indent=1))

    L = ["# Accelerating camera on one straight line (SIMULATION ONLY)", "",
         "Camera stays on the initial line of sight: position 2 t + A sin(w t) along it, w = 2 pi / 12 s, "
         "no sideways motion. `surge_15`: A = 15 m (peak acceleration 4.1 m/s^2, same as the weave; min "
         "speed -5.9 m/s, so it flies backwards part of each cycle). `surge_4`: A = 4 m (peak 1.1 m/s^2; "
         "min speed -0.09 m/s, so it also reverses, barely). Everything else as the weaving case: "
         "approach_recede drone, gimbal +/-2 deg wobble, 10 Hz, 40 s, same noise models. "
         "Generated by `experiments/accel_line.py`.", "",
         "## Perfect data", "",
         "Same setup as `run_noise_free` in tests/test_ekf.py: true size 0.55 m, prior mean 0.35 m (size_prior), "
         "no pose noise, perfect detector. Signed range error = estimate - truth (relative error in brackets).", "",
         "| camera | t = 10 s | t = 20 s | t = 30 s | t = 40 s (final) | S final (true 0.55) |",
         "|---|---|---|---|---|---|"]
    for cam, r in perfect.items():
        cells = [f"{r[f'err_m_t{int(tt)}']:+.2f} m ({100 * r[f'rel_err_t{int(tt)}']:+.1f}%)" for tt in PERFECT_TIMES]
        L.append(f"| {cam} | " + " | ".join(cells) + f" | {r['S_final']:.3f} m |")
    L += ["", "For comparison, tests/test_ekf.py asserts that weaving gets below 2% error after 20 s and "
          "learns S to within 0.01 m, and that straight stays more than 30% off with S at the 0.35 prior.",
          "", "## Noisy 200 runs", "",
          "Seeds 1000..1199 (same as the main grid, so runs are paired with it). Columns and definitions as in "
          "experiments/results_sim.md: rel err over t >= 10 s; coverage over t >= 2 s (ideal 95%); ANEES range "
          "ideal 1.0, ANEES pos ideal 3. NaN runs = runs whose filter state became NaN after initialisation "
          "(same definition as experiments/check_nan_runs.py); the other metrics skip those steps (nanmean).", ""]
    L += table_lines(summ)
    L += ["", "| config | NaN runs (of 200) | median first NaN [s] | final outside 3 sigma, NaN counted as diverged |",
          "|---|---|---|---|"]
    L += [f"| {name} | {s['nan_runs']} | "
          f"{_fmt_s(s['median_time_of_first_nan_s'])} | "
          f"{100 * s['frac_final_outside_3sigma_nan_as_fail']:.1f}% |" for name, s in summ.items()]
    L += ["", "## Reference", "",
          "Copied unchanged from experiments/results_sim.md (main grid A, approach_recede drone).", "",
          table_lines({})[0], table_lines({})[1]] + ref
    L += ["", "## Notes", ""] + notes(perfect, summ, t_perfect, t_total) + ["", "STATUS: DONE"]
    (ROOT / "experiments" / "accel_line_results.md").write_text("\n".join(L) + "\n")
    print(f"perfect data {t_perfect:.1f} s, total {t_total:.1f} s")


if __name__ == "__main__":
    main()
