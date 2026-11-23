"""Camera that speeds up and slows down along the SIDEWAYS axis (perpendicular to the initial
line of sight), with no forward motion. Companion to experiments/accel_line.py (same motion
along the line of sight).

    python -I experiments/accel_lateral.py        # ~20 s

Camera paths (sim/trajectories.py, NOT in CAMERA_PATHS, so the main 48-config grid is unchanged):
  lat_surge_15 : 2 t + 15 sin(w t) along the sideways axis, w = 2 pi / 12 (peak acc 4.1 m/s^2)
  lat_surge_4  : 2 t +  4 sin(w t) along the sideways axis (peak acc 1.1 m/s^2)
Everything else as accel_line.py: approach_recede drone, gimbal +/-2 deg wobble, 10 Hz, 40 s,
same detector and pose noise models.

  1) perfect data (tests/test_ekf.py run_noise_free: true size 0.55 m, prior 0.35 m,
     size_prior, no pose noise, perfect detector)
  2) 200 seeded Monte Carlo runs (seeds 1000..1199) x 5 variants per path

The helpers (perfect_data, run_mc, table_lines) are imported from accel_line.py so both
experiments compute every number the same way. Writes experiments/accel_lateral_results.{md,json};
per-run arrays go to runs/accel_lateral/ (git-ignored).
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.accel_line import (PERFECT_TIMES, _fmt_s, perfect_data, run_mc,  # noqa: E402
                                    table_lines)
from experiments.mc import VARIANTS  # noqa: E402
from experiments.run_experiments import N_RUNS  # noqa: E402

CAMERAS = ("lat_surge_15", "lat_surge_4")
PERFECT_REF = ("lateral", "weaving", "surge_15", "surge_4")      # recomputed here, same function
OUT_NPZ = ROOT / "runs" / "accel_lateral"
REF_SIM = [f"A_approach_{cam}_{v}" for cam in ("lateral", "weaving") for v in VARIANTS]
REF_LINE = [f"{cam}_{v}" for cam in ("surge_15", "surge_4") for v in VARIANTS]


def perfect_row(cam, r):
    cells = [f"{r[f'err_m_t{int(tt)}']:+.2f} m ({100 * r[f'rel_err_t{int(tt)}']:+.1f}%)" for tt in PERFECT_TIMES]
    return f"| {cam} | " + " | ".join(cells) + f" | {r['S_final']:.3f} m |"


def notes(perfect, summ, line_summ, t_perfect, t_total):
    pct = lambda d, name, key: f"{100 * d[name][key]:.1f}%"                       # noqa: E731
    out = [f"- Runtime: perfect-data runs {t_perfect:.1f} s, everything (2 x 5 x {N_RUNS} noisy runs) "
           f"{t_total:.0f} s on a 16-process pool.",
           "- Like `surge_4`, `lat_surge_4` is not strictly one-directional: min speed 2 - 4 (2 pi / 12) = "
           "-0.09 m/s, so it reverses very slightly near each slowest point. `lat_surge_15` reverses for "
           "part of every cycle (min speed -5.9 m/s).",
           "- Comparison pairs: `lat_surge_*` vs `surge_*` differ ONLY in the axis of motion (sideways vs "
           "along the initial line of sight), with identical speed and acceleration profiles. The `lateral` "
           "camera is the constant-velocity sideways baseline (4 m/s, no acceleration).",
           "- size_prior RMS rel err: " + ", ".join(
               f"{n} {pct(d, k, 'rms_rel_err_after10s')}" for n, d, k in [
                   ("lat_surge_15", summ, "lat_surge_15_size_prior"),
                   ("lat_surge_4", summ, "lat_surge_4_size_prior"),
                   ("surge_15", line_summ, "surge_15_size_prior"),
                   ("surge_4", line_summ, "surge_4_size_prior")]) + "; lateral and weaving in the Reference table.",
           f"- Perfect data, final range error: lat_surge_15 {100 * perfect['lat_surge_15']['final_rel_err']:+.1f}%, "
           f"lat_surge_4 {100 * perfect['lat_surge_4']['final_rel_err']:+.1f}%, "
           f"surge_15 {100 * perfect['surge_15']['final_rel_err']:+.1f}%, "
           f"surge_4 {100 * perfect['surge_4']['final_rel_err']:+.1f}%, "
           f"lateral {100 * perfect['lateral']['final_rel_err']:+.1f}%, weaving {100 * perfect['weaving']['final_rel_err']:+.1f}%."]
    sim = json.loads((ROOT / "experiments" / "results_sim.json").read_text())["results"]
    worst = max(abs(summ[f"lat_surge_15_{v}"]["rms_rel_err_after10s"] - sim[f"A_approach_weaving_{v}"]["rms_rel_err_after10s"])
                for v in VARIANTS if v != "bearing_only")
    out.append(f"- `lat_surge_15` reproduces `weaving` almost exactly (RMS rel err within {100 * worst:.1f} percentage "
               "points for every variant except bearing_only): both have the same +/-15 m, 12 s sideways oscillation "
               "and differ only in the 2 m/s drift direction (sideways vs forward). The sideways ACCELERATION is "
               "what matters, and for the same profile sideways beats along-the-line-of-sight motion at both "
               "amplitudes.")
    nan_cfgs = {n: s["nan_run_seeds"] for n, s in summ.items() if s["nan_run_seeds"]}
    if nan_cfgs:
        ref_nan = {"surge_15_size_prior": set(line_summ["surge_15_size_prior"]["nan_run_seeds"])}
        weave_npz = ROOT / "runs" / "sim" / "A_approach_weaving_size_prior.npz"     # local, git-ignored
        if weave_npz.exists():
            z = np.load(weave_npz)
            r, sig = z["r_est"], z["r_sigma"]
            after_init = np.arange(r.shape[1])[None, :] >= np.argmax(np.isfinite(r), axis=1)[:, None]
            bad = (after_init & np.isfinite(r).any(axis=1)[:, None] & ~(np.isfinite(r) & np.isfinite(sig))).any(axis=1)
            ref_nan["A_approach_weaving_size_prior"] = set((1000 + np.flatnonzero(bad)).tolist())
        out.append("- NaN runs (state NaN after init, the known log(Z < 0) issue in experiments/nan_runs.md; "
                   "numpy prints 'invalid value encountered in log' warnings during the run): " +
                   "; ".join(f"{n}: {len(v)} runs, seeds {', '.join(map(str, v))} (also NaN: " +
                             ", ".join(f"{len(set(v) & rs)} in {rn}" for rn, rs in ref_nan.items()) + ")"
                             for n, v in nan_cfgs.items()) +
                   ". Median time of the first NaN: " +
                   ", ".join(f"{n} {_fmt_s(summ[n]['median_time_of_first_nan_s'])} s" for n in nan_cfgs) +
                   ", i.e. early in the run, while range is still very uncertain; most seeds overlap with the "
                   "existing NaN runs of the reference cameras.")
    else:
        out.append("- No run went NaN.")
    return out


def main():
    t0 = time.time()
    OUT_NPZ.mkdir(parents=True, exist_ok=True)

    perfect = {cam: perfect_data(cam) for cam in CAMERAS + PERFECT_REF}
    for cam in CAMERAS:
        r = perfect[cam]
        print(cam, "perfect data: final err", f"{r['final_err_m']:+.2f} m ({100 * r['final_rel_err']:+.1f}%)",
              "S final", f"{r['S_final']:.3f}")
    t_perfect = time.time() - t0

    summ = {}
    with Pool(16) as pool:
        for cam in CAMERAS:
            for v in VARIANTS:
                summ[f"{cam}_{v}"] = run_mc(f"{cam}_{v}", dict(drone="approach_recede", camera=cam, variant=v),
                                            pool, out_dir=OUT_NPZ)
    t_total = time.time() - t0

    sim_md = (ROOT / "experiments" / "results_sim.md").read_text().splitlines()
    line_md = (ROOT / "experiments" / "accel_line_results.md").read_text().splitlines()
    line_summ = json.loads((ROOT / "experiments" / "accel_line_results.json").read_text())["results"]
    ref_sim = [next(l for l in sim_md if l.startswith(f"| {n} |")) for n in REF_SIM]
    ref_line = [next(l for l in line_md if l.startswith(f"| {n} |")) for n in REF_LINE]

    (ROOT / "experiments" / "accel_lateral_results.json").write_text(json.dumps({
        "meta": {"n_runs": N_RUNS, "seeds": f"1000..{1000 + N_RUNS - 1}", "cameras": list(CAMERAS),
                 "perfect_data": "tests/test_ekf.py run_noise_free: S_true 0.55, prior 0.35, no pose noise, "
                                 "perfect detector, variant size_prior",
                 "perfect_data_reference_cameras": list(PERFECT_REF),
                 "runtime_s": round(t_total, 1),
                 "note": "Simulation only."},
        "perfect_data": perfect, "results": summ}, indent=1))

    header = table_lines({})
    L = ["# Accelerating camera along the sideways axis (SIMULATION ONLY)", "",
         "Camera moves only sideways (perpendicular to the initial line of sight), no forward motion: "
         "position 2 t + A sin(w t) along that axis, w = 2 pi / 12 s. `lat_surge_15`: A = 15 m (peak "
         "acceleration 4.1 m/s^2, same as the weave; min speed -5.9 m/s, so it reverses part of each cycle). "
         "`lat_surge_4`: A = 4 m (peak 1.1 m/s^2; min speed -0.09 m/s, so it also reverses, barely). Same "
         "speed and acceleration profiles as `surge_15` / `surge_4` (experiments/accel_line_results.md), which "
         "move along the line of sight instead. Everything else as the weaving case: approach_recede drone, "
         "gimbal +/-2 deg wobble, 10 Hz, 40 s, same noise models. Generated by `experiments/accel_lateral.py`.", "",
         "## Perfect data", "",
         "Same setup as `run_noise_free` in tests/test_ekf.py: true size 0.55 m, prior mean 0.35 m (size_prior), "
         "no pose noise, perfect detector. Signed range error = estimate - truth (relative error in brackets). "
         "The rows below the new paths are the comparison cameras, rerun here with the same function.", "",
         "| camera | t = 10 s | t = 20 s | t = 30 s | t = 40 s (final) | S final (true 0.55) |",
         "|---|---|---|---|---|---|"]
    L += [perfect_row(cam, perfect[cam]) for cam in CAMERAS + PERFECT_REF]
    L += ["", "## Noisy 200 runs", "",
          "Seeds 1000..1199 (same as the main grid and accel_line.py, so runs are paired). Columns and definitions "
          "as in experiments/results_sim.md: rel err over t >= 10 s; coverage over t >= 2 s (ideal 95%); ANEES "
          "range ideal 1.0, ANEES pos ideal 3. NaN runs = runs whose filter state became NaN after "
          "initialisation (same definition as experiments/check_nan_runs.py); the other metrics skip those "
          "steps (nanmean).", ""]
    L += table_lines(summ)
    L += ["", "| config | NaN runs (of 200) | median first NaN [s] | final outside 3 sigma, NaN counted as diverged |",
          "|---|---|---|---|"]
    L += [f"| {name} | {s['nan_runs']} | {_fmt_s(s['median_time_of_first_nan_s'])} | "
          f"{100 * s['frac_final_outside_3sigma_nan_as_fail']:.1f}% |" for name, s in summ.items()]
    L += ["", "## Reference", "",
          "Copied unchanged from experiments/results_sim.md (main grid A: `lateral` = constant 4 m/s sideways, "
          "`weaving`) and experiments/accel_line_results.md (`surge_15`, `surge_4`).", ""]
    L += header[:2] + ref_sim + ref_line
    L += ["", "## Notes", ""] + notes(perfect, summ, line_summ, t_perfect, t_total) + ["", "STATUS: DONE"]
    (ROOT / "experiments" / "accel_lateral_results.md").write_text("\n".join(L) + "\n")
    print(f"perfect data {t_perfect:.1f} s, total {t_total:.1f} s")


if __name__ == "__main__":
    main()
