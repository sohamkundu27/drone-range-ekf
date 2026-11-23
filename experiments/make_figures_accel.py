"""Copy of experiments/make_figures.py for the ACCELERATING-camera grid: same figures, filenames, sizes
and style, read from runs/sim_accel/*.npz (written by experiments/run_accel_grid.py) and saved to
figures_accel/. Camera paths: surge_15 ("straight, accelerating"), lat_surge_15 ("lateral, accelerating"),
weaving (unchanged).

    python -I experiments/make_figures_accel.py

figures_accel/range_error_envelopes.png   range error vs time with the filter's +/-2 sigma envelope
figures_accel/rmse_vs_range.png           RMS range error vs true range, per variant and camera motion
figures_accel/nees_time.png               range NEES (ANEES over 200 runs) vs time with chi-square bounds
figures_accel/uncertainty_vs_motion.png   actual error vs reported sigma, per camera motion
figures_accel/consistency_model_truth.png position NEES when the truth follows the filter's model
New:
figures_accel/accel_vs_constant.png       size_prior: RMS error and 2-sigma coverage, constant vs accelerating
figures_accel/perfect_data_error_vs_time.png  no-noise test: signed range error vs time, time to stay under 2%
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ekf.nees import anees_bounds  # noqa: E402

NPZ = ROOT / "runs" / "sim_accel"          # path switch (was runs/sim)
FIG = ROOT / "figures_accel"               # path switch (was figures)
CAMERAS = ["surge_15", "lat_surge_15", "weaving"]          # straight -> surge_15, lateral -> lat_surge_15
CAM_LABEL = {"straight": "straight", "lateral": "lateral", "weaving": "weaving",
             "surge_15": "straight, accelerating", "lat_surge_15": "lateral, accelerating"}
VARIANTS = ["bearing_only", "size_prior", "tight_prior", "wrong_prior", "wrong_tight"]
LABEL = {"bearing_only": "bearing only", "size_prior": "size prior 0.35±0.15 m",
         "tight_prior": "tight prior 0.35±0.03 m", "wrong_prior": "wrong: true 0.60, prior 0.35±0.15",
         "wrong_tight": "wrong: true 0.60, prior 0.35±0.03"}
# fixed categorical order (validated palette); each variant keeps its colour in every figure
COLOR = dict(zip(VARIANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKER = dict(zip(VARIANTS, ["o", "s", "^", "D", "v"]))          # secondary encoding
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "figure.dpi": 130,
})


def load(name):
    return np.load(NPZ / f"{name}.npz")


def rel_err(d):
    return (d["r_est"] - d["r_true"]) / d["r_true"]


def fig_envelopes():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
    for row, v in enumerate(["size_prior", "wrong_prior"]):
        for col, cam in enumerate(CAMERAS):
            ax = axes[row, col]
            d = load(f"A_approach_{cam}_{v}")
            t, e = d["t"], 100 * rel_err(d)
            two_sig = 100 * 2 * d["r_sigma"] / d["r_est"]
            for k in range(25):
                ax.plot(t, e[k], color=MUTED, lw=0.6, alpha=0.5)
            env = np.nanmedian(two_sig, axis=0)
            ax.fill_between(t, -env, env, color=COLOR[v], alpha=0.18, lw=0, label="filter ±2σ (median over runs)")
            # where 95% of the ACTUAL errors fall (2.5th..97.5th percentile over runs)
            lo, hi = np.nanpercentile(e, [2.5, 97.5], axis=0)
            ax.plot(t, hi, color=INK, lw=1.5, ls="--", label="actual errors: central 95% of runs")
            ax.plot(t, lo, color=INK, lw=1.5, ls="--")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_ylim(-100, 100)
            ax.set_title(f"{CAM_LABEL[cam]} camera\n{LABEL[v]}", fontsize=10)   # 2 lines: longer labels
            if col == 0:
                ax.set_ylabel("range error / true range [%]")
            if row == 1:
                k = int(np.argmin(d["r_true"][0]))
                ax.set_xlabel(f"time [s]  (closest approach {d['r_true'][0, k]:.0f} m at t = {t[k]:.1f} s)")
    axes[0, 0].plot([], [], color=MUTED, lw=0.8, label="25 individual runs")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fins = [load(f"A_approach_{cam}_size_prior")["r_true"][0, -1] for cam in CAMERAS]
    fig.suptitle(f"Range error with the filter's ±2σ envelope, accelerating cameras (approach 200 m → 10 m → "
                 f"{min(fins):.0f}-{max(fins):.0f} m, 200 runs; simulation only)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "range_error_envelopes.png")
    plt.close(fig)


def binned_rms(d, edges):
    e, r = rel_err(d), d["r_true"]
    t_ok = d["t"][None, :] >= 2.0
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) & t_ok & ~np.isnan(e)
        out.append(np.sqrt(np.mean(e[m] ** 2)) if m.sum() > 50 else np.nan)
    return np.array(out)


def fig_rmse_vs_range():
    edges = np.array([10, 15, 20, 30, 40, 60, 80, 100, 130, 160, 200])
    mid = np.sqrt(edges[:-1] * edges[1:])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, cam in zip(axes, CAMERAS):
        for v in VARIANTS:
            ax.plot(mid, 100 * binned_rms(load(f"A_approach_{cam}_{v}"), edges), color=COLOR[v],
                    marker=MARKER[v], ms=5, label=LABEL[v])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
        ax.set_title(f"{CAM_LABEL[cam]} camera")
        ax.set_xlabel("true range [m]")
    axes[0].set_ylabel("RMS range error / true range [%]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Range RMSE vs true range, accelerating cameras (approach_recede drone, 200 runs, t ≥ 2 s; simulation only)")
    fig.tight_layout()
    fig.savefig(FIG / "rmse_vs_range.png")
    plt.close(fig)


def fig_nees_time():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, cam in zip(axes, CAMERAS):
        for v in VARIANTS:
            d = load(f"A_approach_{cam}_{v}")
            ax.plot(d["t"], np.nanmean(d["nees_range"], axis=0), color=COLOR[v], label=LABEL[v], lw=1.6)
        lo, hi = anees_bounds(1, d["nees_range"].shape[0])
        ax.axhspan(lo, hi, color=INK2, alpha=0.15, lw=0, label="95% bounds if consistent")
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 1e3)
        ax.set_title(f"{CAM_LABEL[cam]} camera")
        ax.set_xlabel("time [s]")
    axes[0].set_ylabel("average range NEES (1 dof; ideal 1)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Filter consistency over time, accelerating cameras: above the band = overconfident (200 runs; simulation only)")
    fig.tight_layout()
    fig.savefig(FIG / "nees_time.png")
    plt.close(fig)


def fig_uncertainty_vs_motion():
    """Bars: 68th percentile of |range error| (a robust stand-in for the actual 1-sigma error).
    Ticks: median sigma the filter reports. Honest filter: tick at the top of the bar."""
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.15
    x = np.arange(len(CAMERAS))
    for i, v in enumerate(VARIANTS):
        err68, sig = [], []
        for cam in CAMERAS:
            d = load(f"A_approach_{cam}_{v}")
            late = d["t"] >= 10
            err68.append(100 * np.nanpercentile(np.abs(rel_err(d)[:, late]), 68))
            sig.append(100 * np.nanmedian((d["r_sigma"] / d["r_est"])[:, late]))
        xs = x + (i - 2) * width
        ax.bar(xs, np.minimum(err68, 100), width * 0.9, color=COLOR[v], label=LABEL[v])
        ax.scatter(xs, sig, marker="_", s=300, color=INK, linewidths=2.5, zorder=3,
                   label="σ reported by the filter (median)" if i == 0 else None)
        for xx, r in zip(xs, err68):
            ax.text(xx, min(r, 100) + 1.5, f"{r:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [f"{CAM_LABEL[c]} camera" for c in CAMERAS])
    ax.set_ylim(0, 108)
    ax.set_ylabel("relative range error [%], t ≥ 10 s")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Actual error (bars: 68th percentile of |error|, ≈1σ) vs the σ the filter reports (ticks)\n"
                 "Honest filter: tick level with the top of its bar. Ticks far below = overconfident. Simulation only.")
    fig.tight_layout()
    fig.savefig(FIG / "uncertainty_vs_motion.png")
    plt.close(fig)


def fig_consistency():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    shades = {40: "#86b6ef", 80: "#2a78d6", 150: "#104281"}     # sequential blue: start range
    for ax, cam in zip(axes, ["weaving", "lat_surge_15"]):
        for r0, c in shades.items():
            d = load(f"E_model_truth_{cam}_{r0}m")
            ax.plot(d["t"], np.nanmean(d["nees_pos"], axis=0), color=c, label=f"start range {r0} m", lw=1.6)
        lo, hi = anees_bounds(3, d["nees_pos"].shape[0])
        ax.axhspan(lo, hi, color=INK2, alpha=0.15, lw=0, label="95% bounds if consistent")
        ax.axhline(3.0, color=INK2, lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{CAM_LABEL[cam]} camera, truth from the filter's own model")
        ax.set_xlabel("time [s]")
    axes[0].set_ylabel("average position NEES (3 dof; ideal 3)")
    axes[0].legend(fontsize=8)
    fig.suptitle("NEES check (200 runs): drone motion, size, pose drift and box noise all drawn from the filter's model")
    fig.tight_layout()
    fig.savefig(FIG / "consistency_model_truth.png")
    plt.close(fig)


# ------------------------------------------------------------------ new figures (accelerating grid only)
BAR_CAMS = ["lateral", "lat_surge_15", "straight", "surge_15", "weaving"]
ACCEL = {"lat_surge_15", "surge_15"}
# perfect-data figure: one hue per axis of motion (validated adjacent-pair palette, slots 1-3),
# line style = constant (dashed) vs accelerating (solid) as the secondary encoding
AXIS_COLOR = {"lateral": "#2a78d6", "lat_surge_15": "#2a78d6", "straight": "#eb6834", "surge_15": "#eb6834",
              "weaving": "#1baf7a"}


def fig_accel_vs_constant():
    old = json.loads((ROOT / "experiments" / "results_sim.json").read_text())["results"]
    new = json.loads((ROOT / "experiments" / "accel_grid" / "results_sim.json").read_text())["results"]
    summ = {cam: (new if cam in ACCEL else old)[f"A_approach_{cam}_size_prior"] for cam in BAR_CAMS}
    x = np.array([0, 1, 2.5, 3.5, 5])                      # pairs grouped, weaving on its own
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    panels = [("rms_rel_err_after10s", "RMS range error / true range [%], t ≥ 10 s", None),
              ("coverage_2sigma", "2σ coverage [%], t ≥ 2 s (ideal 95%)", 95)]
    for ax, (key, ylabel, ideal) in zip(axes, panels):
        vals = [100 * summ[c][key] for c in BAR_CAMS]
        for xx, c, val in zip(x, BAR_CAMS, vals):
            acc = c in ACCEL or c == "weaving"
            ax.bar(xx, val, 0.8, color=COLOR["size_prior"], alpha=1.0 if acc else 0.4,
                   edgecolor=SURFACE, linewidth=2)
            ax.text(xx, val + 1.5, f"{val:.1f}", ha="center", fontsize=8, color=INK2)
        if ideal is not None:
            ax.axhline(ideal, color=INK2, lw=1.2, ls="--")
            ax.text(x[-1] + 0.45, ideal + 1.5, "ideal 95%", ha="right", fontsize=8, color=INK2)
        ax.set_xticks(x, [CAM_LABEL[c].replace(", ", ",\n") for c in BAR_CAMS])
        ax.set_ylim(0, 105)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=COLOR["size_prior"], alpha=0.4, label="constant speed (4 m/s)"),
                            Patch(color=COLOR["size_prior"], label="accelerating (peak 4.1 m/s², incl. weaving)")],
                   fontsize=8, loc="upper right")
    axes[0].set_title("Range error (lower is better)")
    axes[1].set_title("Honesty of the reported σ (closer to 95% is better)")
    fig.suptitle("Constant-speed vs accelerating camera, size prior 0.35±0.15 m "
                 "(approach_recede drone, 200 runs; simulation only)")
    fig.tight_layout()
    fig.savefig(FIG / "accel_vs_constant.png")
    plt.close(fig)


def fig_perfect_data():
    from experiments.run_accel_grid import PERFECT_CAMERAS, perfect_runs
    runs = perfect_runs()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhspan(-2, 2, color=INK2, alpha=0.15, lw=0, label="±2% band")
    ax.axhline(0, color=INK2, lw=0.8)
    for cam in PERFECT_CAMERAS:
        r = runs[cam]
        e = 100 * r["rel_err"]
        tu = r["t_under_2pct"]
        when = "never under 2%" if tu is None else f"under 2% from t = {tu:.1f} s"
        ls = "-" if (cam in ACCEL or cam == "weaving") else "--"
        ax.plot(r["t"], e, color=AXIS_COLOR[cam], ls=ls, lw=2.0, label=f"{CAM_LABEL[cam]} — {when}")
        if tu is not None:
            k = int(round(tu / 0.1))
            ax.plot(r["t"][k], e[k], marker="o", ms=8, color=AXIS_COLOR[cam], markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=4)
    lo = min(np.nanmin(100 * r["rel_err"]) for r in runs.values())
    hi = max(np.nanmax(100 * r["rel_err"]) for r in runs.values())
    ax.set_ylim(lo - 5, max(hi, 2) + 5)
    ax.set_xlim(0, 40)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("signed range error / true range [%]")
    ax.plot([], [], marker="o", ms=8, color=MUTED, ls="none", label="gets and stays under 2%")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_title("No-noise test (true drone 0.55 m, prior 0.35±0.15 m, perfect detections and pose): "
                 "range error vs time\nDashed = constant-speed camera, solid = accelerating. Simulation only.")
    fig.tight_layout()
    fig.savefig(FIG / "perfect_data_error_vs_time.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_envelopes()
    fig_rmse_vs_range()
    fig_nees_time()
    fig_uncertainty_vs_motion()
    fig_consistency()
    fig_accel_vs_constant()
    fig_perfect_data()
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
