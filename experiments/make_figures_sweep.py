"""Figures for the sweep-in-place study (experiments/run_sweep.py) in figures/sweep/. SIMULATION ONLY.

    .venv/bin/python -I experiments/make_figures_sweep.py

Same style, palette and helpers as experiments/make_figures.py (imported, not copied). Runs are read from
runs/sim_sweep/ and, for the weaving reference panels, runs/sim/ (grid A). Samples with true range > 200 m
are masked exactly as in the scoring (run_sweep.range_masked).
  rmse_vs_range.png, range_error_envelopes.png, nees_time.png, uncertainty_vs_motion.png : group G
      (sweep +/-15 m, inbound, five variants), each next to the existing weaving camera
  width_curves.png        : RMS vs true range for hover, sweep +/-5, 10, 15, 30 (H + I + G, size_prior)
  behaviour_table.png     : the behaviour table of experiments/sweep_tables.md
  orbit_error_time.png    : orbit target, range error over time, the dip around t = 16 s
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.make_figures as mf  # noqa: E402  (sets the shared rcParams on import)
from experiments.make_figures import (COLOR, GRID, INK, INK2, LABEL, MARKER, MUTED, SURFACE, VARIANTS,  # noqa: E402
                                      binned_rms, plt, rel_err)
from experiments.run_sweep import BUCKETS, OUT_NPZ, range_masked  # noqa: E402
from ekf.nees import anees_bounds  # noqa: E402
from sim.trajectories import CAM_START, ORBIT_R, ORBIT_SPEED, make_scenario  # noqa: E402

FIG = ROOT / "figures" / "sweep"
PANELS = [("weaving camera (existing, grid A)", lambda v: load_main(f"A_approach_weaving_{v}")),
          ("sweep ±15 m in place (grid G)", lambda v: load(f"G_approach_sweep15_{v}"))]
WIDTH_RUNS = [("hover", "H_approach_hover_size_prior"), ("sweep ±5 m", "I_approach_sweep5_size_prior"),
              ("sweep ±10 m", "I_approach_sweep10_size_prior"), ("sweep ±15 m", "G_approach_sweep15_size_prior"),
              ("sweep ±30 m", "I_approach_sweep30_size_prior")]
# hover = neutral; the sweep widths = one hue, light -> dark (an ordered quantity)
WIDTH_COLOR = [MUTED, "#9fc4f2", "#5b97e3", "#2a78d6", "#104281"]
WIDTH_MARKER = ["x", "o", "s", "^", "D"]


def as_dict(z):
    return {k: z[k] for k in z.files}


def load(name):
    return range_masked(as_dict(np.load(OUT_NPZ / f"{name}.npz")))


def load_main(name):
    return range_masked(as_dict(np.load(mf.NPZ / f"{name}.npz")))


def ca_time(d):
    """Time of closest approach (median over runs of the true range minimum)."""
    return float(np.median(d["t"][np.nanargmin(d["r_true"], axis=1)]))


def fig_envelopes():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    for row, v in enumerate(["size_prior", "wrong_prior"]):
        for col, (title, get) in enumerate(PANELS):
            ax = axes[row, col]
            d = get(v)
            t, e = d["t"], 100 * rel_err(d)
            two_sig = 100 * 2 * d["r_sigma"] / d["r_est"]
            for k in range(25):
                ax.plot(t, e[k], color=MUTED, lw=0.6, alpha=0.5)
            env = np.nanmedian(two_sig, axis=0)
            ax.fill_between(t, -env, env, color=COLOR[v], alpha=0.18, lw=0, label="filter ±2σ (median over runs)")
            lo, hi = np.nanpercentile(e, [2.5, 97.5], axis=0)
            ax.plot(t, hi, color=INK, lw=1.5, ls="--", label="actual errors: central 95% of runs")
            ax.plot(t, lo, color=INK, lw=1.5, ls="--")
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_ylim(-100, 100)
            ax.set_title(f"{title} — {LABEL[v]}", fontsize=10)
            if col == 0:
                ax.set_ylabel("range error / true range [%]")
            if row == 1:
                ax.set_xlabel(f"time [s]  (closest approach {np.nanmin(d['r_true']):.0f}–10 m at t ≈ {ca_time(d):.0f} s)")
    axes[0, 0].plot([], [], color=MUTED, lw=0.8, label="25 individual runs")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Range error with the filter's ±2σ envelope, inbound target 200 m → 10 m (200 runs; simulation only)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "range_error_envelopes.png")
    plt.close(fig)


EDGES = np.array([10, 15, 20, 30, 40, 60, 80, 100, 130, 160, 200])
MID = np.sqrt(EDGES[:-1] * EDGES[1:])


def log_range_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    ax.set_yticks([2, 5, 10, 20, 50, 100, 200, 500], ["2", "5", "10", "20", "50", "100", "200", "500"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("true range [m]")


def fig_rmse_vs_range():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, get) in zip(axes, PANELS):
        for v in VARIANTS:
            ax.plot(MID, 100 * binned_rms(get(v), EDGES), color=COLOR[v], marker=MARKER[v], ms=5, label=LABEL[v])
        log_range_axes(ax)
        ax.set_ylim(2, 1000)
        ax.set_title(title)
    axes[0].set_ylabel("RMS range error / true range [%]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Range RMSE vs true range (inbound target, 200 runs, t ≥ 2 s; simulation only)")
    fig.tight_layout()
    fig.savefig(FIG / "rmse_vs_range.png")
    plt.close(fig)


def fig_nees_time():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, get) in zip(axes, PANELS):
        for v in VARIANTS:
            d = get(v)
            ax.plot(d["t"], np.nanmean(d["nees_range"], axis=0), color=COLOR[v], label=LABEL[v], lw=1.6)
        lo, hi = anees_bounds(1, d["nees_range"].shape[0])
        ax.axhspan(lo, hi, color=INK2, alpha=0.15, lw=0, label="95% bounds if consistent")
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 1e3)
        ax.set_title(title)
        ax.set_xlabel("time [s]")
    axes[0].set_ylabel("average range NEES (1 dof; ideal 1)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Filter consistency over time, inbound target: above the band = overconfident (200 runs; simulation only)")
    fig.tight_layout()
    fig.savefig(FIG / "nees_time.png")
    plt.close(fig)


def err68_sigma(d):
    late = d["t"] >= 10
    return (100 * np.nanpercentile(np.abs(rel_err(d)[:, late]), 68),
            100 * np.nanmedian((d["r_sigma"] / d["r_est"])[:, late]))


def fig_uncertainty_vs_motion():
    """Bars: 68th percentile of |range error| (≈ actual 1σ). Ticks: median σ the filter reports."""
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.15
    x = np.arange(len(PANELS))
    for i, v in enumerate(VARIANTS):
        vals = [err68_sigma(get(v)) for _, get in PANELS]
        err68, sig = [a for a, _ in vals], [b for _, b in vals]
        xs = x + (i - 2) * width
        ax.bar(xs, np.minimum(err68, 100), width * 0.9, color=COLOR[v], label=LABEL[v])
        ax.scatter(xs, sig, marker="_", s=300, color=INK, linewidths=2.5, zorder=3,
                   label="σ reported by the filter (median)" if i == 0 else None)
        for xx, r in zip(xs, err68):
            ax.text(xx, min(r, 100) + 1.5, f"{r:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [t for t, _ in PANELS])
    ax.set_ylim(0, 40)
    ax.set_ylabel("relative range error [%], t ≥ 10 s")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Actual error (bars: 68th percentile of |error|, ≈1σ) vs the σ the filter reports (ticks)\n"
                 "Inbound target. Honest filter: tick level with the top of its bar. Simulation only.")
    fig.tight_layout()
    fig.savefig(FIG / "uncertainty_vs_motion.png")
    plt.close(fig)


def fig_width_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    summ = {n: json.loads((OUT_NPZ / f"{n}.json").read_text()) for _, n in WIDTH_RUNS}
    for (lab, n), c, m in zip(WIDTH_RUNS, WIDTH_COLOR, WIDTH_MARKER):
        axes[0].plot(MID, 100 * binned_rms(load(n), EDGES), color=c, marker=m, ms=6, label=lab)
    log_range_axes(axes[0])
    axes[0].set_ylim(4, 100)
    axes[0].set_ylabel("RMS range error / true range [%]")
    axes[0].set_title("RMS error vs true range (t ≥ 2 s)")
    axes[0].legend(fontsize=8, loc="upper left")
    x = np.arange(len(WIDTH_RUNS))
    rms = [100 * summ[n]["rms_rel_err_after10s"] for _, n in WIDTH_RUNS]
    cov = [100 * summ[n]["coverage_2sigma"] for _, n in WIDTH_RUNS]
    axes[1].bar(x, rms, 0.6, color=WIDTH_COLOR)
    for xx, r, cv in zip(x, rms, cov):
        axes[1].text(xx, r + 1, f"{r:.1f}%\n(2σ cov. {cv:.0f}%)", ha="center", fontsize=8, color=INK2)
    axes[1].set_xticks(x, [lab.replace("sweep ", "±").replace("±±", "±") for lab, _ in WIDTH_RUNS])
    axes[1].set_ylim(0, max(rms) * 1.25)
    axes[1].set_ylabel("RMS range error / true range [%], t ≥ 10 s")
    axes[1].set_title("Overall RMS error by sweep half-width")
    axes[1].grid(axis="x", visible=False)
    fig.suptitle("Sweep width (same peak acceleration 4.1 m/s²; period 12·√(A/15) s), inbound target, "
                 "size prior, 200 runs; simulation only", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "width_curves.png")
    plt.close(fig)


def fig_behaviour_table():
    rows = [("inbound", "G_approach_sweep15_size_prior"), ("crossing", "J_constant_velocity_sweep15_size_prior"),
            ("gentle turn", "J_gentle_turn_sweep15_size_prior"), ("dash", "J_dash_sweep15_size_prior"),
            ("jink", "J_jink_sweep15_size_prior"), ("orbit", "J_orbit_sweep15_size_prior")]
    S = {n: json.loads((OUT_NPZ / f"{n}.json").read_text()) for _, n in rows}
    cols = ["RMS err\nt ≥ 10 s"] + [f"{b[:-1]} m" for b in BUCKETS] + ["2σ\ncoverage", "actual σ\n(68th pct)",
                                                                       "reported\nσ", "NaN\nruns"]
    V = np.full((len(rows), 1 + len(BUCKETS)), np.nan)
    text = []
    for i, (_, n) in enumerate(rows):
        s = S[n]
        V[i, 0] = 100 * s["rms_rel_err_after10s"]
        for j, b in enumerate(BUCKETS):
            if b in s["by_range"]:
                V[i, 1 + j] = 100 * s["by_range"][b]["rms_rel_err"]
        text.append([("-" if np.isnan(x) else f"{x:.1f}%") for x in V[i]] +
                    [f"{100 * s['coverage_2sigma']:.0f}%", f"{100 * s['extra']['p68_abs_rel_err_after10s']:.1f}%",
                     f"{100 * s['median_reported_rel_sigma_after10s']:.1f}%", str(s["nan"]["nan_runs"])])
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.grid(False)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    cmap = plt.get_cmap("Blues")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if j < V.shape[1] and not np.isnan(V[i, j]):         # RMS cells: one hue, darker = larger error (log)
                f = np.clip(np.log10(V[i, j] / 3) / np.log10(100 / 3), 0, 1)
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, color=cmap(0.08 + 0.8 * f), lw=0))
                dark = f > 0.55
            else:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, color=GRID if j >= V.shape[1] else SURFACE,
                                           lw=0, alpha=0.45))
                dark = False
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=9, color=SURFACE if dark else INK)
    ax.set_xticks(range(len(cols)), cols, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [r for r, _ in rows], fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(len(BUCKETS) + 0.5, color=MUTED, lw=0.8)
    ax.axvline(0.5, color=MUTED, lw=0.8)
    ax.set_title("Target behaviour vs sweep-in-place observer (±15 m), size prior 0.35 ± 0.15 m, 200 runs, true range "
                 "≤ 200 m\nRMS range error / true range per range bucket; '-' = never in that bucket. Ideal coverage "
                 "95%. Simulation only.", fontsize=9.5, pad=8, y=-0.2)
    fig.tight_layout()
    fig.savefig(FIG / "behaviour_table.png")
    plt.close(fig)


def fig_orbit_error_time():
    d = load("J_orbit_sweep15_size_prior")
    t, e = d["t"], 100 * rel_err(d)
    scn = make_scenario("orbit", "sweep")
    los = scn["drone_p"] - scn["cam_p"]
    los_h = los[:, :2] / np.linalg.norm(los[:, :2], axis=1, keepdims=True)
    perp = 100 * np.abs(los_h[:, 1])                     # |sin| of angle between the sweep axis (x) and the LOS
    t_align = np.pi / 2 / (ORBIT_SPEED / ORBIT_R)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for k in range(25):
        ax.plot(t, e[k], color=MUTED, lw=0.6, alpha=0.5)
    env = np.nanmedian(100 * 2 * d["r_sigma"] / d["r_est"], axis=0)
    ax.fill_between(t, -env, env, color=COLOR["size_prior"], alpha=0.18, lw=0, label="filter ±2σ (median over runs)")
    lo, hi = np.nanpercentile(e, [2.5, 97.5], axis=0)
    ax.plot(t, hi, color=INK, lw=1.5, ls="--", label="actual errors: central 95% of runs")
    ax.plot(t, lo, color=INK, lw=1.5, ls="--")
    rms_t = np.sqrt(np.nanmean(e ** 2, axis=0))
    ax.plot(t, rms_t, color=COLOR["size_prior"], lw=2, label="RMS error over runs")
    ax.plot([], [], color=MUTED, lw=0.8, label="25 individual runs")
    ax.axhline(0, color=INK2, lw=0.8)
    for a in (ax, ax2):
        a.axvline(t_align, color=INK2, lw=1, ls=":")
    ax.text(t_align + 0.4, 37, f"target on the sweep axis\n(t = {t_align:.1f} s)", fontsize=8, color=INK2, va="top")
    ax.set_ylim(-40, 40)
    ax.set_ylabel("range error / true range [%]")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_title("Orbit target (80 m circle at 8 m/s around the sweep centre), sweep ±15 m, size prior, 200 runs\n"
                 "Simulation only.", fontsize=10)
    ax2.plot(t, perp, color=INK2, lw=1.6)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("sweep motion across\nline of sight [%]", fontsize=9)
    ax2.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(FIG / "orbit_error_time.png")
    plt.close(fig)
    win = lambda a, b: float(np.sqrt(np.nanmean(e[:, (t >= a) & (t < b)] ** 2)))     # noqa: E731
    k = int(np.nanargmax(np.where(t >= 10, rms_t, np.nan)))
    return {"t_on_sweep_axis_s": round(t_align, 2), "rms_pct_5_12s": round(win(5, 12), 2),
            "rms_pct_14_20s": round(win(14, 20), 2), "rms_pct_25_40s": round(win(25, 40), 2),
            "peak_rms_after_10s_pct": round(float(rms_t[k]), 2), "peak_rms_time_s": round(float(t[k]), 1),
            "median_2sigma_pct_at_peak": round(float(env[k]), 2)}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_envelopes()
    fig_rmse_vs_range()
    fig_nees_time()
    fig_uncertainty_vs_motion()
    fig_width_curves()
    fig_behaviour_table()
    orbit = fig_orbit_error_time()
    (FIG / "orbit_dip.json").write_text(json.dumps(orbit, indent=1) + "\n")
    print("orbit dip:", orbit)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
