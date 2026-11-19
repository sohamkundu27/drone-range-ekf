"""Measure the real detector's box errors on the held-out DUT Anti-UAV test set, for the simulator.

For every test image: predict (conf >= --conf), keep `drone` predictions, and match them to
ground-truth drones greedily (highest confidence first, IoU >= 0.3; a low IoU threshold so
that small, slightly-off boxes still count as true positives). For each matched pair:
    centre error   du = u_pred - u_gt,  dv = v_pred - v_gt                 [px, 1920x1080 frame]
    width error    log(w_pred / w_gt)                                      [log ratio]
Results are binned by the TRUE box width w_gt. Per bin we report a robust sigma
(1.4826 x median absolute deviation, which ignores a few gross outliers), the plain std, the
bias, the fraction of errors beyond 3 robust sigmas (tail weight), and the recall
(matched GT / all GT in the bin, at this confidence threshold).

    python detector/measure_noise.py --weights runs/detect/drone4_bal/weights/best.pt

Writes experiments/detector_noise.json (read by sim/noise.py) and figures/detector_noise.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "yolo" / "dut_anti_uav" / "images" / "test"
EDGES = [0, 8, 12, 16, 24, 32, 48, 64, 96, 160, 10_000]     # bins of true box width [px]
MIN_MATCHED = 20                                             # bins with fewer pairs are not used


def robust_sigma(x):
    x = np.asarray(x)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "detector_noise.json"))
    ap.add_argument("--fig", default=str(ROOT / "figures" / "detector_noise.png"))
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    imgs = sorted(TEST.glob("*.jpg"))
    if args.max_images:
        imgs = imgs[: args.max_images]

    gt_w, matched = [], []          # all GT widths; (w_gt, du, dv, dlogw) per match
    for r in model.predict([str(p) for p in imgs], imgsz=args.imgsz, conf=args.conf, device=args.device,
                           stream=True, verbose=False, batch=8):
        h, w = r.orig_shape
        gts = []
        for line in Path(r.path.replace("/images/", "/labels/")).with_suffix(".txt").read_text().splitlines():
            c, cx, cy, bw, bh = map(float, line.split())
            if int(c) == 0:
                gts.append(np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
        gt_w += [g[2] - g[0] for g in gts]
        keep = r.boxes.cls.cpu().numpy() == 0
        preds = r.boxes.xyxy.cpu().numpy()[keep]
        order = np.argsort(-r.boxes.conf.cpu().numpy()[keep])
        used = set()
        for p in preds[order]:
            best, best_iou = None, args.iou
            for j, g in enumerate(gts):
                if j not in used and iou(p, g) >= best_iou:
                    best, best_iou = j, iou(p, g)
            if best is None:
                continue
            used.add(best)
            g = gts[best]
            wg, wp = g[2] - g[0], p[2] - p[0]
            matched.append((wg, (p[0] + p[2] - g[0] - g[2]) / 2, (p[1] + p[3] - g[1] - g[3]) / 2, np.log(wp / wg)))

    gt_w, M = np.array(gt_w), np.array(matched)
    bins = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        n_gt = int(((gt_w >= lo) & (gt_w < hi)).sum())
        m = M[(M[:, 0] >= lo) & (M[:, 0] < hi)] if len(M) else np.zeros((0, 4))
        row = {"w_lo": lo, "w_hi": hi, "n_gt": n_gt, "n_matched": int(len(m)),
               "recall": len(m) / n_gt if n_gt else None}
        if len(m) >= MIN_MATCHED:
            duv = np.concatenate([m[:, 1], m[:, 2]])       # pool both axes
            s_c, s_w = robust_sigma(duv), robust_sigma(m[:, 3])
            row.update({
                "w_median_px": float(np.median(m[:, 0])),
                "sigma_center_px": s_c, "std_center_px": float(np.std(duv)),
                "bias_u_px": float(np.mean(m[:, 1])), "bias_v_px": float(np.mean(m[:, 2])),
                "tail_frac_center_gt3sigma": float(np.mean(np.abs(duv - np.median(duv)) > 3 * s_c)),
                "sigma_logw": s_w, "std_logw": float(np.std(m[:, 3])), "bias_logw": float(np.median(m[:, 3])),
                "tail_frac_logw_gt3sigma": float(np.mean(np.abs(m[:, 3] - np.median(m[:, 3])) > 3 * s_w)),
            })
        bins.append(row)
    usable = [b for b in bins if "sigma_center_px" in b]
    out = {"weights": args.weights, "imgsz": args.imgsz, "conf": args.conf, "match_iou": args.iou,
           "test_set": "DUT Anti-UAV test (2,200 images, 1920x1080), never used for training",
           "n_gt": int(len(gt_w)), "n_matched": int(len(M)),
           "note": "sigma_* = 1.4826*MAD (robust); std_* include outliers. sim/noise.py uses the robust "
                   "sigmas and recall of bins with >= %d matches." % MIN_MATCHED,
           "bins": usable, "all_bins": bins}
    path = Path(args.out)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(usable, indent=1))
    print("wrote", path)
    plot(out, args.fig)


def plot(d, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(ROOT))
    from sim.noise import DetectorNoise
    default = DetectorNoise(use_json=False)
    b = d["bins"]
    w = np.array([x["w_median_px"] for x in b])
    ww = np.geomspace(3, 300, 100)
    blue, orange, ink2, grid = "#2a78d6", "#eb6834", "#52514e", "#e1e0d9"
    plt.rcParams.update({"axes.grid": True, "grid.color": grid, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("sigma_center_px", "std_center_px", default.sigma_uv, "box-centre error σ [px]"),
              ("sigma_logw", "std_logw", default.sigma_logw, "box-width error σ of log(w_pred/w_true)"),
              ("recall", None, default.p_detect, "detection rate (recall at conf ≥ %.2f)" % d["conf"])]
    for ax, (key, key_std, fdef, ylabel) in zip(axes, panels):
        ax.plot(w, [x[key] for x in b], color=blue, marker="o", label="measured (robust σ)" if key_std else "measured")
        if key_std:
            ax.plot(w, [x[key_std] for x in b], color=blue, marker="o", ls=":", label="measured (plain std, with outliers)")
        ax.plot(ww, [fdef(x) for x in ww], color=orange, ls="--", label="simulator default")
        ax.set_xscale("log")
        ax.set_xlabel("true box width [px, 1920×1080 frame]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle("Real detector errors on held-out DUT Anti-UAV test drones (matched true positives)", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)


if __name__ == "__main__":
    main()
