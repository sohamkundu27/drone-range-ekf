"""Example detections for the v2 slides, drawn with the NEW model (writes presentation_assets_v2/*.jpg).

    python detector/new_presentation_examples.py --new runs/s_detfly_new/weights/best.pt

* dut_good_small/medium/large.jpg: the SAME three DUT test images as presentation_assets/ — found by
  re-running the original selection of detector/presentation_examples.py (seed 0, 300 candidates,
  old s weights) — but drawn with the NEW model's predictions.
* detfly_missed_1/2.jpg: Det-Fly 020 frames (held-out, never trained on) that the NEW model misses
  (no drone box with IoU >= 0.1 at conf >= 0.25), from a fixed random sample (seed 0) of 300 frames,
  one from the first and one from the last sixth of the misses. The title says what the OLD model did
  on the same frame.
Drawing (boxes, zoom inset, 1600 px wide) reuses detector/presentation_examples.py unchanged.
Writes presentation_assets_v2/examples.json with the chosen files and per-frame results.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))
from presentation_examples import draw, iou, predict_drones, read_gt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", default=str(ROOT / "runs/drone4_bal_backup.aVl2ZP/best.pt"))
    ap.add_argument("--out", default=str(ROOT / "presentation_assets_v2"))
    ap.add_argument("--n_candidates", type=int, default=300)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    old, new = YOLO(args.old), YOLO(args.new)
    info = {"new_weights": args.new, "old_weights": args.old, "dut": {}, "detfly": {}}

    # ---- DUT: reproduce the original picks (old model, identical RNG use), draw with the new model ----
    rng = np.random.default_rng(0)
    d = ROOT / "data/yolo/dut_anti_uav"
    imgs = sorted((d / "images/test").glob("*.jpg"))
    good = []
    for p in rng.choice(imgs, args.n_candidates, replace=False):
        img = cv2.imread(str(p))
        gt = read_gt(d / "labels/test" / (p.stem + ".txt"), img.shape[1], img.shape[0])
        if len(gt) != 1:
            continue
        preds = predict_drones(old, p, 0.25, args.device)
        if len(preds) == 1 and preds[0][1] >= 0.7 and iou(gt[0], preds[0][0]) >= 0.7:
            good.append((gt[0][2] - gt[0][0], p, gt[0], preds, iou(gt[0], preds[0][0])))
    good.sort(key=lambda g: g[0])
    picks = [good[len(good) // 10], good[len(good) // 2], good[(9 * len(good)) // 10]]
    for name, (gw, p, gt, old_preds, old_io) in zip(("small", "medium", "large"), picks):
        img = cv2.imread(str(p))
        preds = predict_drones(new, p, 0.25, args.device)
        best = max(preds, key=lambda bc: iou(gt, bc[0])) if preds else None
        io = iou(gt, best[0]) if best else 0.0
        title = (f"NEW model, DUT test {p.name}: drone {gw:.0f} px wide, "
                 + (f"conf {best[1]:.2f}, IoU {io:.2f}" if best else "MISSED")
                 + f" (old: {old_preds[0][1]:.2f}/{old_io:.2f})  green=truth red=pred")
        f = out_dir / f"dut_good_{name}.jpg"
        cv2.imwrite(str(f), draw(img, gt, preds, title), [cv2.IMWRITE_JPEG_QUALITY, 85])
        info["dut"][name] = {"image": p.name, "gt_width_px": gw,
                             "new": {"conf": float(best[1]) if best else None, "iou": io, "n_drone_boxes": len(preds)},
                             "old": {"conf": float(old_preds[0][1]), "iou": old_io}}
        print("wrote", f)

    # ---- Det-Fly 020: misses of the NEW model ----
    imgs = [Path(p) for p in (ROOT / "data/yolo/new_detfly_test020.txt").read_text().split()]
    rng = np.random.default_rng(0)
    missed, n_checked = [], 0
    for p in sorted(rng.choice(imgs, args.n_candidates, replace=False)):
        img = cv2.imread(str(p))
        gt = read_gt(Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt"), img.shape[1], img.shape[0])
        if len(gt) != 1:
            continue
        n_checked += 1
        preds = predict_drones(new, p, 0.25, args.device)
        if not any(iou(gt[0], b) >= 0.1 for b, _ in preds):
            missed.append((p, gt[0], preds))
    info["detfly"]["sampled_frames"] = n_checked
    info["detfly"]["new_missed_in_sample"] = len(missed)
    picks = [missed[len(missed) // 6], missed[(5 * len(missed)) // 6]] if len(missed) >= 2 else missed
    for i, (p, gt, preds) in enumerate(picks, 1):
        img = cv2.imread(str(p))
        op = predict_drones(old, p, 0.25, args.device)
        ohit = [c for b, c in op if iou(gt, b) >= 0.1]
        old_txt = f"old model: detected (conf {max(ohit):.2f})" if ohit else "old model: also missed"
        s = 1280 / img.shape[1]
        title = (f"Det-Fly 020 {p.name}: NEW model MISSED, {gt[2] - gt[0]:.0f} px wide "
                 f"({(gt[2] - gt[0]) * s:.0f} px at 1280); {old_txt}")
        f = out_dir / f"detfly_missed_{i}.jpg"
        cv2.imwrite(str(f), draw(img, gt, preds, title), [cv2.IMWRITE_JPEG_QUALITY, 85])
        info["detfly"][f"missed_{i}"] = {"image": p.name, "gt_width_px": gt[2] - gt[0],
                                         "new_false_drone_boxes": len(preds), "old_detected": bool(ohit),
                                         "old_conf": max(ohit) if ohit else None}
        print("wrote", f)
    (out_dir / "examples.json").write_text(json.dumps(info, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
