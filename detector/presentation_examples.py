"""Draw example detections for the presentation (writes presentation_assets/*.jpg).

    python -I detector/presentation_examples.py

* 3 good DUT Anti-UAV test predictions: the predicted box (red, with confidence) matches the
  ground-truth box (green) with IoU >= 0.7 and conf >= 0.7. One small, one medium and one large
  drone (by ground-truth box width), so the slide shows the size range.
* 2 Det-Fly (preliminary, partial set) frames where the drone was MISSED: no predicted drone
  box overlaps the ground truth (IoU >= 0.1) at conf >= 0.25. The ground-truth box is green; any
  drone boxes the detector put elsewhere in the frame (false positives) are drawn in red.

Small drones are hard to see in a full frame, so small boxes get a zoom inset of the area around
the ground-truth box (top-right corner). Images are resized to 1600 px wide.
Candidates are a fixed random sample (seed 0) of each test set, so the output is repeatable.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
GREEN, RED, WHITE = (60, 200, 60), (40, 40, 230), (255, 255, 255)
OUT_W = 1600


def read_gt(label_path, w, h):
    """YOLO label file -> list of drone boxes (x1, y1, x2, y2) in pixels."""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().split("\n"):
            p = line.split()
            if len(p) == 5 and p[0] == "0":                      # class 0 = drone
                cx, cy, bw, bh = (float(v) for v in p[1:])
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def predict_drones(model, img_path, conf, device):
    r = model.predict(str(img_path), imgsz=1280, conf=conf, device=device, verbose=False)[0]
    keep = r.boxes.cls.cpu().numpy() == 0
    return list(zip(r.boxes.xyxy.cpu().numpy()[keep], r.boxes.conf.cpu().numpy()[keep]))


def draw(img, gt, pred, title):
    """Resize to OUT_W, draw boxes, add a zoom inset around the ground truth (small boxes only)
    and a title bar."""
    s = OUT_W / img.shape[1]
    out = cv2.resize(img, (OUT_W, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = gt
    if (x2 - x1) * s > 120:                                      # big enough to see: no inset
        box_full(out, gt, pred, s)
        return title_bar(out, title)
    # zoom inset from the ORIGINAL image (sharper than zooming the resized one)
    half = max(3 * (x2 - x1), 3 * (y2 - y1), 60)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax1, ay1 = int(max(0, cx - half)), int(max(0, cy - half))
    ax2, ay2 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
    crop = img[ay1:ay2, ax1:ax2].copy()
    z = 400 / max(crop.shape[:2])
    crop = cv2.resize(crop, (round(crop.shape[1] * z), round(crop.shape[0] * z)), interpolation=cv2.INTER_CUBIC)

    box(crop, gt, GREEN, z, (ax1, ay1), 4)                      # green thicker, under red
    for b, c in pred:
        if iou(gt, b) > 0:                                       # only boxes inside the crop
            p1 = box(crop, b, RED, z, (ax1, ay1), 2)
            cv2.putText(crop, f"drone {c:.2f}", (p1[0], max(15, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    box_full(out, gt, pred, s)
    # paste the inset top-right with a white frame, and a line from the box to the inset
    ih, iw = crop.shape[:2]
    X, Y = OUT_W - iw - 12, 52
    out[Y:Y + ih, X:X + iw] = crop
    cv2.rectangle(out, (X - 2, Y - 2), (X + iw + 1, Y + ih + 1), WHITE, 2)
    cv2.line(out, (round(cx * s), round(cy * s)), (X, Y + ih // 2), WHITE, 1)
    cv2.rectangle(out, (X, Y), (X + 62, Y + 30), (0, 0, 0), -1)
    cv2.putText(out, "zoom", (X + 6, Y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    return title_bar(out, title)


def box(im, b, color, scale, off, t):
    """Draw box b (original-image pixels) on image im, which is scaled by `scale` and offset by `off`."""
    p1 = (round((b[0] - off[0]) * scale), round((b[1] - off[1]) * scale))
    p2 = (round((b[2] - off[0]) * scale), round((b[3] - off[1]) * scale))
    cv2.rectangle(im, p1, p2, color, t)
    return p1


def box_full(out, gt, pred, s):
    """Boxes on the resized full frame, padded a little so tiny boxes stay visible."""
    x1, y1, x2, y2 = gt
    box(out, (x1 - 4 / s, y1 - 4 / s, x2 + 4 / s, y2 + 4 / s), GREEN, s, (0, 0), 3)
    for b, c in pred:
        p1 = box(out, (b[0] - 8 / s, b[1] - 8 / s, b[2] + 8 / s, b[3] + 8 / s), RED, s, (0, 0), 2)
        if iou(gt, b) < 0.1:                                     # false positive: label it on the frame
            cv2.putText(out, f"drone {c:.2f}", (min(p1[0], OUT_W - 120), max(55, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)


def title_bar(out, title):
    cv2.rectangle(out, (0, 0), (OUT_W, 40), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/drone4_bal/weights/best.pt"))
    ap.add_argument("--out", default=str(ROOT / "presentation_assets"))
    ap.add_argument("--n_candidates", type=int, default=300)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    model = YOLO(args.weights)
    rng = np.random.default_rng(0)

    # ---- DUT: good predictions ----
    d = ROOT / "data/yolo/dut_anti_uav"
    imgs = sorted((d / "images/test").glob("*.jpg"))
    good = []
    for p in rng.choice(imgs, args.n_candidates, replace=False):
        img = cv2.imread(str(p))
        gt = read_gt(d / "labels/test" / (p.stem + ".txt"), img.shape[1], img.shape[0])
        if len(gt) != 1:
            continue
        preds = predict_drones(model, p, 0.25, args.device)
        if len(preds) == 1 and preds[0][1] >= 0.7 and iou(gt[0], preds[0][0]) >= 0.7:
            good.append((gt[0][2] - gt[0][0], p, gt[0], preds, iou(gt[0], preds[0][0])))
    good.sort(key=lambda g: g[0])                                # by ground-truth width in px
    picks = [good[len(good) // 10], good[len(good) // 2], good[(9 * len(good)) // 10]]
    for name, (gw, p, gt, preds, io) in zip(("small", "medium", "large"), picks):
        img = cv2.imread(str(p))
        title = (f"DUT Anti-UAV test {p.name} ({img.shape[1]}x{img.shape[0]}): drone {gw:.0f} px wide, "
                 f"conf {preds[0][1]:.2f}, IoU {io:.2f}   green = truth, red = prediction")
        f = out_dir / f"dut_good_{name}.jpg"
        cv2.imwrite(str(f), draw(img, gt, preds, title), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", f, f"({len(good)} good candidates)")

    # ---- Det-Fly: missed drones ----
    d = ROOT / "data/yolo/detfly"
    imgs = sorted((d / "images/test").glob("*.jpg"))
    missed = []
    for p in sorted(rng.choice(imgs, args.n_candidates, replace=False)):
        img = cv2.imread(str(p))
        gt = read_gt(d / "labels/test" / (p.stem + ".txt"), img.shape[1], img.shape[0])
        if len(gt) != 1:
            continue
        preds = predict_drones(model, p, 0.25, args.device)
        if not any(iou(gt[0], b) >= 0.1 for b, _ in preds):
            missed.append((p, gt[0], preds))
    # two misses far apart in the sequence (first and last third), not neighbouring frames
    picks = [missed[len(missed) // 6], missed[(5 * len(missed)) // 6]]
    for i, (p, gt, preds) in enumerate(picks, 1):
        img = cv2.imread(str(p))
        s = 1280 / img.shape[1]
        title = (f"Det-Fly (PRELIMINARY) {p.name}: drone MISSED, {gt[2] - gt[0]:.0f} px wide "
                 f"({(gt[2] - gt[0]) * s:.0f} px at 1280 input)   green = truth"
                 + (", red = false drone box" if preds else ""))
        f = out_dir / f"detfly_missed_{i}.jpg"
        cv2.imwrite(str(f), draw(img, gt, preds, title), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", f, f"({len(missed)} misses among sampled frames; {len(preds)} false drone boxes in this frame)")


if __name__ == "__main__":
    main()
