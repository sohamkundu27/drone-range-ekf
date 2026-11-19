"""False 'drone' detections on images that contain NO drone, only birds / airplanes / helicopters.

For every held-out test image of COCO (bird, airplane) and Open Images (bird, airplane,
helicopter) we run the detector and count detections labelled `drone` above a confidence
threshold. Each image is grouped by the classes it contains (an image with birds -> "bird").

    python detector/false_positives.py --weights runs/detect/drone4_bal/weights/best.pt

Writes experiments/false_positives.json:
  per group and threshold: images, images with >= 1 false drone, false drones per image,
  and how many of those false drones sit on a labelled bird/aircraft (IoU >= 0.1) vs background.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "data" / "yolo"
NAMES = ["drone", "bird", "airplane", "helicopter"]


def read_gt(label_path, w, h):
    """YOLO label file -> list of (cls, x1, y1, x2, y2) in pixels."""
    out = []
    for line in Path(label_path).read_text().splitlines():
        if line.strip():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            out.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--thresholds", default="0.25,0.5")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all (for quick tests)")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "false_positives.json"))
    args = ap.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]

    from ultralytics import YOLO
    model = YOLO(args.weights)

    stats = {}
    for ds in ("coco", "openimages"):
        imgs = sorted((YOLO_DIR / ds / "images" / "test").glob("*.jpg"))
        if args.max_images:
            imgs = imgs[: args.max_images]
        for r in model.predict([str(p) for p in imgs], imgsz=args.imgsz, conf=min(thresholds),
                               device=args.device, stream=True, verbose=False, batch=8):
            h, w = r.orig_shape
            gt = read_gt(Path(r.path.replace("/images/", "/labels/")).with_suffix(".txt"), w, h)
            gt_classes = {NAMES[g[0]] for g in gt}
            assert "drone" not in gt_classes, r.path
            groups = [f"{ds}/{c}" for c in gt_classes] + [f"all/{c}" for c in gt_classes] + ["all/any"]
            boxes = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for t in thresholds:
                fp = [b for b, c, s in zip(boxes, cls, conf) if c == 0 and s >= t]
                on_object = sum(any(iou(b, g[1:]) >= 0.1 for g in gt) for b in fp)
                for g in groups:
                    s = stats.setdefault(g, {}).setdefault(str(t), {"images": 0, "images_with_fp": 0,
                                                                    "false_drones": 0, "on_labelled_object": 0})
                    s["images"] += 1
                    s["images_with_fp"] += int(len(fp) > 0)
                    s["false_drones"] += len(fp)
                    s["on_labelled_object"] += on_object
    for g in stats.values():
        for s in g.values():
            s["frac_images_with_fp"] = s["images_with_fp"] / s["images"]
            s["false_drones_per_image"] = s["false_drones"] / s["images"]
    out = Path(args.out)
    out.write_text(json.dumps({"weights": args.weights, "imgsz": args.imgsz, "groups": stats}, indent=1))
    for g in sorted(stats):
        print(g, {t: f"{100 * s['frac_images_with_fp']:.1f}% imgs, {s['false_drones_per_image']:.3f}/img"
                  for t, s in stats[g].items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
