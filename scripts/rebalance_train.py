"""Build a rebalanced training list so drone images are about half of the training set.

The full union (drone4.yaml) has 5,200 DUT drone images but 13,389 COCO/Open Images
negatives (bird, airplane, helicopter), so drones would be only ~28% of what the model sees.
We keep ALL DUT training images and subsample the negatives to a fixed quota per class.

Which negatives to keep? The confusers that matter for an anti-drone camera are SMALL
birds/aircraft against SKY, not close-up photos of parrots. So each negative image gets a
score that prefers:
  1. small boxes   : median box size relative to the image (from the label file), and
  2. sky-like background: fraction of pixels OUTSIDE the boxes with a low image gradient
     (sky is smooth; trees, buildings and indoor scenes are textured).
score = rank(small boxes) + rank(smooth background); we keep the best-scoring images
of each class up to its quota. This is a heuristic, not a sky segmenter.

Outputs (git-ignored, in data/yolo/):
    drone4_bal_train.txt   one absolute image path per line (Ultralytics accepts a .txt list)
    drone4_bal.yaml        train = that list, val/test = the same splits as drone4.yaml
    drone4_bal_stats.json  what was selected

    python -I scripts/rebalance_train.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLASSES, YOLO  # noqa: E402

# Negatives to keep, per primary class (the class with the most boxes in the image).
# 2000 + 1800 + 1200 = 5000 negatives vs 5,197 drone images -> drones ~51% of training images.
QUOTA = {1: 2000, 2: 1800, 3: 1200}   # bird, airplane, helicopter
NEG_DATASETS = ["coco", "openimages"]


def read_label(lbl_path):
    rows = [l.split() for l in lbl_path.read_text().splitlines() if l.strip()]
    return [(int(r[0]), *map(float, r[1:5])) for r in rows]


def smooth_background_fraction(args):
    """Fraction of non-box pixels whose gradient magnitude is small (sky-like)."""
    img_path, boxes = args
    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 0.0
    s = 256 / max(g.shape)                       # work at 256 px on the long side (fast)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    h, w = g.shape
    mask = np.ones((h, w), bool)                 # True = background pixel
    for _, cx, cy, bw, bh in boxes:              # blank out the labelled objects
        x1, x2 = int((cx - bw / 2) * w), int(np.ceil((cx + bw / 2) * w))
        y1, y2 = int((cy - bh / 2) * h), int(np.ceil((cy + bh / 2) * h))
        mask[max(0, y1):y2, max(0, x1):x2] = False
    if mask.sum() < 100:
        return 0.0
    return float((mag[mask] < 20.0).mean())      # 20 = gentle threshold on 8-bit Sobel output


def main():
    src = yaml.safe_load(open(YOLO / "drone4.yaml"))

    # 1) all drone (DUT) training images are kept
    drone_imgs = sorted((YOLO / "dut_anti_uav" / "images" / "train").glob("*.jpg"))

    # 2) collect negative candidates with their primary class and box size
    cands = []
    for ds in NEG_DATASETS:
        for ip in sorted((YOLO / ds / "images" / "train").glob("*.jpg")):
            boxes = read_label(Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt"))
            if not boxes:
                continue
            cls = np.bincount([b[0] for b in boxes], minlength=4).argmax()
            rel_size = float(np.median([np.sqrt(b[3] * b[4]) for b in boxes]))  # sqrt(w*h), normalised
            cands.append({"img": ip, "cls": int(cls), "rel_size": rel_size, "boxes": boxes})

    # 3) sky-likeness of the background (parallel, reads every candidate image once)
    with ProcessPoolExecutor() as ex:
        smooth = list(ex.map(smooth_background_fraction, [(c["img"], c["boxes"]) for c in cands], chunksize=64))
    for c, s in zip(cands, smooth):
        c["smooth"] = s

    # 4) per class: rank by small size and by smooth background, keep the best `quota`
    keep, stats = [], {"drone_images": len(drone_imgs), "per_class": {}}
    for cls, quota in QUOTA.items():
        cs = [c for c in cands if c["cls"] == cls]
        size_rank = np.argsort(np.argsort([c["rel_size"] for c in cs]))      # 0 = smallest box
        smooth_rank = np.argsort(np.argsort([-c["smooth"] for c in cs]))     # 0 = smoothest background
        order = np.argsort(size_rank + smooth_rank, kind="stable")
        chosen = [cs[i] for i in order[:quota]]
        keep += chosen
        stats["per_class"][CLASSES[cls]] = {
            "available": len(cs), "kept": len(chosen),
            "median_rel_size_all": float(np.median([c["rel_size"] for c in cs])),
            "median_rel_size_kept": float(np.median([c["rel_size"] for c in chosen])),
            "median_smooth_all": float(np.median([c["smooth"] for c in cs])),
            "median_smooth_kept": float(np.median([c["smooth"] for c in chosen])),
            "from_coco": sum("/coco/" in str(c["img"]) for c in chosen),
            "from_openimages": sum("/openimages/" in str(c["img"]) for c in chosen)}

    lines = [str(p) for p in drone_imgs] + [str(c["img"]) for c in keep]
    stats["negative_images"] = len(keep)
    stats["total_images"] = len(lines)
    stats["drone_fraction"] = len(drone_imgs) / len(lines)

    (YOLO / "drone4_bal_train.txt").write_text("\n".join(lines) + "\n")
    out = {"path": str(YOLO), "train": str(YOLO / "drone4_bal_train.txt"),
           "val": src["val"], "test": src["test"], "names": src["names"]}
    with open(YOLO / "drone4_bal.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    (YOLO / "drone4_bal_stats.json").write_text(json.dumps(stats, indent=1))
    assert not any("detfly" in l for l in lines), "Det-Fly leaked into training"
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
