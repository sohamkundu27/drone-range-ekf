"""Draw ground-truth boxes on a random sample of converted images.

For every dataset in data/yolo/ this writes to figures/sanity/:
  <ds>_<split>_<k>.jpg   full image (longest side <= 1600 px) with boxes and class names
  <ds>_crops.jpg         grid of zoomed crops around boxes. Tiny drones are invisible at
                         full-image scale, so this is the view that shows whether the
                         boxes actually sit on the objects.

    python -I scripts/sanity_check.py --n 8 --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLASSES, ROOT, YOLO  # noqa: E402

COLORS = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255)]  # BGR: drone red, bird green, airplane blue, heli magenta
OUT = ROOT / "figures" / "sanity"


def read_labels(p):
    return [(int(r[0]), *map(float, r[1:5])) for r in (l.split() for l in p.read_text().splitlines()) if r]


def to_px(b, W, H):
    _, cx, cy, w, h = b
    return int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="full images per dataset")
    ap.add_argument("--crops", type=int, default=24, help="crops per dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for ds in sorted(p.name for p in YOLO.iterdir() if (p / "labels").is_dir()):
        lbls = [p for p in (YOLO / ds / "labels").glob("*/*.txt") if p.stat().st_size > 0]
        if not lbls:
            continue
        sample = rng.sample(lbls, min(args.n, len(lbls)))
        tiles = []
        for k, lp in enumerate(sample):
            ip = Path(str(lp).replace("/labels/", "/images/")).with_suffix(".jpg")
            img = cv2.imread(str(ip))
            H, W = img.shape[:2]
            boxes = read_labels(lp)
            vis = img.copy()
            t = max(2, int(round(max(W, H) / 640)))
            for b in boxes:
                x1, y1, x2, y2 = to_px(b, W, H)
                cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[b[0]], t)
                # tiny boxes get a big circle so they are findable in the downscaled view
                if (x2 - x1) * (y2 - y1) < (0.02 * max(W, H)) ** 2:
                    cv2.circle(vis, ((x1 + x2) // 2, (y1 + y2) // 2), int(0.04 * max(W, H)), COLORS[b[0]], t)
                cv2.putText(vis, CLASSES[b[0]], (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, COLORS[b[0]], t)
            s = 1600 / max(W, H)
            if s < 1:
                vis = cv2.resize(vis, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT / f"{ds}_{lp.parent.name}_{k}.jpg"), vis)
        # zoomed crops
        for lp in rng.sample(lbls, min(args.crops * 3, len(lbls))):
            if len(tiles) >= args.crops:
                break
            ip = Path(str(lp).replace("/labels/", "/images/")).with_suffix(".jpg")
            img = cv2.imread(str(ip))
            H, W = img.shape[:2]
            b = rng.choice(read_labels(lp))
            x1, y1, x2, y2 = to_px(b, W, H)
            side = max(48, 3 * max(x2 - x1, y2 - y1))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            X1, Y1 = max(0, cx - side // 2), max(0, cy - side // 2)
            crop = img[Y1:Y1 + side, X1:X1 + side].copy()
            if crop.size == 0:
                continue
            cv2.rectangle(crop, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), COLORS[b[0]], 1)
            crop = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, f"{CLASSES[b[0]]} {x2 - x1}x{y2 - y1}px", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            tiles.append(crop)
        if tiles:
            cols = 6
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            cv2.imwrite(str(OUT / f"{ds}_crops.jpg"), grid)
        print(f"{ds}: {len(sample)} full images + {len(tiles)} crops -> {OUT}")


if __name__ == "__main__":
    main()
