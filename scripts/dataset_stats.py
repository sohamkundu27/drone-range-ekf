"""Per-dataset/split statistics of the converted YOLO data.

Reports image counts, background (empty-label) images, boxes per class, and box size
in ORIGINAL image pixels: median width, height and sqrt(w*h), plus the share of boxes
under 16 and 32 px. Also estimates the median drone size after letterboxing to 640
and 1280, which is what the detector actually sees.
Writes figures/dataset_stats.md and figures/dataset_stats.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLASSES, ROOT, YOLO  # noqa: E402


def main():
    out = {}
    lines = ["| dataset | split | images | bg imgs | " + " | ".join(CLASSES) +
             " | med w px | med h px | med sqrt(wh) px | <16px | <32px | med @640 | med @1280 | typical res |",
             "|" + "---|" * (12 + len(CLASSES))]
    for ds in sorted(p.name for p in YOLO.iterdir() if (p / "labels").is_dir()):
        for split in ("train", "val", "test", "all"):
            splits = ("train", "val", "test") if split == "all" else (split,)
            lbls = [p for s in splits for p in sorted((YOLO / ds / "labels" / s).glob("*.txt"))]
            if not lbls:
                continue
            cnt, ws, hs, scale640, scale1280, res, n_bg = Counter(), [], [], [], [], Counter(), 0
            for lp in lbls:
                rows = [r.split() for r in lp.read_text().splitlines() if r.strip()]
                if not rows:
                    n_bg += 1
                    continue
                ip = Path(str(lp).replace("/labels/", "/images/")).with_suffix(".jpg")
                with Image.open(ip) as im:
                    W, H = im.size
                res[f"{W}x{H}"] += 1
                for r in rows:
                    c, bw, bh = int(r[0]), float(r[3]) * W, float(r[4]) * H
                    cnt[CLASSES[c]] += 1
                    ws.append(bw)
                    hs.append(bh)
                    s = np.sqrt(bw * bh)
                    scale640.append(s * 640 / max(W, H))
                    scale1280.append(s * 1280 / max(W, H))
            ws, hs = np.array(ws), np.array(hs)
            sq = np.sqrt(ws * hs) if len(ws) else np.array([np.nan])
            st = {
                "images": len(lbls), "background_images": n_bg, "boxes": dict(cnt),
                "median_w_px": float(np.median(ws)) if len(ws) else None,
                "median_h_px": float(np.median(hs)) if len(hs) else None,
                "median_sqrt_area_px": float(np.median(sq)),
                "frac_lt16": float((sq < 16).mean()), "frac_lt32": float((sq < 32).mean()),
                "median_sqrt_area_at_640": float(np.median(scale640)) if scale640 else None,
                "median_sqrt_area_at_1280": float(np.median(scale1280)) if scale1280 else None,
                "top_resolutions": res.most_common(3),
            }
            out[f"{ds}/{split}"] = st
            f = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
            lines.append(f"| {ds} | {split} | {st['images']} | {n_bg} | " + " | ".join(str(cnt.get(c, 0)) for c in CLASSES) +
                         f" | {f(st['median_w_px'])} | {f(st['median_h_px'])} | {f(st['median_sqrt_area_px'])} | "
                         f"{st['frac_lt16']:.0%} | {st['frac_lt32']:.0%} | {f(st['median_sqrt_area_at_640'])} | "
                         f"{f(st['median_sqrt_area_at_1280'])} | {res.most_common(1)[0][0] if res else '-'} |")
    md = "\n".join(lines)
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "figures" / "dataset_stats.md").write_text(md + "\n")
    (ROOT / "figures" / "dataset_stats.json").write_text(json.dumps(out, indent=1))
    print(md)


if __name__ == "__main__":
    main()
