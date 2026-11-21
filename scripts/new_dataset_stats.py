"""Dataset statistics including the Det-Fly retrain split (writes presentation_assets_v2/dataset_stats.{md,json}).

    python scripts/new_dataset_stats.py

Table 1 ("images · boxes"): what the OLD and NEW models were trained / validated / tested on, per
source dataset, counted from the actual image lists (data/yolo/drone4_bal_train.txt, new_*.txt and
the val/test folders). Each cell is `images · boxes` (all classes) and the drone-box count.
Table 2: the old detailed per-split table (same columns and code path as scripts/dataset_stats.py),
with rows added for the Det-Fly full-release splits.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLASSES, ROOT, YOLO  # noqa: E402

OUT = ROOT / "presentation_assets_v2"
NICE = {"dut_anti_uav": "DUT Anti-UAV", "coco": "COCO", "openimages": "Open Images", "detfly_full": "Det-Fly"}


def label_of(img):
    return Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")


def source_of(img):
    return Path(img).relative_to(YOLO).parts[0]


def count(imgs):
    """-> {source: {"images", "boxes", "drone", per-class...}}"""
    out = defaultdict(Counter)
    for im in imgs:
        src = source_of(im)
        out[src]["images"] += 1
        for r in label_of(im).read_text().splitlines():
            if r.strip():
                c = CLASSES[int(r.split()[0])]
                out[src]["boxes"] += 1
                out[src][c] += 1
    return out


def lst(name):
    return [Path(p) for p in (YOLO / name).read_text().split()]


def folder(ds, split):
    return sorted((YOLO / ds / "images" / split).glob("*.jpg"))


def detail(lbls):
    """Same statistics as scripts/dataset_stats.py for a list of label files."""
    cnt, ws, hs, s640, s1280, res, n_bg = Counter(), [], [], [], [], Counter(), 0
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
            s640.append(s * 640 / max(W, H))
            s1280.append(s * 1280 / max(W, H))
    ws, hs = np.array(ws), np.array(hs)
    sq = np.sqrt(ws * hs)
    return {"images": len(lbls), "background_images": n_bg, "boxes": dict(cnt),
            "median_w_px": float(np.median(ws)), "median_h_px": float(np.median(hs)),
            "median_sqrt_area_px": float(np.median(sq)), "frac_lt16": float((sq < 16).mean()),
            "frac_lt32": float((sq < 32).mean()), "median_sqrt_area_at_640": float(np.median(s640)),
            "median_sqrt_area_at_1280": float(np.median(s1280)), "top_resolutions": res.most_common(3)}


def main():
    OUT.mkdir(exist_ok=True)
    old_val = [p for ds in ("coco", "dut_anti_uav", "openimages") for p in folder(ds, "val")]
    sets = {
        "OLD train": [Path(p) for p in (YOLO / "drone4_bal_train.txt").read_text().split()],
        "NEW train": lst("new_train_drone4_detfly.txt"),
        "OLD val": old_val,
        "NEW val": old_val + lst("new_detfly_val.txt"),
        "test: DUT+COCO+OI (all4)": [p for ds in ("coco", "dut_anti_uav", "openimages") for p in folder(ds, "test")],
        "test: Det-Fly 010 block": lst("new_detfly_test010.txt"),
        "test: Det-Fly 020": lst("new_detfly_test020.txt"),
        "test: Det-Fly preliminary (old)": folder("detfly", "test"),
    }
    counts = {k: count(v) for k, v in sets.items()}
    srcs = ["dut_anti_uav", "coco", "openimages", "detfly_full", "detfly"]
    L = ["# Dataset statistics (Det-Fly retrain)", "",
         "> Reading of the request: the old file had no literal \"images · boxes\" table, so Table 1 gives the "
         "training/validation/test composition in that form, and Table 2 keeps the old detailed per-split table "
         "(identical columns and code) with the new Det-Fly rows added.", "",
         "## Table 1 — images · boxes (all classes); drone boxes in brackets", "",
         "| set | " + " | ".join(NICE.get(s, "Det-Fly prelim (old)") for s in srcs) + " | **total** | drone share of images |",
         "|---|" + "---|" * (len(srcs) + 2)]
    js = {}
    for k, c in counts.items():
        tot = Counter()
        for s in c.values():
            tot.update(s)
        cells = []
        for s in srcs:
            if s in c:
                cells.append(f"{c[s]['images']:,} · {c[s]['boxes']:,} [{c[s]['drone']:,}]")
            else:
                cells.append("—")
        drone_imgs = sum(1 for im in sets[k] if any(r.split()[0] == "0" for r in label_of(im).read_text().splitlines() if r.strip()))
        L.append(f"| {k} | " + " | ".join(cells) + f" | **{tot['images']:,} · {tot['boxes']:,} [{tot['drone']:,}]** | "
                 f"{drone_imgs / max(1, tot['images']):.1%} |")
        js[k] = {"per_source": {s: dict(v) for s, v in c.items()}, "total": dict(tot), "images_with_drone": drone_imgs}
    L += ["", "Train lists: OLD = `data/yolo/drone4_bal_train.txt` (5,200 DUT drone images + 5,000 hard negatives); "
          "NEW = that list + 1,808 Det-Fly 010 frames (`detfly_split.json`). DUT train images: 5,200 with 5,243 drone "
          "boxes (3 are background-only).", ""]
    old = json.loads((ROOT / "figures" / "dataset_stats.json").read_text())
    rows = dict(old)
    for name, lname in (("train", "new_detfly_train.txt"), ("val", "new_detfly_val.txt"),
                        ("test010", "new_detfly_test010.txt"), ("test020", "new_detfly_test020.txt")):
        rows[f"detfly_full/{name}"] = detail([label_of(p) for p in lst(lname)])
    f = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    L += ["## Table 2 — per-split detail (box sizes in ORIGINAL pixels; @640/@1280 = after letterboxing)", "",
          "| dataset | split | images | bg imgs | " + " | ".join(CLASSES) +
          " | med w px | med h px | med sqrt(wh) px | <16px | <32px | med @640 | med @1280 | typical res |",
          "|" + "---|" * (12 + len(CLASSES))]
    for key in sorted(rows):
        st = rows[key]
        ds, split = key.split("/")
        b = st["boxes"]
        res = st["top_resolutions"][0][0] if st["top_resolutions"] else "-"
        L.append(f"| {ds} | {split} | {st['images']} | {st['background_images']} | " +
                 " | ".join(str(b.get(c, 0)) for c in CLASSES) +
                 f" | {f(st['median_w_px'])} | {f(st['median_h_px'])} | {f(st['median_sqrt_area_px'])} | "
                 f"{st['frac_lt16']:.0%} | {st['frac_lt32']:.0%} | {f(st['median_sqrt_area_at_640'])} | "
                 f"{f(st['median_sqrt_area_at_1280'])} | {res} |")
    L += ["", "Rows other than `detfly_full/*` are copied from `figures/dataset_stats.json` (unchanged data). "
          "`detfly` = the old 1,000-frame preliminary slice; `detfly_full` = the full release split by "
          "`scripts/build_detfly_split.py`."]
    (OUT / "dataset_stats.md").write_text("\n".join(L) + "\n")
    (OUT / "dataset_stats.json").write_text(json.dumps({"composition": js, "detail": rows}, indent=1) + "\n")
    print("\n".join(L[:16]))


if __name__ == "__main__":
    main()
