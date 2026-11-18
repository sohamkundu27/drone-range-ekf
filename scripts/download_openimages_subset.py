"""Download an Open Images V7 subset (helicopter, bird, fixed-wing aircraft) and convert to YOLO.

Source: https://storage.googleapis.com/openimages/web/index.html
  Annotations (CC BY 4.0), in data/raw/openimages/csv/:
    oidv6-train-annotations-bbox.csv     https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv
    validation-annotations-bbox.csv      https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv
    test-annotations-bbox.csv            https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv
    oidv7-class-descriptions-boxable.csv https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv
    bbox_labels_600_hierarchy.json       https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json
  (Box annotations did not change between V6 and V7.)
  Images (listed by Google as CC BY 2.0) come from the official bucket
    https://open-images-dataset.s3.amazonaws.com/{train,validation,test}/<ImageID>.jpg

Class mapping (descendants in the OI hierarchy are included):
  Bird (+ Eagle, Duck, Owl, ...) -> bird ; Fixed-wing aircraft -> airplane ; Helicopter -> helicopter
Images are dropped if any relevant box is IsGroupOf (one box over a flock or group) or
IsDepiction (drawing/toy), or if they contain a generic "Aircraft" or "Rocket" box,
because an unlabelled aircraft would become a false negative.
Splits: OI train -> train, OI validation -> val, OI test -> test.
"""
import argparse
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLS, RAW, link_or_copy, out_paths, write_label  # noqa: E402

OI = RAW / "openimages"
CSV = OI / "csv"
FILES = {
    "train": ("oidv6-train-annotations-bbox.csv", "train"),
    "val": ("validation-annotations-bbox.csv", "validation"),
    "test": ("test-annotations-bbox.csv", "test"),
}
ROOT_MIDS = {"/m/015p6": "bird", "/m/0cmf2": "airplane", "/m/09ct_": "helicopter"}
EXCLUDE_MIDS = {"/m/0k5j", "/m/09rvcxw"}  # Aircraft (generic), Rocket


def descendants(node):
    out = []
    for c in node.get("Subcategory", []):
        out += [c["LabelName"]] + descendants(c)
    return out


def find(node, mid):
    if node["LabelName"] == mid:
        return node
    for c in node.get("Subcategory", []):
        r = find(c, mid)
        if r:
            return r
    return None


def build_mid_map():
    tree = json.load(open(CSV / "bbox_labels_600_hierarchy.json"))
    names = {r[0]: r[1] for r in csv.reader(open(CSV / "oidv7-class-descriptions-boxable.csv"))}
    mid2cls = {}
    for mid, cls in ROOT_MIDS.items():
        for m in [mid] + descendants(find(tree, mid)):
            mid2cls[m] = CLS[cls]
    print("OI classes used:", {names.get(m, m): c for m, c in mid2cls.items()})
    return mid2cls


def fetch(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=60) as r:
            data = r.read()
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(dst)
        return True
    except Exception as e:  # noqa: BLE001
        print("FAIL", url, e)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-train", type=int, default=3000, help="max images per class (bird/airplane) in train; helicopter uncapped")
    ap.add_argument("--cap-eval", type=int, default=500, help="max images per class in val/test")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="only print counts")
    args = ap.parse_args()

    mid2cls = build_mid_map()
    keep_mids = set(mid2cls) | EXCLUDE_MIDS
    rng = random.Random(0)
    jobs = []
    for split, (fname, s3dir) in FILES.items():
        cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"]
        parts = []
        for chunk in pd.read_csv(CSV / fname, usecols=cols, chunksize=2_000_000):
            parts.append(chunk[chunk.LabelName.isin(keep_mids)])
        df = pd.concat(parts)
        bad = set(df[df.LabelName.isin(EXCLUDE_MIDS) | (df.IsGroupOf == 1) | (df.IsDepiction == 1)].ImageID)
        df = df[~df.ImageID.isin(bad) & df.LabelName.isin(mid2cls)].copy()
        df["cls"] = df.LabelName.map(mid2cls)
        by_img = {i: g for i, g in df.groupby("ImageID")}
        cap = args.cap_train if split == "train" else args.cap_eval
        selected = set()
        # helicopter first (rarest, uncapped in train), then fill bird/airplane up to the cap
        for c in (CLS["helicopter"], CLS["airplane"], CLS["bird"]):
            ids = sorted(i for i, g in by_img.items() if (g.cls == c).any())
            rng.shuffle(ids)
            limit = len(ids) if (c == CLS["helicopter"] and split == "train") else cap
            already = sum(1 for i in selected if (by_img[i].cls == c).any())
            for i in ids:
                if already >= limit:
                    break
                if i not in selected:
                    selected.add(i)
                    already += 1
        counts = {k: sum(1 for i in selected if (by_img[i].cls == CLS[k]).any()) for k in ("bird", "airplane", "helicopter")}
        print(f"{split}: {len(by_img)} clean candidate images, {len(bad)} excluded; selected {len(selected)} -> {counts}")
        for i in sorted(selected):
            g = by_img[i]
            rows = [(int(r.cls), (r.XMin + r.XMax) / 2, (r.YMin + r.YMax) / 2, r.XMax - r.XMin, r.YMax - r.YMin)
                    for r in g.itertuples() if r.XMax > r.XMin and r.YMax > r.YMin]
            jobs.append((f"https://open-images-dataset.s3.amazonaws.com/{s3dir}/{i}.jpg",
                         OI / "images" / s3dir / f"{i}.jpg", split, i, rows))
    if args.dry_run:
        return
    with ThreadPoolExecutor(args.workers) as ex:
        ok = list(tqdm(ex.map(lambda j: fetch(j[0], j[1]), jobs), total=len(jobs), desc="openimages"))
    n = 0
    for good, (_, raw, split, stem, rows) in zip(ok, jobs):
        if good:
            img_dst, lbl_dst = out_paths("openimages", split, stem)
            link_or_copy(raw, img_dst)
            write_label(lbl_dst, rows)
            n += 1
    print(f"Open Images: wrote {n} images ({len(jobs) - n} failed)")


if __name__ == "__main__":
    main()
