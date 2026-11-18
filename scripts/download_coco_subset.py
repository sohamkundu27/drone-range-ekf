"""Download the COCO 2017 bird + airplane subset and convert it to YOLO.

Source : https://cocodataset.org  (annotations CC BY 4.0, images under Flickr terms)
Needs  : data/raw/coco/zips/annotations_trainval2017.zip
         (http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
Images are fetched one by one from http://images.cocodataset.org/{train,val}2017/<file>
into data/raw/coco/images/, so the 18 GB train2017 zip is not needed.

Splits: train2017 -> train/val (deterministic 90/10 by image id), val2017 -> test.
Only bird (COCO id 16) and airplane (COCO id 5) are labelled; every other object is
background. Images where a bird/airplane is marked iscrowd are skipped, because the
crowd region cannot be expressed as YOLO boxes and would act as a false negative.
"""
import argparse
import json
import random
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLS, RAW, link_or_copy, out_paths, write_label, xyxy_to_yolo  # noqa: E402

COCO = RAW / "coco"
CAT_MAP = {16: CLS["bird"], 5: CLS["airplane"]}


def fetch(url: str, dst: Path) -> bool:
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


def load(zf: zipfile.ZipFile, name: str):
    with zf.open(f"annotations/instances_{name}.json") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=0, help="cap images per class per source split (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    zf = zipfile.ZipFile(COCO / "zips" / "annotations_trainval2017.zip")
    rng = random.Random(0)
    jobs = []  # (url, raw_path, yolo_split, stem, rows)
    for coco_split in ("train2017", "val2017"):
        d = load(zf, coco_split)
        imgs = {im["id"]: im for im in d["images"]}
        anns = defaultdict(list)
        crowd = set()
        for a in d["annotations"]:
            if a["category_id"] in CAT_MAP:
                (crowd.add(a["image_id"]) if a["iscrowd"] else anns[a["image_id"]].append(a))
        ids = sorted(set(anns) - crowd)
        if args.max_per_class:
            keep = set()
            for cat in CAT_MAP:
                cid = [i for i in ids if any(a["category_id"] == cat for a in anns[i])]
                rng.shuffle(cid)
                keep.update(cid[: args.max_per_class])
            ids = sorted(keep)
        print(f"{coco_split}: {len(ids)} images with bird/airplane ({len(crowd)} skipped for iscrowd)")
        for iid in ids:
            im = imgs[iid]
            rows = []
            for a in anns[iid]:
                x, y, w, h = a["bbox"]
                b = xyxy_to_yolo(x, y, x + w, y + h, im["width"], im["height"])
                if b:
                    rows.append((CAT_MAP[a["category_id"]], *b))
            if coco_split == "val2017":
                split = "test"
            else:
                split = "val" if (iid * 2654435761) % 100 < 10 else "train"
            raw = COCO / "images" / coco_split / im["file_name"]
            url = f"http://images.cocodataset.org/{coco_split}/{im['file_name']}"
            jobs.append((url, raw, split, Path(im["file_name"]).stem, rows))

    with ThreadPoolExecutor(args.workers) as ex:
        ok = list(tqdm(ex.map(lambda j: fetch(j[0], j[1]), jobs), total=len(jobs), desc="coco images"))
    n = 0
    for good, (_, raw, split, stem, rows) in zip(ok, jobs):
        if not good:
            continue
        img_dst, lbl_dst = out_paths("coco", split, stem)
        link_or_copy(raw, img_dst)
        write_label(lbl_dst, rows)
        n += 1
    print(f"COCO: wrote {n} images ({len(jobs) - n} failed downloads)")


if __name__ == "__main__":
    main()
