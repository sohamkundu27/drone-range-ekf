"""Convert Det-Fly (Pascal VOC XML, class "UAV") to YOLO -- held-out TEST ONLY.

Input : data/raw/detfly/extracted/{JPEGImages,Annotations}/<seq>/<id>.{jpg,xml}
        (same layout as the official OneDrive folders; <seq> is 010 / 020)
Output: data/yolo/detfly/{images,labels}/test
Det-Fly is never placed in train/val: it is the cross-dataset generalisation test.
Optional args (defaults reproduce the original preliminary conversion exactly):
    python scripts/convert_detfly.py --src data/raw/detfly/full_release --dataset detfly_full --split all
converts the full release with the same per-box logic; the train/val/test split of that
output is defined later by scripts/build_detfly_split.py (frame-index blocks).
The real image size is read from the JPEG header; the XML <size> is only cross-checked.
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLS, RAW, link_or_copy, out_paths, write_label, xyxy_to_yolo  # noqa: E402

SRC = RAW / "detfly" / "extracted"


def main():
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC), help="folder with JPEGImages/ and Annotations/")
    ap.add_argument("--dataset", default="detfly", help="output data/yolo/<dataset>")
    ap.add_argument("--split", default="test", help="output split folder name")
    args = ap.parse_args()
    SRC = Path(args.src).resolve()
    n_img = n_box = n_noann = n_size_mismatch = 0
    for img_path in sorted((SRC / "JPEGImages").glob("*/*.jpg")):
        xml_path = SRC / "Annotations" / img_path.parent.name / f"{img_path.stem}.xml"
        if not xml_path.exists():
            n_noann += 1
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        root = ET.parse(xml_path).getroot()
        xw, xh = int(root.find("size/width").text), int(root.find("size/height").text)
        if (xw, xh) != (w, h):
            n_size_mismatch += 1
        rows = []
        for obj in root.iter("object"):
            assert obj.find("name").text.strip() == "UAV", xml_path
            bb = obj.find("bndbox")
            b = xyxy_to_yolo(*(float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")), w, h)
            if b:
                rows.append((CLS["drone"], *b))
        stem = f"{img_path.parent.name}_{img_path.stem}"
        img_dst, lbl_dst = out_paths(args.dataset, args.split, stem)
        link_or_copy(img_path, img_dst)
        write_label(lbl_dst, rows)
        n_img += 1
        n_box += len(rows)
    print(f"Det-Fly ({SRC}): {n_img} {args.split} images, {n_box} drone boxes; "
          f"{n_noann} images without XML skipped; {n_size_mismatch} XML size mismatches")


if __name__ == "__main__":
    main()
