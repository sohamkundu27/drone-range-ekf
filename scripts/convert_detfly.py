"""Convert Det-Fly (Pascal VOC XML, class "UAV") to YOLO -- held-out TEST ONLY.

Input : data/raw/detfly/extracted/{JPEGImages,Annotations}/<seq>/<id>.{jpg,xml}
        (same layout as the official OneDrive folders; <seq> is 010 / 020)
Output: data/yolo/detfly/{images,labels}/test
Det-Fly is never placed in train/val: it is the cross-dataset generalisation test.
The real image size is read from the JPEG header; the XML <size> is only cross-checked.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLS, RAW, link_or_copy, out_paths, write_label, xyxy_to_yolo  # noqa: E402

SRC = RAW / "detfly" / "extracted"


def main():
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
        img_dst, lbl_dst = out_paths("detfly", "test", stem)
        link_or_copy(img_path, img_dst)
        write_label(lbl_dst, rows)
        n_img += 1
        n_box += len(rows)
    print(f"Det-Fly: {n_img} test images, {n_box} drone boxes; "
          f"{n_noann} images without XML skipped; {n_size_mismatch} XML size mismatches")


if __name__ == "__main__":
    main()
