"""Convert DUT Anti-UAV (detection subset, Pascal VOC XML) to YOLO.

Input : data/raw/dut_anti_uav/extracted/{train,val,test}/{img,xml}
Output: data/yolo/dut_anti_uav/{images,labels}/{train,val,test}
The official train/val/test split is kept. Class "UAV" -> drone (0).
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLS, RAW, link_or_copy, out_paths, write_label, xyxy_to_yolo  # noqa: E402

SRC = RAW / "dut_anti_uav" / "extracted"


def main():
    n_img = n_box = 0
    for split in ("train", "val", "test"):
        for xml_path in sorted((SRC / split / "xml").glob("*.xml")):
            root = ET.parse(xml_path).getroot()
            w = float(root.find("size/width").text)
            h = float(root.find("size/height").text)
            rows = []
            for obj in root.iter("object"):
                assert obj.find("name").text.strip() == "UAV", xml_path
                bb = obj.find("bndbox")
                b = xyxy_to_yolo(*(float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")), w, h)
                if b:
                    rows.append((CLS["drone"], *b))
            img_src = SRC / split / "img" / f"{xml_path.stem}.jpg"
            if not img_src.exists():
                print("missing image", img_src)
                continue
            img_dst, lbl_dst = out_paths("dut_anti_uav", split, xml_path.stem)
            link_or_copy(img_src, img_dst)
            write_label(lbl_dst, rows)
            n_img += 1
            n_box += len(rows)
    print(f"DUT Anti-UAV: {n_img} images, {n_box} drone boxes")


if __name__ == "__main__":
    main()
