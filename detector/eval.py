"""Evaluate a trained detector on one dataset's test split (in-domain or the held-out Det-Fly set).

    python detector/eval.py --weights runs/detect/drone4/weights/best.pt --data data/yolo/drone4.yaml
    python detector/eval.py --weights runs/detect/drone4/weights/best.pt --data data/yolo/detfly.yaml

Prints overall and per-class mAP50 / mAP50-95 / P / R and writes experiments/eval_<data>_<run>.json.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    from ultralytics import YOLO

    m = YOLO(args.weights).val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                               conf=args.conf, device=0, project=str(ROOT / "runs" / "val"), plots=True)
    names = m.names
    res = {"weights": args.weights, "data": args.data, "split": args.split, "imgsz": args.imgsz,
           "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "P": float(m.box.mp), "R": float(m.box.mr),
           "per_class": {names[c]: {"mAP50": float(m.box.ap50[i]), "mAP50-95": float(m.box.ap[i])}
                         for i, c in enumerate(m.box.ap_class_index)}}
    print(json.dumps(res, indent=1))
    out = ROOT / "experiments" / f"eval_{Path(args.data).stem}_{Path(args.weights).parents[1].name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
