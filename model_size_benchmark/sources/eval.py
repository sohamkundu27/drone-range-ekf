"""Evaluate a trained detector on one dataset's test split (in-domain or the held-out Det-Fly set).

    python detector/eval.py --weights runs/detect/drone4_bal/weights/best.pt --data data/yolo/dut_anti_uav.yaml --tag dut
    python detector/eval.py --weights runs/detect/drone4_bal/weights/best.pt --data data/yolo/drone4_bal.yaml --tag all4
    python detector/eval.py --weights runs/detect/drone4_bal/weights/best.pt --data data/yolo/detfly.yaml --tag detfly_partial

Prints overall and per-class P / R / mAP50 / mAP50-95 and writes experiments/eval_<tag>.json.
P and R are Ultralytics' values at the confidence that maximises F1 (per class).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True, help="name for the output JSON")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    m = YOLO(args.weights).val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                               conf=args.conf, device=args.device, project=str(ROOT / "runs" / "val"),
                               name=args.tag, exist_ok=True, plots=True)
    names = m.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[names[int(c)]] = {"P": float(m.box.p[i]), "R": float(m.box.r[i]),
                                    "mAP50": float(m.box.ap50[i]), "mAP50-95": float(m.box.ap[i]),
                                    "n_instances": int(m.nt_per_class[int(c)]) if hasattr(m, "nt_per_class") else None}
    res = {"tag": args.tag, "weights": args.weights, "data": args.data, "split": args.split,
           "imgsz": args.imgsz, "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
           "P": float(m.box.mp), "R": float(m.box.mr), "per_class": per_class,
           "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()}}
    print(json.dumps(res, indent=1))
    out = ROOT / "experiments" / f"eval_{args.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
