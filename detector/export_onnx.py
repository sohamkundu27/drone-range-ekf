"""Export trained weights to ONNX (static shape, optional FP16) for latency tests and deployment.

    python detector/export_onnx.py --weights runs/detect/drone4/weights/best.pt --imgsz 1280
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--half", action="store_true", help="FP16 export (needs a GPU)")
    ap.add_argument("--opset", type=int, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    path = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, half=args.half, simplify=True,
                                     opset=args.opset, device=0 if args.half else "cpu")
    print("exported", path)


if __name__ == "__main__":
    main()
