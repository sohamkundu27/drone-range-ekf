"""Measure detector latency (PyTorch .pt or ONNX via onnxruntime-gpu) on the GPU.

    python detector/latency.py --weights runs/detect/drone4/weights/best.pt --imgsz 1280
    python detector/latency.py --weights runs/detect/drone4/weights/best.onnx --imgsz 1280

Times the full predict call (pre-process + inference + NMS) on real test images
after a warm-up, and reports mean / p50 / p95 in ms plus FPS.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--images", default=str(ROOT / "data" / "yolo" / "dut_anti_uav" / "images" / "test"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    files = sorted(Path(args.images).glob("*.jpg"))
    random.Random(0).shuffle(files)
    imgs = [cv2.imread(str(f)) for f in files[: args.n + args.warmup]]
    model = YOLO(args.weights, task="detect")
    kw = dict(imgsz=args.imgsz, device=0, half=args.half, verbose=False)
    for im in imgs[: args.warmup]:
        model.predict(im, **kw)
    ts = []
    for im in imgs[args.warmup:]:
        t0 = time.perf_counter()
        model.predict(im, **kw)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    print(f"{args.weights} @ {args.imgsz}: mean {ts.mean():.2f} ms, p50 {np.percentile(ts, 50):.2f}, "
          f"p95 {np.percentile(ts, 95):.2f} ms, {1e3 / ts.mean():.1f} FPS over {len(ts)} images")


if __name__ == "__main__":
    main()
