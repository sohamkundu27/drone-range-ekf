"""Measure detector latency (PyTorch .pt or ONNX via onnxruntime-gpu) on the GPU.

    python detector/latency.py --weights runs/detect/drone4/weights/best.pt --imgsz 1280
    python detector/latency.py --weights runs/detect/drone4/weights/best.onnx --imgsz 1280

Times the full predict call (pre-process + inference + post-process) on real 1920x1080 DUT
test images after a warm-up, batch 1, and reports mean / p50 / p95 in ms plus FPS. Also records
Ultralytics' own split (pre-process / inference / post-process). Results are appended to
experiments/latency.json under --label. These are DESKTOP-GPU numbers (RTX 3080), not embedded.
"""
import argparse
import json
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
    ap.add_argument("--label", default=None, help="key in experiments/latency.json")
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
    ts, split = [], []
    for im in imgs[args.warmup:]:
        t0 = time.perf_counter()
        r = model.predict(im, **kw)[0]
        ts.append((time.perf_counter() - t0) * 1e3)
        split.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
    ts, split = np.array(ts), np.array(split).mean(axis=0)
    print(f"{args.weights} @ {args.imgsz}: mean {ts.mean():.2f} ms, p50 {np.percentile(ts, 50):.2f}, "
          f"p95 {np.percentile(ts, 95):.2f} ms, {1e3 / ts.mean():.1f} FPS over {len(ts)} images")
    import torch
    res = {"weights": args.weights, "imgsz": args.imgsz, "half": args.half, "n_images": len(ts),
           "mean_ms": float(ts.mean()), "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
           "fps": float(1e3 / ts.mean()), "preprocess_ms": float(split[0]), "inference_ms": float(split[1]),
           "postprocess_ms": float(split[2]), "gpu": torch.cuda.get_device_name(0),
           "note": "desktop GPU, batch 1, end-to-end predict() call incl. pre/post-processing"}
    path = ROOT / "experiments" / "latency.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    allres[args.label or Path(args.weights).name] = res
    path.write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
