"""Train the 4-class detector (drone, bird, airplane, helicopter) with Ultralytics YOLO.

Full run (rebalanced data from scripts/rebalance_train.py, 7-hour time box, early stopping):
    python detector/train.py --data data/yolo/drone4_bal.yaml --model yolo26s.pt --imgsz 1280 \
        --epochs 150 --batch 8 --workers 12 --time 7 --patience 30 --name drone4_bal
Resume after a crash or reboot (continues from runs/detect/<name>/weights/last.pt):
    python detector/train.py --resume --name drone4_bal

1-epoch smoke test on a tiny random subset, to check the pipeline end to end on the GPU:
    python detector/train.py --smoke

Small objects: the median DUT/Det-Fly drone is only ~10-25 px after letterboxing to 640,
so train at --imgsz 1280 (or tile the 4K Det-Fly frames) rather than the default 640.
Runs go to runs/detect/<name>/ (git-ignored).
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "data" / "yolo"


def make_smoke_subset(n_train=96, n_val=32, seed=0) -> Path:
    """Hard-link a small mixed subset of every training dataset into data/yolo/_smoke/."""
    src = yaml.safe_load(open(YOLO_DIR / "drone4.yaml"))
    dst = YOLO_DIR / "_smoke"
    shutil.rmtree(dst, ignore_errors=True)
    rng = random.Random(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        imgs = [p for d in src[split] for p in sorted((YOLO_DIR / d).glob("*.jpg"))]
        for i, ip in enumerate(rng.sample(imgs, min(n, len(imgs)))):
            lp = Path(str(ip).replace("/images/", "/labels/")).with_suffix(".txt")
            for s, d in ((ip, dst / "images" / split / f"{i:04d}_{ip.name}"),
                         (lp, dst / "labels" / split / f"{i:04d}_{lp.name}")):
                d.parent.mkdir(parents=True, exist_ok=True)
                d.hardlink_to(s)
    y = dst / "smoke.yaml"
    yaml.safe_dump({"path": str(dst), "train": "images/train", "val": "images/val", "names": src["names"]}, open(y, "w"))
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(YOLO_DIR / "drone4.yaml"))
    ap.add_argument("--model", default="yolo26s.pt", help="pretrained weights or a model yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8, help="8 at 1280 px fits a 12 GB RTX 3080 for the s model")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="drone4")
    ap.add_argument("--time", type=float, default=None,
                    help="wall-clock budget in hours; Ultralytics then fits the epoch count/LR schedule to it")
    ap.add_argument("--patience", type=int, default=30, help="early stopping: epochs without val improvement")
    ap.add_argument("--resume", action="store_true", help="resume runs/detect/<name>/weights/last.pt")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a tiny subset")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        # All settings (data, epochs, time, batch, ...) are restored from the checkpoint.
        YOLO(str(ROOT / "runs" / "detect" / args.name / "weights" / "last.pt")).train(resume=True)
        return
    if args.smoke:
        args.data, args.epochs, args.name = str(make_smoke_subset()), 1, "smoke"
    model = YOLO(args.model)
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, workers=args.workers,
        time=args.time, patience=args.patience,
        device=0, project=str(ROOT / "runs" / "detect"), name=args.name, exist_ok=args.smoke,
        seed=0, deterministic=False, plots=not args.smoke,
    )


if __name__ == "__main__":
    main()
