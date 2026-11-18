"""Convert Drone-vs-Bird (WOSDETC) videos + annotations to YOLO frames.

The VIDEOS require a signed data-usage agreement (see DATA.md). Annotations are
public in https://github.com/wosdetc/challenge/tree/master/annotations and are
already in data/raw/drone_vs_bird/annotations/.

Expected input after the manual download:
    data/raw/drone_vs_bird/videos/<name>.{mp4,avi,mov,MOV,MP4}   (name == annotation file stem)
    data/raw/drone_vs_bird/annotations/<name>.txt
Annotation format (one line per frame, 0-based frame index):
    framenum num_objs x_left y_top w h class [x_left y_top w h class ...]
Only "drone" is annotated: birds in these videos are NOT labelled, so frames are
used as drone positives only, with no bird labels.

Split is by VIDEO (no near-duplicate frames leaking across splits): about 70/15/15,
assigned deterministically from a hash of the video name. Every --stride-th frame is kept.
Output: data/yolo/drone_vs_bird/{images,labels}/{train,val,test}

    python -I scripts/convert_drone_vs_bird.py --dry-run   # stats from annotations only
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLS, RAW, out_paths, write_label, xyxy_to_yolo  # noqa: E402

SRC = RAW / "drone_vs_bird"
VIDEO_EXTS = (".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv")


def parse(path: Path):
    frames = {}
    for line in path.read_text().split("\n"):
        t = line.split()
        if len(t) < 2:
            continue
        fi, n = int(t[0]), int(t[1])
        boxes = []
        for k in range(n):
            x, y, w, h = map(float, t[2 + 5 * k: 6 + 5 * k])
            cls = t[6 + 5 * k]
            assert cls == "drone", (path, line)
            boxes.append((x, y, x + w, y + h))
        frames[fi] = boxes
    return frames


def split_of(name: str) -> str:
    v = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    return "train" if v < 70 else ("val" if v < 85 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10, help="keep every N-th frame")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    anns = sorted((SRC / "annotations").glob("*.txt"))
    tot = {"train": [0, 0, 0], "val": [0, 0, 0], "test": [0, 0, 0]}  # videos, frames kept, boxes
    missing = []
    for ann in anns:
        split = split_of(ann.stem)
        frames = parse(ann)
        kept = {i: b for i, b in frames.items() if i % args.stride == 0}
        tot[split][0] += 1
        tot[split][1] += len(kept)
        tot[split][2] += sum(len(b) for b in kept.values())
        if args.dry_run:
            continue
        vid = next((SRC / "videos" / f"{ann.stem}{e}" for e in VIDEO_EXTS if (SRC / "videos" / f"{ann.stem}{e}").exists()), None)
        if vid is None:
            missing.append(ann.stem)
            continue
        import cv2

        cap = cv2.VideoCapture(str(vid))
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi in kept:
                h, w = frame.shape[:2]
                rows = [(CLS["drone"], *b) for b in (xyxy_to_yolo(*bb, w, h) for bb in kept[fi]) if b]
                img_dst, lbl_dst = out_paths("drone_vs_bird", split, f"{ann.stem}_{fi:06d}")
                img_dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_label(lbl_dst, rows)
            fi += 1
        cap.release()
        if fi < max(frames, default=0) + 1:
            print(f"warning: {vid.name} has {fi} frames but annotations reach {max(frames) + 1}")
    for s, (v, f, b) in tot.items():
        print(f"{s}: {v} videos, {f} frames (stride {args.stride}), {b} drone boxes")
    if missing:
        print(f"{len(missing)} videos not found in {SRC / 'videos'} (manual download needed, see DATA.md)")


if __name__ == "__main__":
    main()
