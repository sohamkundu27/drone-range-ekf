"""Shared helpers for dataset download/conversion scripts.

Layout produced by every converter:

    data/yolo/<dataset>/images/<split>/<name>.jpg
    data/yolo/<dataset>/labels/<split>/<name>.txt   (YOLO: cls cx cy w h, normalised)

Images are hard-linked from data/raw/ when possible (no extra disk use),
falling back to a copy.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
YOLO = ROOT / "data" / "yolo"

CLASSES = ["drone", "bird", "airplane", "helicopter"]
CLS = {n: i for i, n in enumerate(CLASSES)}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    """Pixel corners -> normalised (cx, cy, bw, bh), clipped to the image."""
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw / w, bh / h)


def write_label(path: Path, rows) -> None:
    """rows: iterable of (cls_id, cx, cy, bw, bh). Empty file = background image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c, cx, cy, bw, bh in rows:
            f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def out_paths(dataset: str, split: str, stem: str, ext: str = ".jpg"):
    img = YOLO / dataset / "images" / split / f"{stem}{ext}"
    lbl = YOLO / dataset / "labels" / split / f"{stem}.txt"
    return img, lbl
