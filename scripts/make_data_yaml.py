"""Write Ultralytics data YAMLs for whatever converted datasets exist in data/yolo/.

    data/yolo/drone4.yaml   train/val/test = union of all TRAINING datasets (never Det-Fly)
    data/yolo/<ds>.yaml     val = test = that dataset's test split (per-dataset evaluation)
    data/yolo/detfly.yaml   held-out cross-dataset test (Det-Fly only)
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLASSES, YOLO  # noqa: E402

HELD_OUT = {"detfly"}


def has(ds, split):
    d = YOLO / ds / "images" / split
    return d.is_dir() and any(d.iterdir())


def dump(path, d):
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    print("wrote", path)


def main():
    names = {i: n for i, n in enumerate(CLASSES)}
    datasets = sorted(p.name for p in YOLO.iterdir() if (p / "images").is_dir())
    train_ds = [d for d in datasets if d not in HELD_OUT]
    combo = {"path": str(YOLO)}
    for split in ("train", "val", "test"):
        combo[split] = [f"{d}/images/{split}" for d in train_ds if has(d, split)]
    combo["names"] = names
    dump(YOLO / "drone4.yaml", combo)
    for d in datasets:
        if has(d, "test"):
            dump(YOLO / f"{d}.yaml", {"path": str(YOLO), "train": f"{d}/images/test",
                                     "val": f"{d}/images/test", "test": f"{d}/images/test", "names": names})
    assert not any("detfly" in p for s in ("train", "val") for p in combo[s]), "Det-Fly leaked into training"


if __name__ == "__main__":
    main()
