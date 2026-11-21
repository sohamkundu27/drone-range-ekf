"""Build the Det-Fly train/val/test split for the retrain experiment (frame-index blocks only).

    python scripts/build_detfly_split.py

Input : data/yolo/detfly_full/{images,labels}/all/<seq>_<id>.*  (scripts/convert_detfly.py --dataset detfly_full --split all)
Output: detfly_split.json (repo root, committed: frame-index ranges + counts, no images)
        data/yolo/new_detfly_{train,val,test010,test020}.txt   image lists (absolute paths)
        data/yolo/new_train_drone4_detfly.txt                  old rebalanced train list + Det-Fly train frames
        data/yolo/new_drone4_detfly.yaml                       train/val yaml for the retrained models
        data/yolo/new_detfly_{test020,test010,prelim}.yaml     drone-only test yamls (val == test == the list)

Reading of the spec (written to detfly_split.json["notes"] too):
* "frame index" = position of the frame in the id-sorted sequence (ids have gaps), 0-based, end-exclusive.
* 010 is cut at 60 % and 75 % of its 6,358 frames. The 200-frame buffers are taken from the END of the
  train block and the END of the val block, so the 010 test block is exactly the last 25 %.
* Train block subsampled with stride 2 (<= ~2,000 frames), val block with stride 2 (<= ~500 frames).
* 020: every frame is test (cross-recording / later-part test). Never trained on, never used for val.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
Y = ROOT / "data" / "yolo"
IMG = Y / "detfly_full" / "images" / "all"
STRIDE_TRAIN, STRIDE_VAL, BUFFER = 2, 2, 200


def main():
    seqs = {s: sorted(IMG.glob(f"{s}_*.jpg")) for s in ("010", "020")}
    n10 = len(seqs["010"])
    b_val, b_test = round(0.60 * n10), round(0.75 * n10)
    blocks = {
        "010_train_block": (0, b_val - BUFFER),
        "010_buffer_1": (b_val - BUFFER, b_val),
        "010_val_block": (b_val, b_test - BUFFER),
        "010_buffer_2": (b_test - BUFFER, b_test),
        "010_test_block": (b_test, n10),
    }
    f10 = seqs["010"]
    train = f10[blocks["010_train_block"][0]:blocks["010_train_block"][1]:STRIDE_TRAIN]
    val = f10[blocks["010_val_block"][0]:blocks["010_val_block"][1]:STRIDE_VAL]
    test010 = f10[blocks["010_test_block"][0]:blocks["010_test_block"][1]]
    test020 = seqs["020"]
    prelim = sorted((Y / "detfly" / "images" / "test").glob("*.jpg"))

    def boxes(files):
        return sum(len((Y / "detfly_full" / "labels" / "all" / f"{f.stem}.txt").read_text().split("\n")) - 1
                   for f in files)

    def write_list(name, files):
        p = Y / name
        p.write_text("".join(f"{f}\n" for f in files))
        return p

    lists = {k: write_list(f"new_detfly_{k}.txt", v)
             for k, v in (("train", train), ("val", val), ("test010", test010), ("test020", test020))}
    old_train = (Y / "drone4_bal_train.txt").read_text().split()
    new_train = write_list("new_train_drone4_detfly.txt", [Path(p) for p in old_train] + train)

    old = yaml.safe_load((Y / "drone4_bal.yaml").read_text())
    names = old["names"]
    new_yaml = {"path": str(Y), "train": str(new_train),
                "val": list(old["val"]) + [str(lists["val"])], "test": list(old["test"]), "names": names}
    (Y / "new_drone4_detfly.yaml").write_text(yaml.safe_dump(new_yaml, sort_keys=False))
    for tag, lst in (("test020", lists["test020"]), ("test010", lists["test010"])):
        (Y / f"new_detfly_{tag}.yaml").write_text(yaml.safe_dump(
            {"path": str(Y), "train": str(lst), "val": str(lst), "test": str(lst), "names": names}, sort_keys=False))
    prelim_list = write_list("new_detfly_prelim.txt", prelim)
    (Y / "new_detfly_prelim.yaml").write_text(yaml.safe_dump(
        {"path": str(Y), "train": str(prelim_list), "val": str(prelim_list), "test": str(prelim_list),
         "names": names}, sort_keys=False))

    def rng(files):
        return {"first": files[0].stem, "last": files[-1].stem, "frames": len(files), "boxes": boxes(files)}

    prelim_in_train = len({f.name for f in prelim} & {f.name for f in train})
    out = {
        "notes": [
            "frame index = 0-based position in the id-sorted sequence (frame ids have gaps); ranges are [start, end).",
            "010 cut at 60 % / 75 % of its frames; 200-frame buffers taken from the end of the train and val blocks "
            "and dropped, so the 010 test block is exactly the last 25 % of 010.",
            f"train block subsampled with stride {STRIDE_TRAIN}, val block with stride {STRIDE_VAL}; test blocks keep every frame.",
            "020 is entirely test: never trained on, never in validation / early stopping.",
            "020 is the same site as 010 (same hill, buildings, sports courts, dusk light); treat it as a later "
            "part of the same recording, not an independent cross-site test.",
            "frame 020/0207614 has an XML but no JPEG in the source release (skipped); 020_0201689 has an XML "
            "with no UAV object (kept as a background test frame).",
            f"the 1,000-frame preliminary slice (010_0100001..010_0101017) lies inside the 010 train block; "
            f"{prelim_in_train} of its frames are in the new training set, so new models must not be scored on it.",
        ],
        "sequence_frames": {s: len(v) for s, v in seqs.items()},
        "stride": {"train": STRIDE_TRAIN, "val": STRIDE_VAL},
        "buffer_frames": BUFFER,
        "frame_index_ranges_010": {k: {"start": a, "end": b, "first_id": f10[a].stem, "last_id": f10[b - 1].stem}
                                   for k, (a, b) in blocks.items()},
        "frame_index_ranges_020": {"020_test": {"start": 0, "end": len(test020),
                                                "first_id": test020[0].stem, "last_id": test020[-1].stem}},
        "sets": {"detfly_train": rng(train), "detfly_val": rng(val), "detfly_test010": rng(test010),
                 "detfly_test020": rng(test020), "detfly_prelim_slice": rng(prelim)},
        "new_training_set": {"old_rebalanced_images": len(old_train), "detfly_train_images": len(train),
                             "total": len(old_train) + len(train)},
        "files": {k: str(v.relative_to(ROOT)) for k, v in {**lists, "train_all": new_train}.items()},
    }
    (ROOT / "detfly_split.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["sets"], indent=1), json.dumps(out["new_training_set"]))


if __name__ == "__main__":
    main()
