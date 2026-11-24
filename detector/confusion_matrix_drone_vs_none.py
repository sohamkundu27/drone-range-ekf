"""'Drone vs no drone' confusion matrix for the slides, incl. Det-Fly, pooled from per-source 4-class + background
matrices.

    python -I detector/confusion_matrix_drone_vs_none.py                 # TEST split (default)
    python -I detector/confusion_matrix_drone_vs_none.py --split val     # rerun on val (files get a _val suffix)

Same settings as detector/confusion_matrix_conf.py: Ultralytics val, imgsz 1280, conf 0.25 (boxes with
conf > 0.25 enter the matrix), NMS IoU 0.5, match IoU 0.5 (ConfusionMatrix.process_batch, class-agnostic,
highest confidence first). Weights: the retrained YOLO26s (Det-Fly added, runs/s_detfly_new, best epoch 92;
runs/detect/drone4_bal is the pre-Det-Fly s run).

Sources (TEST split), each validated separately through a one-source data yaml:
  dut         DUT Anti-UAV test           2,200 images  (drones)
  coco        COCO test                     209 images  (birds, airplanes)
  openimages  Open Images test            1,325 images  (birds, airplanes, helicopters)
  detfly010   Det-Fly 010 test block      1,590 frames  (detfly_split.json: last 25 % of 010)
  detfly020   Det-Fly 020, all frames     6,912 frames  (same site as 010, later part of the recording)
The Det-Fly sets are the same lists the Det-Fly accuracy-table rows use (data/yolo/new_detfly_test0*0.txt).
For --split val: dut / coco / openimages val and Det-Fly val (data/yolo/new_detfly_val.txt).

1. Per source, a 5x5 raw matrix M[predicted][true] over (drone, bird, airplane, helicopter, background):
     M[d, background] = predicted d where no object is (false positive box)
     M[background, t] = object of class t that no box matched (missed)
2. Pooled 2x4 matrix (rows predicted: drone / no drone = a bird/airplane/helicopter box or no box;
   columns true: drone (DUT), drone (Det-Fly, same site), negatives (bird + airplane + helicopter),
   background (box where nothing is)). Every cell is a sum of per-source 5x5 cells; no drone / background is
   not defined ("-").
Writes figures/confusion_matrix_drone_vs_none{,_val}.json, .png (object columns normalised by column, background
as counts) and _counts.png. Exits with status 1 if a check fails. Do not quote mAP from this run.
"""
import argparse
import inspect
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo"
NAMES = ["drone", "bird", "airplane", "helicopter"]
ROWS = ["drone", "no drone"]
COLS = ["drone\n(DUT)", "drone\n(Det-Fly,\nsame site)", "negatives\n(bird, airplane,\nhelicopter)",
        "background\n(nothing there)"]
COLS_JSON = ["drone (DUT)", "drone (Det-Fly, same site)", "negatives (bird, airplane, helicopter)",
             "background (nothing there)"]
SOURCES = {   # source -> (group, data entry relative to data/yolo) per split
    "test": {"dut": ("dut", "dut_anti_uav/images/test"), "coco": ("neg", "coco/images/test"),
             "openimages": ("neg", "openimages/images/test"),
             "detfly010": ("detfly", "new_detfly_test010.txt"), "detfly020": ("detfly", "new_detfly_test020.txt")},
    "val": {"dut": ("dut", "dut_anti_uav/images/val"), "coco": ("neg", "coco/images/val"),
            "openimages": ("neg", "openimages/images/val"), "detfly_val": ("detfly", "new_detfly_val.txt")},
}
LABEL = {"dut": "DUT", "coco": "COCO", "openimages": "Open Images", "detfly010": "Det-Fly 010 test",
         "detfly020": "Det-Fly 020", "detfly_val": "Det-Fly val"}
# Expected test totals. Negatives: 2,142 label lines, 5 of them exact duplicates (one bird box written twice in
# 5 Open Images test files), which Ultralytics drops on load, so 2,137 are evaluated.
EXPECTED = {"test": {"dut_drone": 2245, "negatives": 2137, "negatives_raw_label_lines": 2142}}


def pool(Ms, groups):
    """Pooled 2x4 matrix (float, NaN = undefined) by explicit sums of per-source 5x5 cells."""
    D, NEG, BG = 0, [1, 2, 3], 4
    P = np.full((2, 4), np.nan)
    for j, g in ((0, "dut"), (1, "detfly")):
        src = [s for s in Ms if groups[s] == g]
        P[0, j] = sum(Ms[s][D, D] for s in src)                     # drones found
        P[1, j] = sum(Ms[s][1:, D].sum() for s in src)              # missed (other class or no box)
    P[0, 2] = sum(Ms[s][D, NEG].sum() for s in Ms)                  # negatives called drone
    P[1, 2] = sum(Ms[s][1:, :][:, NEG].sum() for s in Ms)           # negatives not called drone
    P[0, 3] = sum(Ms[s][D, BG] for s in Ms)                         # false drone boxes on empty background
    return P


def pool_by_matrices(Ms, groups):
    """Same pooling as aggregation matrices R (2x5) @ M @ C_s (5x4) summed over sources, where C_s routes the
    drone column to the source's own drone column. Independent of pool()."""
    R = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 1, 1]])
    P = np.zeros((2, 4))
    for s, M in Ms.items():
        C = np.zeros((5, 4))
        C[0, {"dut": 0, "detfly": 1}.get(groups[s], 0)] = 1     # a negatives source has an all-zero drone column
        C[1:4, 2] = 1
        C[4, 3] = 1
        P += R @ M @ C
    P[1, 3] = np.nan
    return P


def label_counts(entry):
    """Images and GT boxes per class for one data entry (folder or list file). gt_boxes counts unique lines per
    label file (Ultralytics drops exact duplicate labels on load); raw_label_lines counts every line."""
    p = DATA / entry
    if p.suffix == ".txt":
        imgs = [Path(x.strip()) for x in p.read_text().splitlines() if x.strip()]
    else:
        imgs = sorted(q for q in p.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
    cnt, raw, dups, missing_img = {n: 0 for n in NAMES}, {n: 0 for n in NAMES}, {}, 0
    for im in imgs:
        missing_img += not im.exists()
        lab = Path(str(im).replace("/images/", "/labels/")).with_suffix(".txt")
        if lab.exists():
            rows = [tuple(float(x) for x in ln.split()) for ln in lab.read_text().splitlines() if ln.strip()]
            for r in rows:
                raw[NAMES[int(r[0])]] += 1
            for r in set(rows):
                cnt[NAMES[int(r[0])]] += 1
            if len(set(rows)) < len(rows):
                dups[lab.name] = len(rows) - len(set(rows))
    return {"images": len(imgs), "images_missing_on_disk": missing_img, "gt_boxes": cnt, "raw_label_lines": raw,
            "duplicates": dups}


def validate(model, name, entry, args, tmp):
    """One Ultralytics val call on a single source; returns the 5x5 matrix [predicted, true]."""
    y = Path(tmp) / f"{name}.yaml"
    y.write_text(yaml.safe_dump({"path": str(DATA), "train": entry, "val": entry,
                                 "names": dict(enumerate(NAMES))}))
    r = model.val(data=str(y), split="val", imgsz=args.imgsz, batch=8, conf=args.conf, iou=args.iou,
                  plots=True, project=tmp, name=f"cm_{name}", exist_ok=True, verbose=False)
    cm = r.confusion_matrix
    assert [cm.names[i] for i in range(cm.nc)] == NAMES, cm.names
    return cm.matrix.astype(int)


def plot(P, normalise, title, path):
    """Same figure size, colour map and text layout as the earlier drone-vs-none figure (P is [predicted, true])."""
    nc = P.shape[1]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    if normalise:
        A = np.full_like(P, np.nan)
        for j in range(nc - 1):
            A[:, j] = P[:, j] / P[:, j].sum()
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("fraction of the TRUE class (column)")
    else:
        V = np.where(np.isnan(P), np.nan, P + 1)
        im = ax.imshow(V, cmap="Blues", norm=LogNorm(vmin=1, vmax=np.nanmax(P) + 1))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("count (log colour scale)")
    for i in range(2):
        for j in range(nc):
            if np.isnan(P[i, j]):
                txt, dark = "-", False
            elif normalise and j < nc - 1:
                c = int(P[i, j])
                txt = f"{A[i, j]:.2f}\n({c})" if c else "0"
                dark = A[i, j] > 0.5
            elif normalise:
                c = int(P[i, j])
                txt = f"{c}\n(count)"
                dark = False
            else:
                c = int(P[i, j])
                txt = str(c)
                dark = np.log10(c + 1) > 0.6 * np.log10(np.nanmax(P) + 1)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if dark else "black")
    ax.set_xticks(range(nc), COLS)
    ax.set_yticks(range(2), ROWS)
    ax.set_xlabel("TRUE (ground truth)", fontsize=11, fontweight="bold")
    ax.set_ylabel("PREDICTED (detector output)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/s_detfly_new/weights/best.pt"))
    ap.add_argument("--split", default="test", choices=list(SOURCES))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default=str(ROOT / "figures"))
    args = ap.parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix

    match_iou = inspect.signature(ConfusionMatrix.process_batch).parameters["iou_thres"].default
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "confusion_matrix_drone_vs_none" + ("" if args.split == "test" else f"_{args.split}")
    src = SOURCES[args.split]
    groups = {s: g for s, (g, _) in src.items()}

    model = YOLO(args.weights)
    Ms, labels = {}, {}
    with tempfile.TemporaryDirectory() as tmp:             # Ultralytics' own plots and the source yamls go nowhere
        for s, (_, entry) in src.items():
            Ms[s] = validate(model, s, entry, args, tmp)
            labels[s] = label_counts(entry)
            print(f"{s}: {labels[s]['images']} images, matrix column totals {Ms[s][:, :4].sum(0).tolist()}", flush=True)
    P = pool(Ms, groups)

    # ---- checks
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    gt = {s: labels[s]["gt_boxes"] for s in src}
    gt_drone = {g: sum(gt[s]["drone"] for s in src if groups[s] == g) for g in ("dut", "detfly", "neg")}
    gt_neg = sum(gt[s][n] for s in src for n in NAMES[1:])
    check("match IoU used by the confusion matrix is 0.5", match_iou == 0.5, f"process_batch iou_thres = {match_iou}")
    check("every listed image exists on disk", all(labels[s]["images_missing_on_disk"] == 0 for s in src),
          {s: labels[s]["images_missing_on_disk"] for s in src})
    for s in src:
        check(f"{s}: 5x5 column totals = GT boxes per class in its label files",
              Ms[s][:, :4].sum(0).tolist() == [gt[s][n] for n in NAMES],
              f"matrix {Ms[s][:, :4].sum(0).tolist()} vs labels {[gt[s][n] for n in NAMES]}")
    check("negatives sources contain no drones", gt_drone["neg"] == 0, f"{gt_drone['neg']} GT drones in COCO / Open Images")
    dut_tot, df_tot, neg_tot = (int(np.nansum(P[:, j])) for j in range(3))
    check("DUT drone column total = GT drones in DUT", dut_tot == gt_drone["dut"], f"{dut_tot} vs {gt_drone['dut']}")
    check("Det-Fly drone column total = GT drones in the Det-Fly label files", df_tot == gt_drone["detfly"],
          f"{df_tot} vs {gt_drone['detfly']}")
    check("negatives column total = GT birds + airplanes + helicopters", neg_tot == gt_neg, f"{neg_tot} vs {gt_neg}")
    if args.split in EXPECTED:
        e = EXPECTED[args.split]
        check("DUT drone column total = 2,245", dut_tot == e["dut_drone"], f"{dut_tot}")
        check("negatives column total = 2,137 (2,142 label lines - 5 exact duplicates)", neg_tot == e["negatives"],
              f"{neg_tot}")
        raw_neg = sum(labels[s]["raw_label_lines"][n] for s in src for n in NAMES[1:])
        n_dup_neg = sum(v for s in src if groups[s] == "neg" for v in labels[s]["duplicates"].values())
        check("negatives label lines incl. duplicates = 2,142", raw_neg == e["negatives_raw_label_lines"],
              f"{raw_neg} lines, {n_dup_neg} exact duplicates")
    P2 = pool_by_matrices(Ms, groups)
    check("pooled cells = sums of the 5x5 cells (two independent computations)",
          np.array_equal(np.nan_to_num(P, nan=-1), np.nan_to_num(P2, nan=-1)), P.tolist())
    other_bg = int(sum(M[1:4, 4].sum() for M in Ms.values()))
    total = int(sum(M.sum() for M in Ms.values()))
    check("pooled cells + non-drone boxes on background = all 5x5 entries", int(np.nansum(P)) + other_bg == total,
          f"{int(np.nansum(P))} + {other_bg} vs {total}")

    # informational: per-source sum vs the earlier combined DUT + COCO + Open Images run (batching can differ)
    info = {}
    old = out / "confusion_matrix_drone_vs_none_dutonly.json"
    if args.split == "test" and old.exists():
        o = np.array(json.loads(old.read_text())["raw_5x5"])
        s3 = Ms["dut"] + Ms["coco"] + Ms["openimages"]
        info = {"same_as_earlier_combined_run": bool(np.array_equal(o, s3)),
                "cells_that_differ": int((o != s3).sum()), "max_abs_cell_difference": int(np.abs(o - s3).max()),
                "note": "The DUT-only figure validated DUT + COCO + Open Images in one call; here each source is "
                        "validated separately. Ultralytics letterboxes each batch to a shared rectangular shape, so "
                        "which images share a batch can change the padding and, rarely, a detection."}
        print("per-source sum vs earlier combined run:", info, flush=True)

    cell = lambda i, j: None if np.isnan(P[i, j]) else int(P[i, j])               # noqa: E731
    found, missed = cell(0, 0) + cell(0, 1), cell(1, 0) + cell(1, 1)
    res = {
        "index": "pooled[predicted][true]; raw_5x5_per_source[source][predicted][true]",
        "pooled_rows_predicted": ROWS, "pooled_cols_true": COLS_JSON,
        "pooled": [[cell(i, j) for j in range(4)] for i in range(2)],
        "pooled_named": {"dut_drones_found": cell(0, 0), "dut_drones_missed": cell(1, 0),
                         "detfly_drones_found": cell(0, 1), "detfly_drones_missed": cell(1, 1),
                         "negatives_called_drone": cell(0, 2), "negatives_not_called_drone": cell(1, 2),
                         "false_drone_boxes_on_background": cell(0, 3), "no_drone_on_background": None},
        "pooled_drones_dut_plus_detfly": {"found": found, "missed": missed, "total": found + missed,
                                          "recall": found / (found + missed)},
        "recall": {"dut": cell(0, 0) / dut_tot, "detfly": cell(0, 1) / df_tot},
        "pooled_from_5x5": {"drone (DUT) / drone (Det-Fly) found": "M_s[drone][drone] summed over the column's sources",
                            "missed": "M_s[bird|airplane|helicopter|background][drone] summed",
                            "negatives_called_drone": "M_s[drone][bird|airplane|helicopter] summed over all sources",
                            "negatives_not_called_drone": "M_s[bird|airplane|helicopter|background][bird|airplane|helicopter]",
                            "false_drone_boxes_on_background": "M_s[drone][background] summed over all sources",
                            "no_drone_on_background": "not defined"},
        "false_drone_boxes_on_background_per_source": {s: int(Ms[s][0, 4]) for s in src},
        "raw_5x5_names": NAMES + ["background"], "raw_5x5_per_source": {s: Ms[s].tolist() for s in src},
        "non_drone_boxes_on_background": other_bg,
        "weights": str(Path(args.weights).resolve().relative_to(ROOT)), "split": args.split,
        "source_data": {s: f"data/yolo/{e}" for s, (_, e) in src.items()},
        "detfly_definition": "detfly_split.json sets detfly_test010 (last 25 % of sequence 010) and detfly_test020 "
                             "(all of 020); same lists as the Det-Fly accuracy-table rows",
        "imgsz": args.imgsz, "conf": args.conf, "nms_iou": args.iou, "match_iou": match_iou,
        "images_per_source": {s: labels[s]["images"] for s in src},
        "images": sum(labels[s]["images"] for s in src),
        "gt_per_class_per_source": gt,
        "gt_label_lines_incl_duplicates_per_source": {s: labels[s]["raw_label_lines"] for s in src},
        "duplicate_label_lines": {s: labels[s]["duplicates"] for s in src if labels[s]["duplicates"]},
        "gt_note": "GT counts are after removing exact duplicate label lines, as Ultralytics does on load.",
        "checks": checks, "comparison_with_earlier_combined_run": info,
        "note": "Separate validation calls for the confusion matrix only, at conf 0.25. Do not quote mAP from them "
                "(mAP must be computed at low conf). Det-Fly 020 is a later part of the same site as the Det-Fly "
                "training frames (010), not an independent site.",
    }
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1) + "\n")
    other = " + ".join(f"{LABEL[s]} {labels[s]['images']:,}" for s in src if groups[s] != "detfly")
    detfly = " + ".join(f"{LABEL[s]} {labels[s]['images']:,}" for s in src if groups[s] == "detfly")
    settings = (f"{args.split.upper()} split: {other}\n+ {detfly} images\n"
                f"{Path(args.weights).parent.parent.name}/best.pt, conf {args.conf}, NMS IoU {args.iou}, "
                f"match IoU {match_iou}")
    plot(P, True, "Drone vs no drone, normalised by TRUE column (background: counts)\n" + settings, out / f"{stem}.png")
    plot(P, False, "Drone vs no drone, raw counts\n" + settings, out / f"{stem}_counts.png")
    print(json.dumps({k: res[k] for k in ("pooled_named", "pooled", "pooled_drones_dut_plus_detfly", "recall")}, indent=1))
    if not all(c["ok"] for c in checks):
        sys.exit("CHECK FAILED (see above)")


if __name__ == "__main__":
    main()
