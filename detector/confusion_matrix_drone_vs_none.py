"""'Drone vs no drone' confusion matrix for the slides, pooled from the usual 4-class + background matrix.

    python -I detector/confusion_matrix_drone_vs_none.py                 # TEST split (default)
    python -I detector/confusion_matrix_drone_vs_none.py --split val     # rerun on val (files get a _val suffix)

Same validation call and settings as detector/confusion_matrix_conf.py: Ultralytics val, imgsz 1280,
conf 0.25 (boxes with conf > 0.25 enter the matrix), NMS IoU 0.5, match IoU 0.5 (ConfusionMatrix.process_batch,
class-agnostic, highest confidence first). Default weights: the retrained YOLO26s (Det-Fly added,
runs/s_detfly_new, 106 epochs, best epoch 92). runs/detect/drone4_bal is the pre-Det-Fly s run.
Default data: data/yolo/drone4_bal.yaml, whose test split is DUT test + COCO test + Open Images test
(2,200 + 209 + 1,325 = 3,734 images) and whose val split also has no Det-Fly frames.

1. 5x5 raw matrix M[predicted][true], classes (drone, bird, airplane, helicopter, background):
     M[d, background] = predicted d where no object is (false positive box)
     M[background, t] = object of class t that no box matched (missed)
2. Pooled 2x3 matrix (rows predicted, columns true), every cell a sum of 5x5 cells:
     rows    drone     = a drone-class box;   no drone = a bird/airplane/helicopter box or no box
     columns drone, negatives (true bird + airplane + helicopter), background (box where nothing is)
     no drone / background is not defined ("-").
Writes figures/confusion_matrix_drone_vs_none{,_val}.json, .png (drone and negatives columns normalised by
column, background column as counts) and _counts.png (raw counts). Do not quote mAP from this run.
Exits with status 1 if a check fails (column totals vs label files, pooled vs 5x5).
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
NAMES = ["drone", "bird", "airplane", "helicopter"]
ROWS = ["drone", "no drone"]
COLS = ["drone", "negatives\n(bird, airplane,\nhelicopter)", "background\n(nothing there)"]
COLS_JSON = ["drone", "negatives (bird, airplane, helicopter)", "background (nothing there)"]
# Expected test totals. Negatives: 2,142 boxes in the label files, of which 5 are exact duplicates (one bird box
# written twice in 5 Open Images test files); Ultralytics drops exact duplicate labels on load, so it evaluates 2,137.
EXPECTED = {"test": {"drone": 2245, "negatives": 2137, "negatives_raw_label_lines": 2142}}


def pool(M):
    """Pooled 2x3 matrix (float, NaN = undefined) from the 5x5 [predicted, true] matrix, by explicit sums."""
    D, NEG, BG = 0, [1, 2, 3], 4
    P = np.full((2, 3), np.nan)
    P[0, 0] = M[D, D]                                   # drones found
    P[1, 0] = M[1:, D].sum()                            # drones missed (called bird/airplane/heli, or no box)
    P[0, 1] = M[D, NEG].sum()                           # negatives called drone
    P[1, 1] = M[1:, :][:, NEG].sum()                    # negatives not called drone (other class or no box)
    P[0, 2] = M[D, BG]                                  # false drone boxes on empty background
    return P


def pool_by_matrices(M):
    """Same pooling by aggregation matrices R (2x5) @ M @ C (5x3), as an independent check."""
    R = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 1, 1]])
    C = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
    P = (R @ M @ C).astype(float)
    P[1, 2] = np.nan                                    # contains bird/airplane/heli boxes on background: undefined
    return P


def label_counts(data_yaml, split):
    """Images and ground-truth boxes per class, per source folder, read from the YOLO label files.
    gt_boxes counts unique lines per file (Ultralytics removes exact duplicate labels when loading);
    raw_label_lines counts every line; duplicates lists the files that had exact duplicate lines."""
    d = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(d.get("path", ROOT / "data" / "yolo"))
    entries = d[split] if isinstance(d[split], list) else [d[split]]
    out = {}
    for e in entries:
        p = Path(e) if Path(e).is_absolute() else base / e
        if p.suffix == ".txt":                          # list file of image paths
            imgs = [Path(x.strip()) for x in p.read_text().splitlines() if x.strip()]
            src = p.stem
        else:
            imgs = sorted(q for q in p.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
            src = p.parts[-3]                           # <source>/images/<split>
        cnt, raw, dups = {n: 0 for n in NAMES}, {n: 0 for n in NAMES}, {}
        for im in imgs:
            lab = Path(str(im).replace("/images/", "/labels/")).with_suffix(".txt")
            if lab.exists():
                lines = [ln.strip() for ln in lab.read_text().splitlines() if ln.strip()]
                rows = [tuple(float(x) for x in ln.split()) for ln in lines]
                for r in rows:
                    raw[NAMES[int(r[0])]] += 1
                for r in set(rows):
                    cnt[NAMES[int(r[0])]] += 1
                if len(set(rows)) < len(rows):
                    dups[lab.name] = len(rows) - len(set(rows))
        out[src] = {"images": len(imgs), "gt_boxes": cnt, "raw_label_lines": raw, "duplicates": dups}
    return out


def plot(P, normalise, title, path):
    """Same figure size, colour map and text layout as confusion_matrix_conf.py (P is [predicted, true])."""
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    if normalise:
        A = np.full_like(P, np.nan)
        for j in (0, 1):
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
        for j in range(3):
            if np.isnan(P[i, j]):
                txt, dark = "-", False
            elif normalise and j < 2:
                c = int(P[i, j])
                txt = f"{A[i, j]:.2f}\n({c})" if c else "0"
                dark = A[i, j] > 0.5
            elif normalise:
                c = int(P[i, j])
                txt = f"{c}\n(count)" if c else "0\n(count)"
                dark = False
            else:
                c = int(P[i, j])
                txt = str(c)
                dark = np.log10(c + 1) > 0.6 * np.log10(np.nanmax(P) + 1)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if dark else "black")
    ax.set_xticks(range(3), COLS)
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
    ap.add_argument("--data", default=str(ROOT / "data/yolo/drone4_bal.yaml"))
    ap.add_argument("--split", default="test", choices=["test", "val"])
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

    model = YOLO(args.weights)
    with tempfile.TemporaryDirectory() as tmp:             # Ultralytics' own plots go nowhere
        r = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=8, conf=args.conf, iou=args.iou,
                      plots=True, project=tmp, name="cm", verbose=False)
    cm = r.confusion_matrix
    M = cm.matrix.astype(int)
    assert [cm.names[i] for i in range(cm.nc)] == NAMES, cm.names
    P = pool(M)

    # ---- checks
    labels = label_counts(args.data, args.split)
    gt = {n: sum(s["gt_boxes"][n] for s in labels.values()) for n in NAMES}
    n_img = sum(s["images"] for s in labels.values())
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("match IoU used by the confusion matrix is 0.5", match_iou == 0.5, f"process_batch iou_thres = {match_iou}")
    check("5x5 column totals = GT boxes per class in the label files",
          M[:, :4].sum(0).tolist() == [gt[n] for n in NAMES], f"matrix {M[:, :4].sum(0).tolist()} vs labels {[gt[n] for n in NAMES]}")
    drone_tot, neg_tot = int(np.nansum(P[:, 0])), int(np.nansum(P[:, 1]))
    check("drone column total = GT drones", drone_tot == gt["drone"], f"{drone_tot} vs {gt['drone']} GT drones")
    check("negatives column total = GT birds + airplanes + helicopters",
          neg_tot == gt["bird"] + gt["airplane"] + gt["helicopter"],
          f"{neg_tot} vs {gt['bird']} + {gt['airplane']} + {gt['helicopter']}")
    if args.split in EXPECTED:
        e = EXPECTED[args.split]
        check("drone total = expected (DUT test 2,245)", drone_tot == e["drone"], f"{drone_tot} vs {e['drone']}")
        check("negatives total = expected (2,137 = 2,142 label lines - 5 exact duplicates)", neg_tot == e["negatives"],
              f"{neg_tot} vs {e['negatives']}")
        raw_neg = sum(s["raw_label_lines"][n] for s in labels.values() for n in NAMES[1:])
        n_dup = sum(sum(s["duplicates"].values()) for s in labels.values())
        check("negatives label lines incl. duplicates = 2,142", raw_neg == e["negatives_raw_label_lines"] and
              raw_neg - n_dup == neg_tot, f"{raw_neg} lines, {n_dup} exact duplicates, {raw_neg - n_dup} evaluated")
    P2 = pool_by_matrices(M)
    check("pooled cells = sums of the 5x5 cells (two independent computations)",
          np.array_equal(np.nan_to_num(P, nan=-1), np.nan_to_num(P2, nan=-1)), P.tolist())
    other_bg = int(M[1:4, 4].sum())
    check("pooled cells + non-drone boxes on background = all 5x5 entries",
          int(np.nansum(P)) + other_bg == int(M.sum()), f"{int(np.nansum(P))} + {other_bg} vs {int(M.sum())}")

    cell = lambda i, j: None if np.isnan(P[i, j]) else int(P[i, j])               # noqa: E731
    res = {
        "index": "pooled[predicted][true]; raw_5x5[predicted][true]",
        "pooled_rows_predicted": ROWS, "pooled_cols_true": COLS_JSON,
        "pooled": [[cell(i, j) for j in range(3)] for i in range(2)],
        "pooled_named": {"drones_found": cell(0, 0), "drones_missed": cell(1, 0),
                         "negatives_called_drone": cell(0, 1), "negatives_not_called_drone": cell(1, 1),
                         "false_drone_boxes_on_background": cell(0, 2), "no_drone_on_background": None},
        "pooled_from_5x5": {"drones_found": "M[drone][drone]",
                            "drones_missed": "M[bird][drone] + M[airplane][drone] + M[helicopter][drone] + M[background][drone]",
                            "negatives_called_drone": "M[drone][bird] + M[drone][airplane] + M[drone][helicopter]",
                            "negatives_not_called_drone": "sum of M[bird|airplane|helicopter|background][bird|airplane|helicopter]",
                            "false_drone_boxes_on_background": "M[drone][background]",
                            "no_drone_on_background": "not defined"},
        "raw_5x5_names": NAMES + ["background"], "raw_5x5": M.tolist(),
        "non_drone_boxes_on_background": other_bg,
        "weights": str(Path(args.weights).resolve().relative_to(ROOT)), "data": Path(args.data).name, "split": args.split,
        "imgsz": args.imgsz, "conf": args.conf, "nms_iou": args.iou, "match_iou": match_iou,
        "images": n_img, "images_per_source": {k: v["images"] for k, v in labels.items()},
        "gt_per_class": gt, "gt_per_class_per_source": {k: v["gt_boxes"] for k, v in labels.items()},
        "gt_label_lines_incl_duplicates_per_source": {k: v["raw_label_lines"] for k, v in labels.items()},
        "duplicate_label_lines": {k: v["duplicates"] for k, v in labels.items() if v["duplicates"]},
        "gt_note": "GT counts are after removing exact duplicate label lines, as Ultralytics does on load (5 Open Images "
                   "test files each contain one bird box twice: 2,142 negative label lines, 2,137 evaluated).",
        "checks": checks,
        "note": "Separate validation call for the confusion matrix only, at conf 0.25. Do not quote mAP from it "
                "(mAP must be computed at low conf). Det-Fly is not in this split; it has its own row in the "
                "accuracy table.",
    }
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1) + "\n")
    srcs = " + ".join(f"{k.replace('dut_anti_uav', 'DUT').replace('coco', 'COCO').replace('openimages', 'Open Images')}"
                      f" {v['images']:,}" for k, v in labels.items())
    settings = (f"{args.split.upper()} split: {srcs} images\n"
                f"{Path(args.weights).parent.parent.name}/best.pt, conf {args.conf}, NMS IoU {args.iou}, match IoU {match_iou}")
    plot(P, True, "Drone vs no drone, normalised by TRUE column (background: counts)\n" + settings, out / f"{stem}.png")
    plot(P, False, "Drone vs no drone, raw counts\n" + settings, out / f"{stem}_counts.png")
    print(json.dumps({k: res[k] for k in ("pooled_named", "pooled", "raw_5x5", "gt_per_class", "images_per_source")}, indent=1))
    if not all(c["ok"] for c in checks):
        sys.exit("CHECK FAILED (see above)")


if __name__ == "__main__":
    main()
