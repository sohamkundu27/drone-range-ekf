"""Confusion matrix at a deployment-like confidence threshold (for the slides).

    python -I detector/confusion_matrix_conf.py            # conf 0.25, NMS IoU 0.5

Why: the confusion_matrix.png that Ultralytics saved at the end of training was made at the
validator's default conf = 0.001, so it counts thousands of very low-confidence boxes. This
script makes a SEPARATE validation call on the same split (drone4_bal.yaml, split "val":
COCO + DUT Anti-UAV + Open Images val, 4,316 images) with conf = 0.25 and iou = 0.5, only to
get the confusion matrix. Do not quote mAP from this call (mAP must be computed at low conf).

Ultralytics' matrix is indexed [predicted, true]; the last row/column is "background":
  matrix[d, bg] = predicted class d where there is no object (false positive)
  matrix[bg, t] = true object of class t that no box matched (missed)
A prediction and a truth are matched at IoU >= 0.5 (class-agnostic, so cross-class
confusions are counted).

Writes (into presentation_assets/):
  confusion_matrix_conf025.png          normalised by TRUE class (each column sums to 1)
  confusion_matrix_conf025_counts.png   raw counts
  confusion_matrix_conf025.json         the raw matrix and the settings
"""
import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def plot(M, names, normalise, title, path):
    """M is [predicted, true]. Drawn with TRUE class on the x-axis, PREDICTED class on the y-axis."""
    n = len(names)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    if normalise:
        col = M.sum(axis=0, keepdims=True)
        A = np.where(col > 0, M / np.maximum(col, 1), 0.0)
        A[:, -1] = np.nan                       # true background has no total: shown as counts
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("fraction of the TRUE class (column)")
    else:
        im = ax.imshow(M + 1, cmap="Blues", norm=LogNorm(vmin=1, vmax=M.max() + 1))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("count (log colour scale)")
    for i in range(n):          # predicted (row)
        for j in range(n):      # true (column)
            c = int(M[i, j])
            if normalise and j < n - 1:
                txt = f"{A[i, j]:.2f}\n({c})" if c else "0"
                dark = A[i, j] > 0.5
            elif normalise:                     # background column: counts only
                txt = f"{c}\n(count)" if c else "-"
                dark = False
            else:
                txt = str(c) if c else "0"
                dark = np.log10(c + 1) > 0.6 * np.log10(M.max() + 1)
            if i == n - 1 and j == n - 1:
                txt = "-"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if dark else "black")
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    ax.set_xlabel("TRUE class (ground truth)", fontsize=11, fontweight="bold")
    ax.set_ylabel("PREDICTED class (detector output)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/drone4_bal/weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data/yolo/drone4_bal.yaml"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "presentation_assets"))
    args = ap.parse_args()
    out = Path(args.out)
    tag = f"conf{str(args.conf).replace('0.', '0')}"         # 0.25 -> conf025

    model = YOLO(args.weights)
    with tempfile.TemporaryDirectory() as tmp:                # Ultralytics' own plots go nowhere
        r = model.val(data=args.data, split="val", imgsz=1280, batch=8, conf=args.conf, iou=args.iou,
                      plots=True, project=tmp, name="cm", verbose=False)   # plots=True fills the matrix
    cmx = r.confusion_matrix
    M = cmx.matrix.astype(int)
    names = [cmx.names[i] for i in range(cmx.nc)] + ["background"]
    n_img = int(r.seen) if hasattr(r, "seen") else None

    settings = (f"validation split (COCO + DUT + Open Images val), best.pt, conf {args.conf}, "
                f"NMS IoU {args.iou}, match IoU 0.5")
    plot(M, names, True, "Confusion matrix, normalised by TRUE class (columns sum to 1)\n" + settings,
         out / f"confusion_matrix_{tag}.png")
    plot(M, names, False, "Confusion matrix, raw counts\n" + settings,
         out / f"confusion_matrix_{tag}_counts.png")
    (out / f"confusion_matrix_{tag}.json").write_text(json.dumps({
        "index": "matrix[predicted][true]", "names": names, "matrix": M.tolist(),
        "weights": str(Path(args.weights).relative_to(ROOT)), "data": Path(args.data).name, "split": "val",
        "conf": args.conf, "nms_iou": args.iou, "match_iou": 0.5,
        "true_instances_per_class": dict(zip(names[:-1], M[:, :-1].sum(0).tolist())),
        "note": "Separate validation call for the confusion matrix only. Do not quote mAP from it."}, indent=1))
    bg = len(names) - 1
    print("matrix [predicted, true]:\n", M)
    print("false drone predictions on true background:", M[0, bg])
    print("missed drones (true drone predicted as background):", M[bg, 0])


if __name__ == "__main__":
    main()
