"""Drone-only Det-Fly evaluation with a block-bootstrap CI (retrain experiment).

    python detector/new_eval_detfly.py --weights <best.pt> --model-tag old_s --set test020
    sets: test020 (all of 020), test010 (010 test block), prelim (old 1,000-frame slice)

Same validator call as detector/eval.py (split=test, imgsz 1280, batch 8, conf 0.001, NMS IoU 0.7).
While Ultralytics validates, the per-image matching stats it computes (tp at IoU .50:.95, conf, class)
are captured, so every metric below is recomputed from the validator's own matches:
  * P, R, F1 at conf >= 0.25, IoU 0.5, drone class only
  * mAP50, mAP50-95 (Ultralytics ap_per_class, drone only)
  * the Ultralytics-reported drone P/R (at its max-F1 confidence) for comparison with earlier results
  * 95 % block-bootstrap CI: frames in time order, cut into 100-frame chunks, chunks resampled
    with replacement (1,000 resamples, seed 0), percentile interval.
Writes detfly_retrain/eval/<model-tag>_<set>.json and runs/new_val/<tag>_<set>/per_image.npz.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YAML = {"test020": "new_detfly_test020", "test010": "new_detfly_test010", "prelim": "new_detfly_prelim"}
CONF = 0.25


def metrics(tp, conf, pred_cls, target_cls):
    from ultralytics.utils.metrics import ap_per_class
    keep = pred_cls == 0
    tp, conf = tp[keep], conf[keep]
    n_gt = int((target_cls == 0).sum())
    hi = conf >= CONF
    n_tp, n_pred = int(tp[hi, 0].sum()), int(hi.sum())
    p = n_tp / n_pred if n_pred else float("nan")
    r = n_tp / n_gt if n_gt else float("nan")
    f1 = 2 * p * r / (p + r) if n_pred and n_gt and (p + r) > 0 else 0.0
    if n_gt and len(conf):
        ap = ap_per_class(tp, conf, np.zeros(len(conf)), np.zeros(n_gt))[5][0]
        map50, map5095 = float(ap[0]), float(ap.mean())
    else:
        map50 = map5095 = 0.0
    return {"P@0.25": p, "R@0.25": r, "F1@0.25": f1, "mAP50": map50, "mAP50-95": map5095,
            "tp@0.25": n_tp, "pred@0.25": n_pred, "gt": n_gt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--set", required=True, choices=list(YAML))
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-dir", default=str(ROOT / "detfly_retrain" / "eval"))
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils.metrics import DetMetrics

    rec = []
    orig = DetMetrics.update_stats

    def update_stats(self, stat):
        rec.append({k: np.asarray(stat[k]).copy() for k in ("tp", "conf", "pred_cls", "target_cls")}
                   | {"im_name": stat["im_name"]})
        return orig(self, stat)

    DetMetrics.update_stats = update_stats
    name = f"{args.model_tag}_{args.set}"
    t0 = time.time()
    m = YOLO(args.weights).val(data=str(ROOT / "data" / "yolo" / f"{YAML[args.set]}.yaml"), split="test",
                               imgsz=1280, batch=8, conf=0.001, iou=0.7, device=0,
                               project=str(ROOT / "runs" / "new_val"), name=name, exist_ok=True, plots=True)
    DetMetrics.update_stats = orig
    secs = time.time() - t0

    rec.sort(key=lambda r: r["im_name"])  # <seq>_<frame id>: time order
    ult = {}
    for i, c in enumerate(m.box.ap_class_index):
        if int(c) == 0:
            ult = {"P_maxF1": float(m.box.p[i]), "R_maxF1": float(m.box.r[i]),
                   "mAP50": float(m.box.ap50[i]), "mAP50-95": float(m.box.ap[i])}

    def cat(rs):
        return (np.concatenate([r["tp"] for r in rs]).reshape(-1, 10).astype(bool),
                np.concatenate([r["conf"] for r in rs]), np.concatenate([r["pred_cls"] for r in rs]),
                np.concatenate([r["target_cls"] for r in rs]))

    point = metrics(*cat(rec))
    chunks = [cat(rec[i:i + args.chunk]) for i in range(0, len(rec), args.chunk)]
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(chunks), len(chunks))
        boot.append(metrics(*(np.concatenate([chunks[j][k] for j in idx]) for k in range(4))))
    ci = {k: [float(np.nanpercentile([b[k] for b in boot], 2.5)), float(np.nanpercentile([b[k] for b in boot], 97.5))]
          for k in ("P@0.25", "R@0.25", "F1@0.25", "mAP50", "mAP50-95")}
    np.savez_compressed(ROOT / "runs" / "new_val" / name / "per_image.npz",
                        im_name=np.array([r["im_name"] for r in rec]),
                        **{k: np.array([r[k] for r in rec], dtype=object) for k in ("tp", "conf", "pred_cls", "target_cls")})
    res = {"model_tag": args.model_tag, "set": args.set, "weights": args.weights,
           "data": f"data/yolo/{YAML[args.set]}.yaml", "frames": len(rec), "drone_boxes": point["gt"],
           "protocol": {"imgsz": 1280, "batch": 8, "val_conf": 0.001, "nms_iou": 0.7, "pr_conf": CONF,
                        "match_iou_for_PR": 0.5, "class": "drone only"},
           "point": point, "ci95": ci,
           "bootstrap": {"chunk_frames": args.chunk, "n_chunks": len(chunks), "resamples": args.n_boot,
                         "seed": 0, "method": "percentile"},
           "ultralytics_drone": ult, "seconds": secs}
    out = Path(args.out_dir) / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("point", "ci95", "ultralytics_drone")}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
