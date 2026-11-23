"""Continue training the retrained YOLO26m (runs/m_detfly_new, 49 epochs, stopped by its 7 h time cap
while still improving) as a NEW run, so the original run folder is never touched.

    flock /tmp/gpu.lock .venv/bin/python -I detector/cont_train_m.py 2>&1 | tee logs/train_m_cont.log

Recipe = runs/m_detfly_new/args.yaml verbatim (same data yaml / splits, imgsz 1280, batch 4, seed 0,
patience 30, default augmentation incl. close_mosaic 10, nbs 64, workers 12) with only these changes:
  model          runs/m_detfly_new/weights/best.pt (epoch 49 = its best and last epoch)
  name           m_detfly_cont (new folder, exist_ok=False)
  time, epochs   2 h cap, epochs 150 (the time cap ends the run; Ultralytics re-fits the epoch count
                 and the linear LR schedule to the time budget after every epoch)
  optimizer      MuSGD, set explicitly ("auto" picked MuSGD for the original run but forces lr0 = 0.01)
  lr0            0.002 (lrf stays 0.01, so the schedule floor is 2e-5; MuSGD runs the detection-head
                 groups at 3x lr)
  warmup_epochs  1
  warmup_bias_lr 0.0 -- "auto" sets this to 0 internally, so the original run had no bias-lr spike;
                 with an explicit optimizer the default would be 0.1 (50x lr0) during warmup.
Momentum stays at its default 0.937, which is what the original run used after warmup (auto builds the
optimizer with 0.9, but the warmup then ramps momentum to args.momentum = 0.937).

Writes detfly_retrain/train_m_cont.json: epochs, hours, best epoch (max val fitness = mAP50-95 in this
Ultralytics version), the hyper-parameters actually applied (from the log), and the final learning rates.
"""
import csv
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector.new_train_detfly import sha  # noqa: E402

SRC = ROOT / "runs" / "m_detfly_new"
NAME = "m_detfly_cont"
RUN = ROOT / "runs" / NAME
LOG = ROOT / "logs" / "train_m_cont.log"
OUT = ROOT / "detfly_retrain" / "train_m_cont.json"
OVERRIDES = dict(time=2.0, epochs=150, optimizer="MuSGD", lr0=0.002, warmup_epochs=1.0, warmup_bias_lr=0.0)


def recipe():
    rec = yaml.safe_load((SRC / "args.yaml").read_text())
    for k in ("save_dir", "model", "project", "name", "exist_ok", "resume"):
        rec.pop(k, None)
    rec.update(OVERRIDES, project=str(ROOT / "runs"), name=NAME, exist_ok=False)
    return rec


def facts():
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(RUN / "results.csv"))]
    fit = lambda r: float(r["metrics/mAP50-95(B)"])                                # noqa: E731
    best = max(rows, key=fit)
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", LOG.read_text(errors="ignore")) if LOG.exists() else ""
    a = yaml.safe_load((RUN / "args.yaml").read_text())
    done = re.findall(r"(\d+) epochs completed in ([\d.]+) hours", text)
    lr_cols = [k for k in rows[-1] if k.startswith("lr/")]
    src_rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(SRC / "results.csv"))]
    return {
        "run": str(RUN.relative_to(ROOT)), "start_weights": str((SRC / "weights/best.pt").relative_to(ROOT)),
        "start_weights_sha256": sha(SRC / "weights/best.pt"),
        "start_weights_val_mAP50-95": fit(src_rows[-1]), "start_weights_val_mAP50": float(src_rows[-1]["metrics/mAP50(B)"]),
        "epochs_completed": int(rows[-1]["epoch"]),
        "csv_elapsed_hours": float(rows[-1]["time"]) / 3600,
        "log_hours": float(done[-1][1]) if done else None,
        "best_epoch": int(best["epoch"]), "best_epoch_rule": "max val mAP50-95 (Ultralytics 8.4 fitness; selects best.pt)",
        "best_val_mAP50-95": fit(best), "best_val_mAP50": float(best["metrics/mAP50(B)"]),
        "val_per_epoch": [{"epoch": int(r["epoch"]), "mAP50": float(r["metrics/mAP50(B)"]), "mAP50-95": fit(r)}
                          for r in rows],
        "final_lr": {k: float(rows[-1][k]) for k in lr_cols},
        "lr_floor": {"base_groups": a["lr0"] * a["lrf"], "head_groups_3x": 3 * a["lr0"] * a["lrf"]},
        "applied_args": {k: a.get(k) for k in ("model", "data", "epochs", "time", "optimizer", "lr0", "lrf", "momentum",
                                               "warmup_epochs", "warmup_bias_lr", "warmup_momentum", "batch", "nbs",
                                               "imgsz", "seed", "patience", "close_mosaic", "mosaic", "workers")},
        "log_optimizer_lines": re.findall(r"optimizer: .*", text),
        "log_stop_lines": re.findall(r".*(?:epochs completed|Stopping training|EarlyStopping|time).*hours.*", text)[-3:],
        "batch_reductions": re.findall(r"CUDA out of memory with batch=(\d+)\. Reducing to batch=(\d+)", text),
        "best_pt_sha256": sha(RUN / "weights/best.pt"),
    }


def main():
    if "--facts-only" not in sys.argv:
        from ultralytics import YOLO
        rec = recipe()
        print("RECIPE", json.dumps(rec, sort_keys=True), flush=True)
        YOLO(str(SRC / "weights" / "best.pt")).train(**rec)
    f = facts()
    OUT.write_text(json.dumps(f, indent=1) + "\n")
    print(json.dumps({k: v for k, v in f.items() if k != "val_per_epoch"}, indent=1))


if __name__ == "__main__":
    main()
