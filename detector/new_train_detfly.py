"""Retrain YOLO26 n/s/m with Det-Fly frames added (identical recipe to the old runs otherwise).

    python detector/new_train_detfly.py --size s --batch 8 --time 7.0

Recipe = the saved arguments of the old s run (runs/detect/drone4_bal/args.yaml, which the old n/m
benchmark runs also used verbatim apart from batch). Only these keys are changed:
  data    -> data/yolo/new_drone4_detfly.yaml (old train + Det-Fly train; old val + Det-Fly val)
  model   -> COCO-pretrained yolo26<size>.pt (same files the old runs started from)
  batch   -> 8 for n/s, 4 for m (old m ran at 4 after batch 8 ran out of memory)
  time    -> wall-clock cap in hours (old runs: 7.0); lower only if the deadline forces it
  project/name -> runs/<size>_detfly_new
If runs/<size>_detfly_new/weights/last.pt exists and the run is unfinished, it resumes (resume=True).
Writes detfly_retrain/train_<size>.json after training.
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = {"n": ROOT / "model_size_benchmark/pretrained/yolo26n.pt", "s": ROOT / "yolo26s.pt",
              "m": ROOT / "model_size_benchmark/pretrained/yolo26m.pt"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def facts(run, log):
    rows = list(csv.DictReader(open(run / "results.csv")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", Path(log).read_text(errors="ignore")) if Path(log).exists() else ""
    m_best = re.findall(r"Best results observed at epoch (\d+)", text)
    m_done = re.findall(r"(\d+) epochs completed in ([\d.]+) hours", text)
    best = max(rows, key=lambda r: 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]))
    a = yaml.safe_load((run / "args.yaml").read_text())
    reductions = re.findall(r"CUDA out of memory with batch=(\d+)\. Reducing to batch=(\d+)", text)
    return {"epochs_completed": int(rows[-1]["epoch"]),
            "best_epoch": int(m_best[-1]) if m_best else int(best["epoch"]),
            "best_epoch_source": "log EarlyStopping line" if m_best else "max fitness (0.1 mAP50 + 0.9 mAP50-95) in results.csv",
            "best_val_mAP50": float(best["metrics/mAP50(B)"]), "best_val_mAP50-95": float(best["metrics/mAP50-95(B)"]),
            "csv_elapsed_hours": float(rows[-1]["time"]) / 3600,
            "log_hours": float(m_done[-1][1]) if m_done else None,
            "stop_reason": "early stopping (patience)" if m_best else "time cap / epoch limit",
            "requested": {k: a.get(k) for k in ("epochs", "time", "patience", "batch", "imgsz", "seed", "nbs", "model", "data")},
            "batch_reductions": reductions,
            "sec_per_epoch_first5": [float(rows[i]["time"]) - (float(rows[i - 1]["time"]) if i else 0.0)
                                     for i in range(min(5, len(rows)))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(PRETRAINED))
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--time", type=float, required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--facts-only", action="store_true")
    args = ap.parse_args()
    name = f"{args.size}_detfly_new"
    run = ROOT / "runs" / name
    out = ROOT / "detfly_retrain" / f"train_{args.size}.json"

    if not args.facts_only:
        from ultralytics import YOLO
        last = run / "weights" / "last.pt"
        if last.exists():
            print("RESUMING", last)
            try:
                YOLO(str(last)).train(resume=True)
            except AssertionError as e:  # run already finished: nothing to resume, just record facts
                if "nothing to resume" not in str(e):
                    raise
                print("already finished:", e)
        else:
            rec = yaml.safe_load((ROOT / "runs/detect/drone4_bal/args.yaml").read_text())
            for k in ("save_dir", "model", "project", "name", "exist_ok", "resume"):
                rec.pop(k, None)
            rec.update(data=str(ROOT / "data/yolo/new_drone4_detfly.yaml"), batch=args.batch, time=args.time,
                       project=str(ROOT / "runs"), name=name, exist_ok=False)
            print("RECIPE", json.dumps(rec, sort_keys=True))
            YOLO(str(PRETRAINED[args.size])).train(**rec)

    f = facts(run, args.log)
    f.update(size=args.size, run=str(run.relative_to(ROOT)), pretrained=str(PRETRAINED[args.size].relative_to(ROOT)),
             pretrained_sha256=sha(PRETRAINED[args.size]),
             best_pt_sha256=sha(run / "weights/best.pt"), time_cap_hours=args.time)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(f, indent=1) + "\n")
    print(json.dumps(f, indent=1))


if __name__ == "__main__":
    main()
