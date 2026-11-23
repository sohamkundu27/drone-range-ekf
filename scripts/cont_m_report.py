"""Report for the continued YOLO26m run (detector/cont_train_m.py + detector/cont_eval_m.py).

    python -I scripts/cont_m_report.py

Side by side, all with the same eval scripts and test splits as DETFLY_RETRAIN_RESULTS.md:
  m, 49 ep      runs/m_detfly_new   ("new m" in DETFLY_RETRAIN_RESULTS.md; tag new_m)
  m, continued  runs/m_detfly_cont  (started from the 49-epoch best.pt; tag cont_m)
  s             runs/s_detfly_new   ("new s" in DETFLY_RETRAIN_RESULTS.md; tag new_s)
Writes experiments/detector_results_m_cont.md, detfly_retrain/m_cont_summary.json and copies the new
run's results.png, PR curve and confusion matrices to detfly_retrain/m_cont/. Overwrites nothing else.
"""
import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "detfly_retrain" / "eval"
RUN = ROOT / "runs" / "m_detfly_cont"
PLOTS = ROOT / "detfly_retrain" / "m_cont"
MODELS = [("new_m", "m, 49 epochs (retrained)"), ("cont_m", "m, continued"), ("new_s", "s (retrained)")]


def J(name):
    return json.loads((EV / f"{name}.json").read_text())


def f3(x):
    return f"{x:.3f}"


def ci(d, k):
    lo, hi = d["ci95"][k]
    return f"{d['point'][k]:.3f} [{lo:.3f}, {hi:.3f}]"


def fp(tag):
    g = J(f"{tag}_false_positives")["groups"]["all/any"]["0.25"]
    return g["images_with_fp"], g["images"], g["false_drones"]


def lr_table(run_dir, lr0):
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(run_dir / "results.csv"))]
    last = rows[-1]
    base, head = float(last["lr/pg1"]), float(last["lr/pg0"])     # pg1 = regular group, pg0 = 3x head group
    return {"epoch": int(last["epoch"]), "lr_pg0_head3x": head, "lr_pg1_base": base, "lr0": lr0,
            "schedule_factor": base / lr0}


def main():
    tc = json.loads((ROOT / "detfly_retrain" / "train_m_cont.json").read_text())
    tm = json.loads((ROOT / "detfly_retrain" / "train_m.json").read_text())
    ts = json.loads((ROOT / "detfly_retrain" / "train_s.json").read_text())
    lr_new = lr_table(RUN, tc["applied_args"]["lr0"])
    lr_old = lr_table(ROOT / "runs" / "m_detfly_new", 0.01)
    log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", (ROOT / "logs" / "train_m_cont.log").read_text(errors="ignore"))
    log = log.replace("\r", "\n")
    mosaic_closed = re.findall(r"Closing dataloader mosaic", log)
    # epoch at which mosaic was closed: the first results.csv epoch after the message, from the epoch header lines
    m_close = re.search(r"Closing dataloader mosaic[\s\S]*?\n\s*(\d+)/(\d+)\s", log)

    # best.pt must be the max-val-fitness epoch: compare its saved metrics to that results.csv row
    import torch
    ck = torch.load(RUN / "weights" / "best.pt", map_location="cpu", weights_only=False)
    tm_ck = ck.get("train_metrics", {})
    best_row = next(r for r in tc["val_per_epoch"] if r["epoch"] == tc["best_epoch"])
    best_pt_ok = abs(float(tm_ck.get("metrics/mAP50-95(B)", -1)) - best_row["mAP50-95"]) < 1e-4

    summ = {"train_m_cont": {k: v for k, v in tc.items() if k != "val_per_epoch"},
            "final_lr_new": lr_new, "final_lr_old_m_49ep": lr_old,
            "best_pt_matches_best_val_epoch": best_pt_ok, "mosaic_closed_at": m_close.groups() if m_close else None}
    for tag, _ in MODELS:
        summ[tag] = {s: J(f"{tag}_{s}") for s in ("test020", "test010", "dut_test", "all4_test")}
        summ[tag]["false_positives_all_any_0.25"] = dict(zip(("images_with_fp", "images", "false_drones"), fp(tag)))
        for s in ("test020", "test010"):
            summ[tag][s] = {k: summ[tag][s][k] for k in ("point", "ci95", "frames", "drone_boxes")}

    ep_total = tm["epochs_completed"] + tc["epochs_completed"]
    L = ["# Detector results — YOLO26m continued training", "",
         f"YOLO26m, 4 classes (drone, bird, airplane, helicopter), imgsz 1280. The retrained m (Det-Fly added; "
         f"`runs/m_detfly_new`) stopped at its 7 h time cap after {tm['epochs_completed']} epochs with its best "
         f"epoch = its last ({tm['best_epoch']}). This run continues from that best.pt as a NEW run "
         f"(`runs/m_detfly_cont`) for a {tc['applied_args']['time']} h time cap: {tc['epochs_completed']} more epochs "
         f"({ep_total} in total), {tc['csv_elapsed_hours']:.2f} h. Same data yaml and splits (12,008 train / 4,693 val), "
         "batch 4, seed 0, patience 30, default augmentation. Test sets and scripts identical to "
         "DETFLY_RETRAIN_RESULTS.md.", "",
         "The best epoch was chosen on VALIDATION fitness only (Ultralytics best.pt, val mAP50-95); each test split "
         "was run once on that checkpoint.", "",
         "## Side by side", "",
         "DUT test = DUT Anti-UAV test (2,200 images, drones only). All-class and bird = DUT + COCO + Open Images test "
         "(all4_test). P/R there are Ultralytics' values at the F1-optimal confidence per class (as in "
         "detector_results.md).", "",
         "| model | DUT drone P | DUT drone R | DUT drone mAP50 | DUT drone mAP50-95 | all-class mAP50 | "
         "all-class mAP50-95 | bird R | bird mAP50 | images with a false drone (of 1,534) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, label in MODELS:
        d, a = J(f"{tag}_dut_test"), J(f"{tag}_all4_test")
        dd = d["per_class"]["drone"]
        n_fp, n_img, _ = fp(tag)
        L.append(f"| {label} | {f3(dd['P'])} | {f3(dd['R'])} | {f3(dd['mAP50'])} | {f3(dd['mAP50-95'])} | "
                 f"{f3(a['mAP50'])} | {f3(a['mAP50-95'])} | {f3(a['per_class']['bird']['R'])} | "
                 f"{f3(a['per_class']['bird']['mAP50'])} | {n_fp} |")
    L += ["", "Det-Fly (drone only; P/R at conf 0.25 with IoU >= 0.5; 95% block-bootstrap CI, 100-frame chunks, "
              "1,000 resamples). 020 = 6,912 frames, a later part of the same site, never trained or validated on.", "",
          "| model | Det-Fly set | P @0.25 [95% CI] | R @0.25 [95% CI] | mAP50 [95% CI] | mAP50-95 [95% CI] |",
          "|---|---|---|---|---|---|"]
    for st, sl in (("test020", "020 (all frames)"), ("test010", "010 test block")):
        for tag, label in MODELS:
            d = J(f"{tag}_{st}")
            L.append(f"| {label} | {sl} | {ci(d, 'P@0.25')} | {ci(d, 'R@0.25')} | {ci(d, 'mAP50')} | {ci(d, 'mAP50-95')} |")

    L += ["", "## Accuracy (continued m), detector_results.md format", "",
          "| test set | class | P | R | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|"]
    d = J("cont_m_dut_test")
    L.append(f"| DUT Anti-UAV test (in-domain) | all | {f3(d['P'])} | {f3(d['R'])} | {f3(d['mAP50'])} | {f3(d['mAP50-95'])} |")
    dd = d["per_class"]["drone"]
    L.append(f"| | drone | {f3(dd['P'])} | {f3(dd['R'])} | {f3(dd['mAP50'])} | {f3(dd['mAP50-95'])} |")
    a = J("cont_m_all4_test")
    L.append(f"| DUT + COCO + Open Images test | all | {f3(a['P'])} | {f3(a['R'])} | {f3(a['mAP50'])} | {f3(a['mAP50-95'])} |")
    for c in ("drone", "bird", "airplane", "helicopter"):
        p = a["per_class"][c]
        L.append(f"| | {c} | {f3(p['P'])} | {f3(p['R'])} | {f3(p['mAP50'])} | {f3(p['mAP50-95'])} |")
    for st, sl in (("test020", "Det-Fly 020 (all 6,912 frames), conf 0.25"), ("test010", "Det-Fly 010 test block, conf 0.25")):
        p = J(f"cont_m_{st}")["point"]
        L.append(f"| {sl} | drone | {f3(p['P@0.25'])} | {f3(p['R@0.25'])} | {f3(p['mAP50'])} | {f3(p['mAP50-95'])} |")

    L += ["", "Latency: unchanged (same YOLO26m architecture); not re-measured.", "",
          "## Training", "",
          "| model | epochs run | best epoch (val) | best val mAP50-95 | wall-clock h | stop | final lr (regular / 3x head groups) | schedule factor at stop (floor = lrf 0.01) |",
          "|---|---|---|---|---|---|---|---|"]
    L.append(f"| m, 49 epochs (retrained) | {tm['epochs_completed']} | {tm['best_epoch']} | {tm['best_val_mAP50-95']:.4f} | "
             f"{tm['csv_elapsed_hours']:.2f} | {tm['stop_reason']} | {lr_old['lr_pg1_base']:.3g} / {lr_old['lr_pg0_head3x']:.3g} "
             f"(lr0 0.01) | {lr_old['schedule_factor']:.3f} |")
    L.append(f"| m, continued | {tc['epochs_completed']} (epochs {tm['epochs_completed'] + 1}-{ep_total} overall) | "
             f"{tc['best_epoch']} (= epoch {tm['epochs_completed'] + tc['best_epoch']} overall) | {tc['best_val_mAP50-95']:.4f} | "
             f"{tc['csv_elapsed_hours']:.2f} | time cap ({tc['applied_args']['time']} h) | {lr_new['lr_pg1_base']:.3g} / "
             f"{lr_new['lr_pg0_head3x']:.3g} (lr0 {lr_new['lr0']}) | {lr_new['schedule_factor']:.3f} |")
    L.append(f"| s (retrained) | {ts['epochs_completed']} | {ts['best_epoch']} | {ts['best_val_mAP50-95']:.4f} | "
             f"{ts['csv_elapsed_hours']:.2f} | {ts['stop_reason']} | - | - |")
    L += ["", f"Validation mAP50-95 per epoch of the continued run (start checkpoint: {tc['start_weights_val_mAP50-95']:.4f}):", "",
          "| epoch (continued run) | overall epoch | val mAP50 | val mAP50-95 |", "|---|---|---|---|"]
    L += [f"| {r['epoch']} | {tm['epochs_completed'] + r['epoch']} | {r['mAP50']:.4f} | {r['mAP50-95']:.4f} |"
          for r in tc["val_per_epoch"]]
    L += ["", "## Recipe and checks", "",
          "* Start: `runs/m_detfly_new/weights/best.pt` (SHA-256 `" + tc["start_weights_sha256"] + "`), epoch 49 = best "
          "and last epoch of that run (max val mAP50-95 in its results.csv).",
          "* Applied (from the run's args.yaml and log): " + ", ".join(f"{k}={v}" for k, v in tc["applied_args"].items()
                                                                    if k not in ("model", "data")) + ".",
          "* Log: `" + "`; `".join(tc["log_optimizer_lines"]) + "`. With `optimizer=auto` (original run) Ultralytics "
          "ignores lr0, uses 0.01 and also sets warmup_bias_lr = 0; the continued run sets MuSGD explicitly with "
          "lr0 = 0.002, and warmup_bias_lr = 0 to keep that behaviour (the default 0.1 would be 50x lr0 for the "
          "bias groups during warmup). Momentum 0.937 is the default and what the original run used after warmup.",
          f"* Final learning rate: with `time` set, Ultralytics re-fits the epoch count E and the linear LR schedule "
          f"to the time budget after every epoch. The linear schedule gives the last epoch (index E-1) the factor "
          f"lrf + (1 - lrf)/E, so the LR never reaches the lrf floor inside training: the floor is where the schedule "
          f"would be after one more epoch. Here E = {tc['epochs_completed']}: logged factor {lr_new['schedule_factor']:.3f} "
          f"= 0.01 + 0.99/{tc['epochs_completed']} = {0.01 + 0.99 / tc['epochs_completed']:.3f} (floor 0.01). Original "
          f"run: factor {lr_old['schedule_factor']:.3f}, consistent with E = 50 (0.01 + 0.99/50 = 0.030), i.e. the same "
          f"effect, not a run cut short before its schedule ended.",
          ("* close_mosaic = 10 (default) counts back from the re-fitted epoch count, so mosaic augmentation was "
           f"switched off from epoch {m_close.group(1)} of {m_close.group(2)} (re-fitted) of this run."
           if m_close else "* Mosaic was not closed during this run." if not mosaic_closed else
           "* close_mosaic = 10 (default): mosaic was switched off during this run (see log)."),
          f"* best.pt = epoch {tc['best_epoch']} of the continued run, the max-val-fitness epoch "
          f"({'verified against the checkpoint' if best_pt_ok else 'NOT matching the checkpoint metrics, check'}).",
          f"* best.pt SHA-256 `{tc['best_pt_sha256']}` (weights not committed).", "",
          "## Plots", "", "`detfly_retrain/m_cont/`: results.png (training curves), BoxPR_curve.png, confusion_matrix.png, "
          "confusion_matrix_normalized.png (all from the continued run's validation set)."]
    (ROOT / "experiments" / "detector_results_m_cont.md").write_text("\n".join(L) + "\n")
    (ROOT / "detfly_retrain" / "m_cont_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    PLOTS.mkdir(exist_ok=True)
    for f in ("results.png", "BoxPR_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
        shutil.copy2(RUN / f, PLOTS / f)
    print("\n".join(L))


if __name__ == "__main__":
    main()
