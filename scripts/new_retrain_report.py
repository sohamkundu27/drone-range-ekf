"""Regenerate DETFLY_FULL_RESULTS.md and DETFLY_RETRAIN_RESULTS.md from detfly_retrain/ JSONs.

    python scripts/new_retrain_report.py

Works with partial results (missing entries are shown as "pending"). Also writes
detfly_retrain/summary.json (every number in the two tables) and detfly_retrain/old_training_facts.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))
from new_train_detfly import facts, sha  # noqa: E402

EV = ROOT / "detfly_retrain" / "eval"
SIZES = ["n", "s", "m"]
OLD_W = {"s": "runs/drone4_bal_backup.aVl2ZP/best.pt",
         "n": "model_size_benchmark/train/n_batch8/weights/best.pt",
         "m": "model_size_benchmark/train/m_batch4/weights/best.pt"}
OLD_RUN = {"s": ("runs/detect/drone4_bal", "runs/logs/train_drone4_bal.log", 8),
           "n": ("model_size_benchmark/train/n_batch8", "model_size_benchmark/logs/n_train_batch8.log", 8),
           "m": ("model_size_benchmark/train/m_batch4", "model_size_benchmark/logs/m_train_batch4.log", 4)}
SPLIT = json.loads((ROOT / "detfly_split.json").read_text())
# "not clearly worse" thresholds used when the decision rule is applied (stated in the report)
DUT_MAP50_TOL, FP_TOL = 0.01, 5


def J(name):
    p = EV / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(d, k):
    if not d:
        return "pending"
    lo, hi = d["ci95"][k]
    return f"{d['point'][k]:.3f} [{lo:.3f}, {hi:.3f}]"


def detfly_rows(tag, label):
    rows = []
    for st, name in (("test020", "020 (all frames)"), ("test010", "010 test block")):
        d = J(f"{tag}_{st}")
        if d:
            rows.append(f"| {label} | {name} | {d['frames']} | {ci(d, 'P@0.25')} | {ci(d, 'R@0.25')} | "
                        f"{ci(d, 'F1@0.25')} | {ci(d, 'mAP50')} | {ci(d, 'mAP50-95')} |")
        else:
            rows.append(f"| {label} | {name} | | pending | pending | pending | pending | pending |")
    return rows


DETFLY_HEAD = ["| model | Det-Fly set | frames | P @0.25 [95% CI] | R @0.25 [95% CI] | F1 @0.25 [95% CI] | "
               "mAP50 [95% CI] | mAP50-95 [95% CI] |", "|---|---|---|---|---|---|---|---|"]

HEADER_NOTES = [
    "> **Read first (choices made without asking, per instructions):**",
    "> * **020 is the same site as 010** (same hill, buildings, sports courts and dusk light in sampled frames; "
    "both 3840×2160). It is treated as a *later part of the same recording*, not an independent cross-site test. "
    "Det-Fly as a whole is still a different dataset/camera from the DUT training data.",
    "> * 13,270 Det-Fly frames: 010 = 6,358, 020 = 6,912. Frame 020/0207614 is missing from the source release "
    "(its XML exists, the JPEG does not); 020_0201689 has no UAV box (kept as a background frame), so 020 has "
    "6,911 drone boxes.",
    "> * Split (`detfly_split.json`): 010 cut at 60 %/75 % of its frames, 200-frame buffers dropped from the end "
    "of the train and val blocks, train and val subsampled with stride 2 → 1,808 train / 377 val frames; "
    "010 test = last 25 % (1,590 frames, all kept); 020 = 6,912 test frames, never trained or validated on.",
    "> * Protocol: Ultralytics val, imgsz 1280, batch 8, conf 0.001, NMS IoU 0.7 (as detector/eval.py), drone "
    "class only. P/R/F1 at confidence ≥ 0.25 with IoU ≥ 0.5 matching (recomputed from the validator's own "
    "matches). 95 % CI = block bootstrap, 100-frame time-ordered chunks, 1,000 resamples, percentile.",
]


def full_results(summary):
    L = ["# Det-Fly full-release results — OLD n / s / m models", ""] + HEADER_NOTES + [""]
    L += ["## Old models on full Det-Fly (drone class)", ""] + DETFLY_HEAD
    for s in SIZES:
        L += detfly_rows(f"old_{s}", f"old {s}")
    d = J("old_s_prelim")
    L += ["", "## Sanity check: old s on the 1,000-frame preliminary slice", ""]
    if d:
        u = d["ultralytics_drone"]
        L += ["| | P | R | mAP50 | mAP50-95 |", "|---|---|---|---|---|",
              "| earlier result (experiments/eval_detfly_partial.json) | 0.842 | 0.285 | 0.304 | 0.119 |",
              f"| re-run now (Ultralytics max-F1 P/R) | {u['P_maxF1']:.3f} | {u['R_maxF1']:.3f} | {u['mAP50']:.3f} | "
              f"{u['mAP50-95']:.3f} |",
              f"| re-run now, at conf 0.25 | {d['point']['P@0.25']:.3f} | {d['point']['R@0.25']:.3f} | "
              f"{d['point']['mAP50']:.3f} | {d['point']['mAP50-95']:.3f} |", "",
              "The preliminary slice is the first 1,000 frames of 010; 500 of them are now in the new training "
              "set, so it is used only for this old-model sanity check."]
    else:
        L += ["pending"]
    L += ["", "## Old weights", "", "| model | weights | SHA-256 |", "|---|---|---|"]
    for s in SIZES:
        L.append(f"| old {s} | `{OLD_W[s]}` | `{summary['sha'].get('old_' + s, 'missing')}` |")
    L += ["", "Produced by `scripts/run_retrain_queue.py` (stage old_eval) → `detector/new_eval_detfly.py`; "
          "JSONs in `detfly_retrain/eval/old_<size>_<set>.json`."]
    (ROOT / "DETFLY_FULL_RESULTS.md").write_text("\n".join(L) + "\n")


def fp_count(tag):
    d = J(f"{tag}_false_positives")
    if not d:
        return None, None
    g = d["groups"]["all/any"]["0.25"]
    return g["images_with_fp"], g["images"]


def dut(tag, kind="all4_test"):
    return J(f"{tag}_{kind}")


def decision(s):
    o20, n20 = J(f"old_{s}_test020"), J(f"new_{s}_test020")
    od, nd = dut(f"old_{s}", "dut_test"), dut(f"new_{s}", "dut_test")
    ofp, _ = fp_count(f"old_{s}")
    nfp, _ = fp_count(f"new_{s}")
    if None in (o20, n20, od, nd, ofp, nfp):
        return None
    recall_up = n20["point"]["R@0.25"] > o20["ci95"]["R@0.25"][1] and \
        n20["ci95"]["R@0.25"][0] > o20["point"]["R@0.25"]
    dmap = nd["mAP50"] - od["mAP50"]
    dfp = nfp - ofp
    ok = recall_up and dmap > -DUT_MAP50_TOL and dfp <= FP_TOL
    return {"recall020_old": o20["point"]["R@0.25"], "recall020_old_ci": o20["ci95"]["R@0.25"],
            "recall020_new": n20["point"]["R@0.25"], "recall020_new_ci": n20["ci95"]["R@0.25"],
            "recall_up_beyond_ci": recall_up, "dut_mAP50_old": od["mAP50"], "dut_mAP50_new": nd["mAP50"],
            "dut_mAP50_change": dmap, "false_drone_imgs_old": ofp, "false_drone_imgs_new": nfp,
            "false_drone_change": dfp, "rule_met": ok}


def retrain_results(summary, old_facts):
    L = ["# Det-Fly retrain — OLD vs NEW YOLO26 n / s / m", ""] + HEADER_NOTES
    L += ["> * NEW = same recipe as the old run (saved `args.yaml` of the old s run, as used by the old n/m benchmark), "
          "with 1,808 Det-Fly train frames added to training and 377 Det-Fly val frames added to validation "
          "(early stopping / best.pt selection therefore also sees Det-Fly val; the old runs did not).",
          "> * Decision rule (stated, not acted on): *new is clearly better if Det-Fly 020 recall is up beyond the "
          "CI AND DUT mAP50 and false-drone count are not clearly worse.* Applied here as: new 020 recall@0.25 above "
          "the old CI upper bound and new CI lower bound above the old point; DUT-test drone mAP50 drop < "
          f"{DUT_MAP50_TOL}; false-drone images increase ≤ {FP_TOL} (of 1,534).", ""]
    # plain-language summary
    L += ["## Plain-language summary", ""]
    any_new = False
    for s in ("s", "n", "m"):
        dec = summary["decision"].get(s)
        tr = summary["train_new"].get(s)
        if not tr:
            L.append(f"* **{s}**: new model not finished yet." if s not in summary["skipped"]
                     else f"* **{s}**: new model NOT trained (deadline); see STATUS.md.")
            continue
        any_new = True
        if dec:
            L.append(f"* **{s}**: Det-Fly 020 recall@0.25 {dec['recall020_old']:.3f} → {dec['recall020_new']:.3f}; "
                     f"DUT drone mAP50 {dec['dut_mAP50_old']:.3f} → {dec['dut_mAP50_new']:.3f}; false-drone images "
                     f"{dec['false_drone_imgs_old']} → {dec['false_drone_imgs_new']} of 1,534. Decision rule "
                     f"{'MET' if dec['rule_met'] else 'NOT met'}"
                     f"{'' if dec['rule_met'] else ' (' + ', '.join(x for x, c in (('020 recall not up beyond CI', not dec['recall_up_beyond_ci']), ('DUT mAP50 clearly worse', dec['dut_mAP50_change'] <= -DUT_MAP50_TOL), ('false drones clearly worse', dec['false_drone_change'] > FP_TOL)) if c) + ')'}."
                     + (" SHORTENED training run." if tr.get("shortened") else ""))
        else:
            L.append(f"* **{s}**: trained; evaluation pending.")
    if not any_new:
        L.append("* No new model finished yet; old-model numbers are in DETFLY_FULL_RESULTS.md.")
    L += ["", "## (c) Det-Fly drone, conf 0.25 (020 = later part of the same site; 010 test = held-out block)", ""]
    L += DETFLY_HEAD
    for s in SIZES:
        L += detfly_rows(f"old_{s}", f"old {s}") + detfly_rows(f"new_{s}", f"**new {s}**")
    L += ["", "## (a) DUT test — all 4 classes (all4_test: DUT + COCO + Open Images test; eval.py protocol)", "",
          "P/R are Ultralytics' values at the max-F1 confidence (as in all earlier eval_*.json).", "",
          "| model | class | P | R | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|"]
    for s in SIZES:
        for tag in (f"old_{s}", f"new_{s}"):
            d = dut(tag)
            lab = tag.replace("_", " ")
            if not d:
                L.append(f"| {lab} | all | pending | | | |")
                continue
            L.append(f"| {lab} | **overall** | {f3(d['P'])} | {f3(d['R'])} | {f3(d['mAP50'])} | {f3(d['mAP50-95'])} |")
            for c, v in d["per_class"].items():
                L.append(f"| {lab} | {c} | {f3(v['P'])} | {f3(v['R'])} | {f3(v['mAP50'])} | {f3(v['mAP50-95'])} |")
    L += ["", "DUT-only test (2,200 images, drones only; `dut_anti_uav.yaml`):", "",
          "| model | P | R | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for s in SIZES:
        for tag in (f"old_{s}", f"new_{s}"):
            d = dut(tag, "dut_test")
            L.append(f"| {tag.replace('_', ' ')} | " + (" | ".join(f3(d[k]) for k in ("P", "R", "mAP50", "mAP50-95"))
                                                       if d else "pending | | |") + " |")
    L += ["", "## (b) False drones (bird/airplane/helicopter-only test images with ≥ 1 drone detection at conf 0.25)",
          "", "| model | images with a false drone | of | false drones total |", "|---|---|---|---|"]
    for s in SIZES:
        for tag in (f"old_{s}", f"new_{s}"):
            d = J(f"{tag}_false_positives")
            if d:
                g = d["groups"]["all/any"]["0.25"]
                L.append(f"| {tag.replace('_', ' ')} | {g['images_with_fp']} | {g['images']} | {g['false_drones']} |")
            else:
                L.append(f"| {tag.replace('_', ' ')} | pending | | |")
    L += ["", "## (d) Training facts", "",
          "| model | epochs run | best epoch | wall-clock h | batch | time cap h | stop | train images (DUT drone / negatives / Det-Fly) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in SIZES:
        o = old_facts.get(s)
        if o:
            L.append(f"| old {s} | {o['epochs_completed']} | {o['best_epoch']} | {o['csv_elapsed_hours']:.2f} | "
                     f"{OLD_RUN[s][2]} | 7.0 | {o['stop_reason']} | 5,200 / 5,000 / 0 |")
        t = summary["train_new"].get(s)
        if t:
            L.append(f"| **new {s}** | {t['epochs_completed']} | {t['best_epoch']} | {t['csv_elapsed_hours']:.2f} | "
                     f"{t['requested']['batch']} | {t['time_cap_hours']}{' (SHORTENED)' if t.get('shortened') else ''} | "
                     f"{t['stop_reason']} | 5,200 / 5,000 / {SPLIT['sets']['detfly_train']['frames']:,} |")
        else:
            L.append(f"| new {s} | pending | | | | | | |")
    L += ["", "Recipe differences, honestly: old n, s and new n, s: batch 8; old m and new m: batch 4 (old m ran out "
          "of memory at batch 8; nbs = 64 keeps the effective batch at 64 via gradient accumulation). All runs: "
          "epochs 150, time cap 7 h unless marked SHORTENED, patience 30, imgsz 1280, seed 0, optimizer auto, "
          "default augmentation, COCO-pretrained start. With `time` set, Ultralytics re-fits the epoch count and LR "
          "schedule to the wall-clock budget, so a larger training set means fewer epochs in the same time. "
          "Old m hit the 7 h time cap (it did not early-stop).", "",
          "## Weights (never committed)", "", "| model | path | SHA-256 |", "|---|---|---|"]
    for s in SIZES:
        L.append(f"| old {s} | `{OLD_W[s]}` | `{summary['sha'].get('old_' + s, '—')}` |")
        if s in summary["train_new"]:
            L.append(f"| new {s} | `runs/{s}_detfly_new/weights/best.pt` | `{summary['train_new'][s]['best_pt_sha256']}` |")
    L += ["", "All numbers: `detfly_retrain/summary.json`, per-run JSONs in `detfly_retrain/eval/` and "
          "`detfly_retrain/train_<size>.json`. Produced by `scripts/run_retrain_queue.py`."]
    (ROOT / "DETFLY_RETRAIN_RESULTS.md").write_text("\n".join(L) + "\n")


def main():
    (ROOT / "detfly_retrain").mkdir(exist_ok=True)
    shaf = ROOT / "detfly_retrain" / "weights_sha256.json"
    shas = json.loads(shaf.read_text()) if shaf.exists() else {}
    for s in SIZES:
        if f"old_{s}" not in shas and (ROOT / OLD_W[s]).exists():
            shas[f"old_{s}"] = sha(ROOT / OLD_W[s])
    ofp = ROOT / "detfly_retrain" / "old_training_facts.json"
    if ofp.exists():
        old_facts = json.loads(ofp.read_text())
    else:
        old_facts = {s: facts(ROOT / r, ROOT / lg) for s, (r, lg, _) in OLD_RUN.items()}
        ofp.write_text(json.dumps(old_facts, indent=1) + "\n")
    train_new, skipped = {}, []
    mk = ROOT / "runs" / "retrain_markers"
    for s in SIZES:
        p = ROOT / "detfly_retrain" / f"train_{s}.json"
        if (mk / f"skipped_{s}.done").exists():
            skipped.append(s)
        if p.exists():
            t = json.loads(p.read_text())
            t["shortened"] = t["time_cap_hours"] < 7.0
            train_new[s] = t
            shas[f"new_{s}"] = t["best_pt_sha256"]
    shaf.parent.mkdir(parents=True, exist_ok=True)
    shaf.write_text(json.dumps(shas, indent=1) + "\n")
    summary = {"sha": shas, "train_new": train_new, "skipped": skipped,
               "decision": {s: decision(s) for s in SIZES}, "decision_tolerances":
               {"dut_mAP50_drop_lt": DUT_MAP50_TOL, "false_drone_increase_le": FP_TOL},
               "eval": {p.stem: json.loads(p.read_text()) for p in sorted(EV.glob("*.json"))} if EV.exists() else {}}
    full_results(summary)
    retrain_results(summary, old_facts)
    slim = {k: v for k, v in summary.items() if k != "eval"}
    slim["eval_points"] = {k: ({"point": v.get("point"), "ci95": v.get("ci95")} if "point" in v else
                               {kk: v.get(kk) for kk in ("P", "R", "mAP50", "mAP50-95")} if "mAP50" in v else
                               {"all_any_0.25": v["groups"]["all/any"]["0.25"]} if "groups" in v else v)
                           for k, v in summary["eval"].items()}
    (ROOT / "detfly_retrain" / "summary.json").write_text(json.dumps(slim, indent=1, default=str) + "\n")
    print("reports written")


if __name__ == "__main__":
    main()
