"""Write presentation_assets_v2/{detector_results.md, model_size_table.md/.json, NUMBERS.md}.

    python scripts/new_slide_assets.py

Pure bookkeeping (no GPU): reads the JSONs produced by the retrain queue and by
scripts/new_final_assets.sh and puts every slide number side by side as OLD -> NEW with its source.
Missing inputs are shown as "n/a" so it can be re-run at any stage.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "presentation_assets_v2"
EV = ROOT / "detfly_retrain" / "eval"
NEWSIM = ROOT / "experiments" / "new_results"


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def ev(name):
    return J(EV / f"{name}.json")


def g(d, *keys, fmt="{:.3f}"):
    try:
        for k in keys:
            d = d[k]
        return fmt.format(d) if fmt else d
    except (KeyError, TypeError, IndexError):
        return "n/a"


def pct(x):
    return "n/a" if x in (None, "n/a") else f"{100 * x:.1f}%"


def detfly_cell(d, k):
    if not d:
        return "n/a"
    lo, hi = d["ci95"][k]
    return f"{d['point'][k]:.3f} [{lo:.3f}, {hi:.3f}]"


def fp(d, key="images_with_fp"):
    return g(d, "groups", "all/any", "0.25", key, fmt="{}")


def detector_results(lat):
    L = ["# Detector results — NEW YOLO26s (Det-Fly frames added to training)", "",
         "Same layout as `presentation_assets/detector_results.md`; the OLD s value is given next to each NEW value. "
         "P/R in the accuracy table are Ultralytics' values at the F1-optimal confidence per class (as before). "
         "Det-Fly rows use P/R/F1 at conf 0.25 with 95 % block-bootstrap CIs (see DETFLY_RETRAIN_RESULTS.md).", ""]
    L += ["## Accuracy", "", "| test set | class | P new (old) | R new (old) | mAP50 new (old) | mAP50-95 new (old) |",
          "|---|---|---|---|---|---|"]
    for tag, title in (("dut_test", "DUT Anti-UAV test (in-domain)"), ("all4_test", "DUT + COCO + Open Images test")):
        n, o = ev(f"new_s_{tag}"), ev(f"old_s_{tag}")
        L.append(f"| {title} | all | " + " | ".join(f"{g(n, k)} ({g(o, k)})" for k in ("P", "R", "mAP50", "mAP50-95")) + " |")
        for c in (n or o or {"per_class": {}})["per_class"]:
            L.append(f"| | {c} | " + " | ".join(f"{g(n, 'per_class', c, k)} ({g(o, 'per_class', c, k)})"
                                                for k in ("P", "R", "mAP50", "mAP50-95")) + " |")
    L += ["", "## Det-Fly (drone only, conf 0.25; 020 = held out, same site as the 010 training frames)", "",
          "| set | model | P | R | F1 | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|---|"]
    for st, title in (("test020", "Det-Fly 020 (6,912 frames)"), ("test010", "Det-Fly 010 test block (1,590)")):
        for m in ("new_s", "old_s"):
            d = ev(f"{m}_{st}")
            L.append(f"| {title} | {m.replace('_', ' ')} | " + " | ".join(detfly_cell(d, k) for k in
                     ("P@0.25", "R@0.25", "F1@0.25", "mAP50", "mAP50-95")) + " |")
    o = ev("old_s_prelim")
    L.append(f"| Det-Fly preliminary 1,000 (old slide) | old s | {g(o, 'point', 'P@0.25')} | {g(o, 'point', 'R@0.25')} | "
             f"{g(o, 'point', 'F1@0.25')} | {g(o, 'point', 'mAP50')} | {g(o, 'point', 'mAP50-95')} |")
    fn, fo = ev("new_s_false_positives"), ev("old_s_false_positives")
    L += ["", "## False 'drone' detections on images with no drone (conf 0.25 / 0.5)", "",
          "| images | conf | images | % with a false drone new (old) | false drones new (old) | on object new (old) |",
          "|---|---|---|---|---|---|"]
    if fn:
        for grp in sorted(fn["groups"]):
            for t, s in fn["groups"][grp].items():
                so = (fo or {}).get("groups", {}).get(grp, {}).get(t, {})
                L.append(f"| {grp} | {t} | {s['images']} | {100 * s['frac_images_with_fp']:.1f}% "
                         f"({100 * so.get('frac_images_with_fp', float('nan')):.1f}%) | {s['false_drones']} "
                         f"({so.get('false_drones', 'n/a')}) | {s['on_labelled_object']}/{s['false_drones']} "
                         f"({so.get('on_labelled_object', 'n/a')}/{so.get('false_drones', 'n/a')}) |")
    if lat:
        L += ["", "## Latency (DESKTOP GPU: RTX 3080, batch 1, 1920x1080 input, imgsz 1280) — REUSED", "",
              "Not re-measured: the new s has the identical YOLO26s architecture and input size, so the old "
              "measurement (`experiments/latency.json`) applies. Not representative of embedded hardware.", "",
              "| backend | mean ms | p50 ms | p95 ms | FPS | inference-only ms |", "|---|---|---|---|---|---|"]
        for k, v in lat.items():
            L.append(f"| {k} | {v['mean_ms']:.1f} | {v['p50_ms']:.1f} | {v['p95_ms']:.1f} | {v['fps']:.1f} | {v['inference_ms']:.1f} |")
    dn, do = J(V2 / "detector_noise.json"), J(ROOT / "experiments/detector_noise.json")
    if dn:
        L += ["", "## Box errors on matched true positives (DUT test) -> simulator noise (NEW s; old in brackets)", "",
              f"{dn['n_matched']} of {dn['n_gt']} test drones matched at conf >= {dn['conf']} (IoU >= {dn['match_iou']}) "
              f"[old: {do['n_matched']} of {do['n_gt']}]. σ = robust (1.4826·MAD); std includes outliers; tail = fraction beyond 3 robust σ.", "",
              "| true width px | n matched | recall | centre σ px (std) | log-width σ (std) | log-width bias | tail centre / width |",
              "|---|---|---|---|---|---|---|"]
        ob = {(b["w_lo"], b["w_hi"]): b for b in do["all_bins"]}
        for b in dn["all_bins"]:
            hi = "inf" if b["w_hi"] >= 10_000 else b["w_hi"]
            o = ob.get((b["w_lo"], b["w_hi"]), {})
            if "sigma_center_px" in b:
                L.append(f"| {b['w_lo']}-{hi} | {b['n_matched']} [{o.get('n_matched')}] | {b['recall']:.2f} | "
                         f"{b['sigma_center_px']:.2f} ({b['std_center_px']:.2f}) [{o.get('sigma_center_px', float('nan')):.2f}] | "
                         f"{b['sigma_logw']:.3f} ({b['std_logw']:.3f}) [{o.get('sigma_logw', float('nan')):.3f}] | "
                         f"{b['bias_logw']:+.3f} [{o.get('bias_logw', float('nan')):+.3f}] | "
                         f"{100 * b['tail_frac_center_gt3sigma']:.1f}% / {100 * b['tail_frac_logw_gt3sigma']:.1f}% |")
            else:
                rec = f"{b['recall']:.2f}" if b["recall"] is not None else "-"
                L.append(f"| {b['w_lo']}-{hi} | {b['n_matched']} | {rec} | too few | | | |")
    (V2 / "detector_results.md").write_text("\n".join(L) + "\n")


def model_size_table():
    bench = {r["size"]: r for r in csv.DictReader(open(ROOT / "model_size_benchmark/results.csv"))}
    rows, js = [], {}
    for s in ("n", "s", "m"):
        b = bench.get(s, {})
        for ver in ("old", "new"):
            t = f"{ver}_{s}"
            d20, d10, dut, a4, fpd = ev(f"{t}_test020"), ev(f"{t}_test010"), ev(f"{t}_dut_test"), ev(f"{t}_all4_test"), ev(f"{t}_false_positives")
            e = {"dut_drone_mAP50": g(dut, "mAP50", fmt=None), "all4_mAP50": g(a4, "mAP50", fmt=None),
                 "detfly020_R@0.25": g(d20, "point", "R@0.25", fmt=None), "detfly020_R_ci95": g(d20, "ci95", "R@0.25", fmt=None),
                 "detfly020_mAP50": g(d20, "point", "mAP50", fmt=None),
                 "detfly010_R@0.25": g(d10, "point", "R@0.25", fmt=None), "detfly010_mAP50": g(d10, "point", "mAP50", fmt=None),
                 "false_drone_images": fp(fpd), "params": b.get("fused_parameters"), "gflops_1280": b.get("gflops_1280"),
                 "pytorch_fp16_ms": b.get("pytorch_fp16_mean_ms"), "latency_note": "old measurement reused (same architecture)"}
            js[t] = e
            f = lambda x: "n/a" if x in ("n/a", None) else f"{float(x):.3f}"  # noqa: E731
            rows.append(f"| {ver} {s} | {int(b['fused_parameters']) / 1e6:.2f} M | {float(b['pytorch_fp16_mean_ms']):.1f} | "
                        f"{f(e['dut_drone_mAP50'])} | {f(e['all4_mAP50'])} | {detfly_cell(d20, 'R@0.25')} | "
                        f"{f(e['detfly020_mAP50'])} | {detfly_cell(d10, 'R@0.25')} | {e['false_drone_images']} |"
                        if b else f"| {ver} {s} | n/a |")
    L = ["# Model size table inputs (old vs new n / s / m)", "",
         "Latency and parameter counts are the old benchmark's (`model_size_benchmark/results.csv`): the architecture "
         "and input size are unchanged, so they were not re-measured (desktop RTX 3080, PyTorch FP16, end-to-end).", "",
         "| model | params (fused) | PyTorch FP16 ms | DUT drone mAP50 | all4 mAP50 | Det-Fly 020 R@0.25 [CI] | Det-Fly 020 mAP50 | "
         "Det-Fly 010-test R@0.25 [CI] | false-drone images /1,534 |", "|---|---|---|---|---|---|---|---|---|"] + rows
    (V2 / "model_size_table.md").write_text("\n".join(L) + "\n")
    (V2 / "model_size_table.json").write_text(json.dumps(js, indent=1, default=str) + "\n")


def sim_numbers():
    old, new = J(ROOT / "experiments/results_sim.json"), J(NEWSIM / "results_sim.json")
    onan, nnan = J(ROOT / "experiments/nan_runs.json"), J(NEWSIM / "nan_runs.json")
    rows = []

    def r(cfg, key, label, f=pct, sub=None):
        def val(d):
            try:
                x = d["results"][cfg][key] if sub is None else d["results"][cfg][key][sub]["rms_rel_err"]
                return f(x)
            except (KeyError, TypeError):
                return "n/a"
        rows.append((label, val(old), val(new), f"`experiments/results_sim.json` → `experiments/new_results/results_sim.json` [{cfg}]"))

    r("A_approach_weaving_tight_prior", "coverage_2sigma", "Weaving + tight prior: 2σ coverage (slide 22)")
    r("A_approach_weaving_tight_prior", "anees_range", "Weaving + tight prior: ANEES range (slide 22)", f=lambda x: f"{x:.2f}")
    r("A_approach_straight_size_prior", "coverage_2sigma", "Straight + size prior: 2σ coverage ('33 %', slide 23)")
    r("A_approach_straight_size_prior", "rms_rel_err_after10s", "Straight + size prior: RMS rel range error")
    r("A_approach_straight_size_prior", "median_reported_rel_sigma_after10s", "Straight + size prior: reported σ/r ('~7 %', slide 20)")
    r("A_approach_weaving_size_prior", "rms_rel_err_after10s", "Weaving + size prior: RMS rel range error")
    r("A_approach_weaving_size_prior", "median_reported_rel_sigma_after10s", "Weaving + size prior: reported σ/r ('4 %', slide 20)")
    r("A_approach_weaving_size_prior", "coverage_2sigma", "Weaving + size prior: 2σ coverage")
    for b in ("10-20m", "20-40m", "40-80m", "80-120m", "120-160m", "160-200m"):
        r("A_approach_weaving_size_prior", "by_range", f"Weaving + size prior: RMS rel err at {b} (slide 21)", sub=b)
    r("A_approach_straight_wrong_prior", "mean_rel_err_after10s", "Straight + wrong prior: bias ('27 % short', slide 23)")
    r("A_approach_weaving_wrong_prior", "mean_rel_err_after10s", "Weaving + wrong prior: bias")
    for rng in ("40m", "80m", "150m"):
        r(f"E_model_truth_weaving_{rng}", "anees_range", f"Model-truth weaving {rng}: ANEES range (slide 22)", f=lambda x: f"{x:.2f}")
        r(f"E_model_truth_weaving_{rng}", "anees_pos", f"Model-truth weaving {rng}: ANEES pos", f=lambda x: f"{x:.2f}")
    tot = lambda d: sum(v["nan_runs"] for v in d.values()) if d else "n/a"  # noqa: E731
    rows.append(("NaN runs (of 9,600) (slide 23)", str(tot(onan)), str(tot(nnan)),
                 "`experiments/nan_runs.json` → `experiments/new_results/nan_runs.json`"))
    return rows


def numbers(lat):
    L = ["# NUMBERS — every slide number / asset, OLD → NEW", "",
         "Read this first. NEW = YOLO26s retrained with 1,808 Det-Fly 010 frames added (same recipe otherwise). "
         "OLD = the deployed s from the earlier slides. Det-Fly 020 is the same site as the 010 training frames "
         "(later part of the same recording); say so on the slide. Sim numbers are SIMULATION ONLY; latency is DESKTOP GPU.", ""]
    rows = []
    ts = J(ROOT / "detfly_retrain/train_s.json")
    of = J(ROOT / "detfly_retrain/old_training_facts.json") or {}
    o = of.get("s", {})
    rows += [("Training: epochs run (slide 6)", g(o, "epochs_completed", fmt="{}"), g(ts, "epochs_completed", fmt="{}"), "`detfly_retrain/old_training_facts.json`, `detfly_retrain/train_s.json`"),
             ("Training: best epoch", g(o, "best_epoch", fmt="{}"), g(ts, "best_epoch", fmt="{}"), "same"),
             ("Training: wall-clock h", g(o, "csv_elapsed_hours", fmt="{:.2f}"), g(ts, "csv_elapsed_hours", fmt="{:.2f}"), "same (results.csv time column)"),
             ("Training: stop reason", g(o, "stop_reason", fmt="{}"), g(ts, "stop_reason", fmt="{}"), "training log"),
             ("Training images (DUT drone / negatives / Det-Fly)", "5,200 / 5,000 / 0", "5,200 / 5,000 / 1,808", "`presentation_assets_v2/dataset_stats.md`"),
             ("Drone share of training images (slide 5)", "51.0%", "58.3%", "`presentation_assets_v2/dataset_stats.md` Table 1")]
    for tag, lab in (("dut_test", "DUT test"), ("all4_test", "DUT+COCO+OI test")):
        n, od = ev(f"new_s_{tag}"), ev(f"old_s_{tag}")
        for k in ("P", "R", "mAP50", "mAP50-95"):
            rows.append((f"{lab}: {k} (overall, slide 7)", g(od, k), g(n, k), f"`detfly_retrain/eval/old_s_{tag}.json` → `new_s_{tag}.json` (detector/eval.py)"))
    n, od = ev("new_s_all4_test"), ev("old_s_all4_test")
    for c in ("drone", "bird", "airplane", "helicopter"):
        rows.append((f"all4 test {c}: R / mAP50", f"{g(od, 'per_class', c, 'R')} / {g(od, 'per_class', c, 'mAP50')}",
                     f"{g(n, 'per_class', c, 'R')} / {g(n, 'per_class', c, 'mAP50')}", "same"))
    fn, fo = ev("new_s_false_positives"), ev("old_s_false_positives")
    rows.append(("False-drone images at conf 0.25 (of 1,534) (slide 7)", fp(fo), fp(fn), "`detfly_retrain/eval/*_false_positives.json` (detector/false_positives.py)"))
    rows.append(("False drones total at conf 0.25", fp(fo, "false_drones"), fp(fn, "false_drones"), "same"))
    po = ev("old_s_prelim")
    rows.append(("Det-Fly PRELIMINARY 1,000 (old slide 8): P / R / mAP50 (Ultralytics max-F1)",
                 f"{g(po, 'ultralytics_drone', 'P_maxF1')} / {g(po, 'ultralytics_drone', 'R_maxF1')} / {g(po, 'ultralytics_drone', 'mAP50')}",
                 "not applicable (500 of these frames are now training data)", "`detfly_retrain/eval/old_s_prelim.json`"))
    for st, lab in (("test020", "Det-Fly 020 (held out, 6,912 frames)"), ("test010", "Det-Fly 010 test block (1,590)")):
        a, b = ev(f"old_s_{st}"), ev(f"new_s_{st}")
        for k in ("P@0.25", "R@0.25", "F1@0.25", "mAP50", "mAP50-95"):
            rows.append((f"{lab}: {k} [95% CI] (slide 8)", detfly_cell(a, k), detfly_cell(b, k),
                         f"`detfly_retrain/eval/old_s_{st}.json` → `new_s_{st}.json` (detector/new_eval_detfly.py)"))
    dn, do = J(V2 / "detector_noise.json"), J(ROOT / "experiments/detector_noise.json")

    def bin_(d, lo, k):
        for b_ in (d or {}).get("all_bins", []):
            if b_["w_lo"] == lo and k in b_:
                return b_[k]
        return None
    for lo, hi in ((16, 24), (24, 32), (32, 48)):
        rows.append((f"Box noise {lo}-{hi} px: centre σ px / log-width σ (slide 10)",
                     f"{bin_(do, lo, 'sigma_center_px') or float('nan'):.2f} / {bin_(do, lo, 'sigma_logw') or float('nan'):.3f}",
                     f"{bin_(dn, lo, 'sigma_center_px') or float('nan'):.2f} / {bin_(dn, lo, 'sigma_logw') or float('nan'):.3f}" if dn else "n/a",
                     "`experiments/detector_noise.json` → `presentation_assets_v2/detector_noise.json` (detector/measure_noise.py)"))
    rows.append(("Box noise: matched / GT drones", f"{g(do, 'n_matched', fmt='{}')} / {g(do, 'n_gt', fmt='{}')}",
                 f"{g(dn, 'n_matched', fmt='{}')} / {g(dn, 'n_gt', fmt='{}')}", "same"))
    if lat:
        for k in ("pytorch_fp16", "pytorch_fp32", "onnx_fp16", "onnx_fp32"):
            rows.append((f"Latency {k} mean ms (slide 9, DESKTOP GPU)", f"{lat[k]['mean_ms']:.1f}",
                         f"{lat[k]['mean_ms']:.1f} (reused: same architecture)", "`experiments/latency.json`"))
    rows += sim_numbers()
    cmo, cmn = J(ROOT / "presentation_assets/confusion_matrix_conf025.json"), J(V2 / "confusion_matrix_conf025.json")
    if cmo and cmn:
        bg = len(cmo["names"]) - 1
        rows.append(("Confusion @0.25 (old val split): false drones on background / missed drones",
                     f"{cmo['matrix'][0][bg]} / {cmo['matrix'][bg][0]}", f"{cmn['matrix'][0][bg]} / {cmn['matrix'][bg][0]}",
                     "`presentation_assets/confusion_matrix_conf025.json` → `presentation_assets_v2/…` (detector/confusion_matrix_conf.py)"))
    L += ["| number | OLD | NEW | source (file / command) |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    L += ["", "## Assets (same file names as presentation_assets/)", "",
          "| asset | what changed | produced by |", "|---|---|---|",
          "| PR_curve.png, confusion_matrix.png | new s validated on the OLD val split (COCO + DUT + Open Images val, 4,316 images) so they compare 1:1 with the old ones; `*_newval.png` = on the new val split incl. 377 Det-Fly frames | `scripts/new_final_assets.sh` (Ultralytics val plots) |",
          "| results.png | training curves of the new s run (`runs/s_detfly_new/results.png`) | Ultralytics training |",
          "| confusion_matrix_conf025{,_counts}.png/.json | conf 0.25, NMS IoU 0.5, old val split | `detector/confusion_matrix_conf.py --weights runs/s_detfly_new/weights/best.pt --out presentation_assets_v2` |",
          "| dut_good_small/medium/large.jpg | same DUT images as before, new model's boxes (old conf/IoU in the title) | `detector/new_presentation_examples.py` (`examples.json`) |",
          "| detfly_missed_1/2.jpg | Det-Fly 020 frames the NEW model misses; title says what the old model did | same |",
          "| dataset_stats.md/.json | composition incl. Det-Fly train/val/test | `scripts/new_dataset_stats.py` |",
          "| detector_results.md | new s numbers with old s in brackets | `scripts/new_slide_assets.py` |",
          "| detector_noise.json/.png | box noise re-measured with new s on DUT test | `detector/measure_noise.py --weights runs/s_detfly_new/weights/best.pt` |",
          "| model_size_table.md/.json | old vs new n/s/m | `scripts/new_slide_assets.py` |",
          "| rmse_vs_range, nees_time, range_error_envelopes, uncertainty_vs_motion, consistency_model_truth .png; results_sim.md; nan_runs.md | sim rerun with the new noise (also in `experiments/new_results/`) | `scripts/new_sim_shadow.sh` |",
          "", "Numbers read off figures only (not in any JSON), e.g. the 68th-percentile actual error on "
          "`uncertainty_vs_motion.png` (slide 20), must be re-read from the new figure."]
    (V2 / "NUMBERS.md").write_text("\n".join(L) + "\n")


def main():
    V2.mkdir(exist_ok=True)
    lat = J(ROOT / "experiments/latency.json")
    detector_results(lat)
    model_size_table()
    numbers(lat)
    print("wrote", V2 / "NUMBERS.md")


if __name__ == "__main__":
    main()
