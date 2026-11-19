"""Collect the detector result JSONs into experiments/detector_results.md.

    python detector/summarize.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def load(name):
    p = EXP / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    L = ["# Detector results", "",
         "YOLO26s, 4 classes (drone, bird, airplane, helicopter), imgsz 1280, trained on DUT Anti-UAV + "
         "COCO + Open Images (rebalanced, drones ~51% of training images). "
         "P/R are Ultralytics' values at the F1-optimal confidence per class.", ""]

    L += ["## Accuracy", "", "| test set | class | P | R | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|"]
    for tag, title in [("dut_test", "DUT Anti-UAV test (in-domain)"),
                       ("all4_test", "DUT + COCO + Open Images test"),
                       ("detfly_partial", "Det-Fly **PRELIMINARY** (partial: 1,000 of 13,271 images, sequence 010 only)")]:
        e = load(f"eval_{tag}.json")
        if e is None:
            L.append(f"| {title} | (not run) | | | | |")
            continue
        L.append(f"| {title} | all | {e['P']:.3f} | {e['R']:.3f} | {e['mAP50']:.3f} | {e['mAP50-95']:.3f} |")
        for c, v in e["per_class"].items():
            L.append(f"| | {c} | {v['P']:.3f} | {v['R']:.3f} | {v['mAP50']:.3f} | {v['mAP50-95']:.3f} |")

    fp = load("false_positives.json")
    if fp:
        L += ["", "## False 'drone' detections on images with no drone", "",
              "Held-out COCO and Open Images test images containing only birds / airplanes / helicopters. "
              "'on object' = the false drone box overlaps a labelled bird/aircraft (IoU >= 0.1).", "",
              "| images | conf | images | % images with a false drone | false drones / image | on object |",
              "|---|---|---|---|---|---|"]
        for g in sorted(fp["groups"]):
            for t, s in fp["groups"][g].items():
                L.append(f"| {g} | {t} | {s['images']} | {100 * s['frac_images_with_fp']:.1f}% | "
                         f"{s['false_drones_per_image']:.3f} | {s['on_labelled_object']}/{s['false_drones']} |")

    lat = load("latency.json")
    if lat:
        L += ["", "## Latency (DESKTOP GPU: RTX 3080, batch 1, 1920x1080 input, imgsz 1280)", "",
              "End-to-end `predict()` time including pre/post-processing. Not representative of embedded hardware.", "",
              "| backend | mean ms | p50 ms | p95 ms | FPS | inference-only ms |", "|---|---|---|---|---|---|"]
        for k, v in lat.items():
            L.append(f"| {k} | {v['mean_ms']:.1f} | {v['p50_ms']:.1f} | {v['p95_ms']:.1f} | {v['fps']:.1f} | "
                     f"{v['inference_ms']:.1f} |")

    dn = load("detector_noise.json")
    if dn:
        L += ["", "## Box errors on matched true positives (DUT test) -> simulator noise", "",
              f"{dn['n_matched']} of {dn['n_gt']} test drones matched at conf >= {dn['conf']} (IoU >= {dn['match_iou']}). "
              "σ = robust (1.4826·MAD); std includes outliers; tail = fraction beyond 3 robust σ.", "",
              "| true width px | n matched | recall | centre σ px (std) | log-width σ (std) | log-width bias | tail centre / width |",
              "|---|---|---|---|---|---|---|"]
        for b in dn["all_bins"]:
            hi = "inf" if b["w_hi"] >= 10_000 else b["w_hi"]
            if "sigma_center_px" in b:
                L.append(f"| {b['w_lo']}-{hi} | {b['n_matched']} | {b['recall']:.2f} | "
                         f"{b['sigma_center_px']:.2f} ({b['std_center_px']:.2f}) | {b['sigma_logw']:.3f} ({b['std_logw']:.3f}) | "
                         f"{b['bias_logw']:+.3f} | {100 * b['tail_frac_center_gt3sigma']:.1f}% / {100 * b['tail_frac_logw_gt3sigma']:.1f}% |")
            else:
                rec = f"{b['recall']:.2f}" if b["recall"] is not None else "-"
                L.append(f"| {b['w_lo']}-{hi} | {b['n_matched']} | {rec} | too few | | | |")
    (EXP / "detector_results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
