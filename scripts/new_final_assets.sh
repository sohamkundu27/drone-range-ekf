#!/usr/bin/env bash
# Slide assets v2 for the NEW s model + simulation rerun with its measured noise.
#     bash scripts/new_final_assets.sh
# Called by scripts/run_retrain_queue.py, which already holds /tmp/gpu.lock around this whole
# script (so no flock in here). Never writes into presentation_assets/ or existing experiments/ files.
# Each step continues on failure; the exit code is non-zero if any step failed.
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
cd "$R"
PY="$R/.venv/bin/python"
V2=presentation_assets_v2
RUN=runs/s_detfly_new
NEW=$RUN/weights/best.pt
fail=0
step() { local n=$1; shift; echo "===== $n: $*"; "$@" || { echo "FAILED step $n"; fail=1; }; }
mkdir -p $V2 experiments/new_results
[ -f "$NEW" ] || { echo "no new s weights at $NEW"; exit 1; }

# training curves + the training run's own final-val plots (new val split incl. Det-Fly val)
step curves cp $RUN/results.png $V2/results.png
step newval_plots bash -c "cp $RUN/BoxPR_curve.png $V2/PR_curve_newval.png && cp $RUN/confusion_matrix.png $V2/confusion_matrix_newval.png"

# PR curve + conf-0.001 confusion matrix on the OLD val split (1:1 comparable with presentation_assets/)
step oldval_plots "$PY" -c "
from ultralytics import YOLO
YOLO('$NEW').val(data='data/yolo/drone4_bal.yaml', split='val', imgsz=1280, batch=8, conf=0.001, iou=0.7,
                 device=0, project='$R/runs/new_val', name='new_s_oldval', exist_ok=True, plots=True)"
step oldval_copy bash -c "cp runs/new_val/new_s_oldval/BoxPR_curve.png $V2/PR_curve.png && cp runs/new_val/new_s_oldval/confusion_matrix.png $V2/confusion_matrix.png"

# conf 0.25 confusion matrix (same script and split as the old asset)
step cm025 "$PY" detector/confusion_matrix_conf.py --weights "$R/$NEW" --out $V2

# example detections
step examples "$PY" detector/new_presentation_examples.py --new "$R/$NEW"

# box noise of the new s on held-out DUT drones (same method as before)
step noise "$PY" detector/measure_noise.py --weights "$NEW" --out $V2/detector_noise.json --fig $V2/detector_noise.png

# simulation with the new noise (shadow copy of the unchanged sim code)
step sim bash scripts/new_sim_shadow.sh $V2/detector_noise.json newnoise experiments/new_results
step sim_copy bash -c "cp experiments/new_results/results_sim.md experiments/new_results/nan_runs.md experiments/new_results/*.png $V2/"

# tables
step stats "$PY" scripts/new_dataset_stats.py
step numbers "$PY" scripts/new_slide_assets.py
echo "final assets done, fail=$fail"
exit $fail
