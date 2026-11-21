#!/usr/bin/env bash
# Re-run the simulation experiments with a given detector-noise JSON WITHOUT touching
# experiments/ or figures/ (their existing files are frozen for the old slides).
#
#     bash scripts/new_sim_shadow.sh <detector_noise.json> <tag> <out_dir>
#
# How: byte-identical copies of ekf/, sim/ and experiments/*.py are placed in
# runs/sim_shadow_<tag>/ (every script locates its inputs/outputs relative to its own file, so
# the copy reads <shadow>/experiments/detector_noise.json and writes into the shadow tree).
# No sim or EKF logic is changed; the known NaN behaviour is kept as is.
# Then copies results_sim.{md,json}, nan_runs.{md,json} and the five figures to <out_dir>.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
NOISE=$(realpath "$1"); TAG=$2; OUT=$(realpath -m "$3")
SH="$R/runs/sim_shadow_$TAG"
rm -rf "$SH"
mkdir -p "$SH/experiments" "$SH/figures" "$SH/runs" "$OUT"
rsync -a --exclude __pycache__ "$R/ekf" "$R/sim" "$SH/"
cp "$R"/experiments/*.py "$SH/experiments/"
cp "$NOISE" "$SH/experiments/detector_noise.json"
sha256sum "$R"/ekf/*.py "$R"/sim/*.py "$R"/experiments/*.py > "$SH/code_sha256.txt"
cd "$SH"
nice -n 10 "$R/.venv/bin/python" experiments/run_experiments.py
nice -n 10 "$R/.venv/bin/python" experiments/make_figures.py
nice -n 10 "$R/.venv/bin/python" experiments/check_nan_runs.py
cp experiments/results_sim.md experiments/results_sim.json experiments/nan_runs.md experiments/nan_runs.json "$OUT/"
cp figures/rmse_vs_range.png figures/nees_time.png figures/range_error_envelopes.png \
   figures/uncertainty_vs_motion.png figures/consistency_model_truth.png "$OUT/"
cp "$NOISE" "$OUT/detector_noise_used.json"
echo "sim ($TAG) done -> $OUT"
