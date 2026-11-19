#!/usr/bin/env bash
# Full detector evaluation after training. Run from the repo root with the venv active:
#     bash detector/run_eval_all.sh runs/detect/drone4_bal/weights/best.pt
# Writes experiments/eval_*.json, false_positives.json, detector_noise.json, latency.json,
# detector_results.md and figures/detector_noise.png. Takes ~15-20 min on an RTX 3080.
set -euo pipefail
W=${1:-runs/detect/drone4_bal/weights/best.pt}
D=$(dirname "$W")

# 1. accuracy: DUT test (drones), all four classes (DUT+COCO+Open Images test), Det-Fly partial
python detector/eval.py --weights "$W" --data data/yolo/dut_anti_uav.yaml --tag dut_test
python detector/eval.py --weights "$W" --data data/yolo/drone4_bal.yaml --tag all4_test
python detector/eval.py --weights "$W" --data data/yolo/detfly.yaml --tag detfly_partial

# 2. false 'drone' alarms on bird / airplane / helicopter test images
python detector/false_positives.py --weights "$W"

# 3. real box errors vs box size -> sim/noise.py
python detector/measure_noise.py --weights "$W"

# 4. latency: PyTorch FP32/FP16, ONNX FP32/FP16 (onnxruntime-gpu, CUDA provider)
python detector/export_onnx.py --weights "$W" --imgsz 1280 && mv "$D/best.onnx" "$D/best_fp32.onnx"
python detector/export_onnx.py --weights "$W" --imgsz 1280 --half && mv "$D/best.onnx" "$D/best_fp16.onnx"
python detector/latency.py --weights "$W" --label pytorch_fp32
python detector/latency.py --weights "$W" --half --label pytorch_fp16
python detector/latency.py --weights "$D/best_fp32.onnx" --label onnx_fp32
python detector/latency.py --weights "$D/best_fp16.onnx" --half --label onnx_fp16

# 5. one markdown table with everything
python detector/summarize.py
