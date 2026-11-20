# YOLO26 model-size benchmark

**l and x were NOT run.** Only m and n are trained; existing s is reused. Results update after every stage.

## Protocol
Equal time budget, not equal epochs: each new model requests epochs=150, time=7.0 hours, patience=30. Existing s used that same budget and stopped early. All use seed=0, deterministic=false, imgsz=1280, batch=8, nbs=64, AMP, workers=12, optimizer=auto (s selected MuSGD), and the complete saved s args in sources/s_args.yaml. Only pretrained architecture, output locations, and the explicitly authorized m batch-4 OOM fallback may differ. A fallback restarts from pretrained weights in a separate run directory; failed-attempt duration is separately logged.
Data: original data/yolo/drone4_bal.yaml, unchanged. Best checkpoint selected on combined validation mAP50–95. Test evaluation: DUT, combined, partial 1,000-image Det-Fly; batch=8, imgsz=1280, conf=0.001, NMS IoU=0.7, GPU 0. No evaluation batch fallback. P/R use the library mean-F1-selected confidence, not conf 0.25.
False drone count: existing false_positives.py on COCO/Open Images test, conf=0.25. Latency: frozen original latency.py, 20 warmup + 200 timed single-image predict calls, seed-0 shuffled DUT image list in latency_images.json; arrays decoded before timing. Includes preprocessing/transfers, inference, NMS/postprocessing and Python overhead; excludes model load, disk/JPEG reads and warmup. All four original backend/precision combinations; ONNX exports static 1280-square while PyTorch retains original rectangular preprocessing. Images have mixed native resolutions. Latencies are desktop GPU measurements.
Complexity: checkpoint parameter count and fused deployment parameter count are separate; GFLOPs at square 1280 use fused NMS-head model and 2×THOP MACs. A profiler failure is NOT MEASURED.

## Commands and provenance
Exact executed commands, return codes, and durations: commands.jsonl. Worker implementations: runner.py. Original scripts are frozen under sources/ with hashes in provenance.json. Requested train arguments, checkpoint provenance, per-stage JSON and logs are retained. All writes are isolated here. Existing experiments/ and runs/ are not overwritten.
Launch: `nohup .venv/bin/python -u model_size_benchmark/runner.py run > model_size_benchmark/logs/orchestrator.log 2>&1 &`

## Hardware/software
```json
{
  "python": "3.12.3",
  "torch": "2.14.1+cu130",
  "cuda_build": "13.0",
  "packages": {
    "ultralytics": "8.4.215",
    "onnxruntime-gpu": "1.32.0",
    "onnx": "1.23.2",
    "numpy": "2.5.4",
    "ultralytics-thop": "2.2.5"
  },
  "gpu_query": "name, uuid, driver_version, memory.total [MiB], memory.used [MiB], utilization.gpu [%]\nNVIDIA GeForce RTX 3080, GPU-eb97c7d4-6687-7807-1761-24e211a20fd3, 580.173.02, 12288 MiB, 1 MiB, 0 %\n"
}
```

## Results
Full per-class and overall metrics, all latencies and timing fields: results.csv. NOT MEASURED means unavailable; absent classes in single-class test sets are also NOT MEASURED.
| Size | Status | Epochs | Best epoch | Train seconds (CSV) | Early stop | DUT AP50–95 | DUT recall | Det-Fly recall | PT FP16 mean ms | Deviation |
|---|---|---|---|---|---|---|---|---|---|---|
| n | pending | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| s | reused checkpoint; fresh speed/evaluation pending | 106 | 76 | 21676.4 | True | 0.7490165422829068 | 0.9670378619153674 | 0.285 | NOT MEASURED | none; existing s checkpoint reused |
| m | train | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | m batch 8 OOM; fresh retry at batch 4, nbs=64; currently training |

### Backend latency and model complexity
| Size | Backend | Mean ms | p95 ms | Timed images | Checkpoint parameters | Fused parameters | GFLOPs at 1280 |
|---|---|---|---|---|---|---|---|
| n | pytorch_fp32 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| n | pytorch_fp16 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| n | onnx_fp32 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| n | onnx_fp16 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| s | pytorch_fp32 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| s | pytorch_fp16 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| s | onnx_fp32 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| s | onnx_fp16 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| m | pytorch_fp32 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| m | pytorch_fp16 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| m | onnx_fp32 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| m | onnx_fp16 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

Training-stage total seconds in results.csv includes downloads/setup/final validation and failed attempts; csv_elapsed_seconds is the library training-loop duration. Existing s only has its historical loop duration.

### Executed worker commands
```bash
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage train --batch 8
```

## Honest interpretation
Single seed (0); no repeated-seed uncertainty estimate. Small differences may be run-to-run variation and must not be called real improvements. Dominance below uses strictly slower PyTorch FP16 mean latency AND lower DUT drone mAP50–95; other deployment backends can rank differently.
Comparison pending: where s sits, dominance across all three sizes, and the value of added size are NOT MEASURED until all three comparable measurements are available.

## Failures / pending actions
{
  "status": "running",
  "failures": []
}
