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
| n | complete | 174 | 144 | 20619.6 | False | 0.7317421427348211 | 0.9623200684934009 | 0.2725671632115101 | 4.499312411062419 | none |
| s | complete | 106 | 76 | 21676.4 | True | 0.7490165422829068 | 0.9670378619153674 | 0.285 | 5.337705845013261 | none; existing s checkpoint reused |
| m | complete | 57 | 56 | 25241.2 | False | 0.7555688791276456 | 0.9705416759930563 | 0.314 | 8.7681251950562 | m batch 8 OOM; fresh retry at batch 4, nbs=64 |

### Backend latency and model complexity
| Size | Backend | Mean ms | p95 ms | Timed images | Checkpoint parameters | Fused parameters | GFLOPs at 1280 |
|---|---|---|---|---|---|---|---|
| n | pytorch_fp32 | 4.668648545630276 | 5.07103456184268 | 200 | 2535344 | 2390464 | 22.9872128 |
| n | pytorch_fp16 | 4.499312411062419 | 4.876226605847478 | 200 | 2535344 | 2390464 | 22.9872128 |
| n | onnx_fp32 | 6.490783481858671 | 6.893785111606121 | 200 | 2535344 | 2390464 | 22.9872128 |
| n | onnx_fp16 | 5.3991014417260885 | 6.04536784812808 | 200 | 2535344 | 2390464 | 22.9872128 |
| s | pytorch_fp32 | 7.033395394682884 | 7.768256589770317 | 200 | 9950960 | 9466728 | 86.028288 |
| s | pytorch_fp16 | 5.337705845013261 | 5.954185826703906 | 200 | 9950960 | 9466728 | 86.028288 |
| s | onnx_fp32 | 11.09811784233898 | 11.493110191076994 | 200 | 9950960 | 9466728 | 86.028288 |
| s | onnx_fp16 | 8.002369832247496 | 8.397756144404411 | 200 | 9950960 | 9466728 | 86.028288 |
| m | pytorch_fp32 | 13.562640775926411 | 13.93322367221117 | 200 | 21779056 | 20352536 | 275.3445888 |
| m | pytorch_fp16 | 8.7681251950562 | 9.117262298241258 | 200 | 21779056 | 20352536 | 275.3445888 |
| m | onnx_fp32 | 22.04219736624509 | 22.47879500500858 | 200 | 21779056 | 20352536 | 275.3445888 |
| m | onnx_fp16 | 14.221938918344676 | 14.651026856154203 | 200 | 21779056 | 20352536 | 275.3445888 |

Training-stage total seconds in results.csv includes downloads/setup/final validation and failed attempts; csv_elapsed_seconds is the library training-loop duration. Existing s only has its historical loop duration.

### Executed worker commands
```bash
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage train --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage train --batch 4
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage eval_dut --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage eval_combined --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage eval_detfly --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage false_positives --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage complexity --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage export_fp32 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage export_fp16 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage latency_pytorch_fp32 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage latency_pytorch_fp16 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage latency_onnx_fp32 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size m --stage latency_onnx_fp16 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage train --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage eval_dut --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage eval_combined --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage eval_detfly --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage false_positives --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage complexity --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage export_fp32 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage export_fp16 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage latency_pytorch_fp32 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage latency_pytorch_fp16 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage latency_onnx_fp32 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size n --stage latency_onnx_fp16 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage eval_dut --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage eval_combined --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage eval_detfly --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage false_positives --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage complexity --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage export_fp32 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage export_fp16 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage latency_pytorch_fp32 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage latency_pytorch_fp16 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage latency_onnx_fp32 --batch 8
/home/soham/github/drone-range-ekf/.venv/bin/python -u /home/soham/github/drone-range-ekf/model_size_benchmark/runner.py worker --size s --stage latency_onnx_fp16 --batch 8
```

## Honest interpretation
Single seed (0); no repeated-seed uncertainty estimate. Small differences may be run-to-run variation and must not be called real improvements. Dominance below uses strictly slower PyTorch FP16 mean latency AND lower DUT drone mAP50–95; other deployment backends can rank differently.
YOLO26n: observed dominators = none.
YOLO26s: observed dominators = none.
YOLO26m: observed dominators = none.
n minus s: DUT AP50–95 delta -0.017274; PT FP16 latency delta -0.8384 ms.
m minus s: DUT AP50–95 delta +0.006552; PT FP16 latency delta +3.4304 ms.
These are observed differences only. Whether an accuracy gain is meaningful for deployment requires a specified minimum benefit and repeated-seed measurements; this benchmark cannot establish statistical significance. Any m batch-4 deviation also limits attribution solely to model size.

## Failures / pending actions
{
  "status": "complete",
  "failures": []
}
