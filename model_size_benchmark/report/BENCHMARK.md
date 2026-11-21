# YOLO26 model-size benchmark

Single run per size, seed 0. Small differences may be within run-to-run noise. **l and x were NOT run.**

The recipe gives each size the same 7 h training budget, so epoch counts differ. n and s stopped after 30 epochs without improvement. m reached the time cap after a batch-8 CUDA OOM and a fresh batch-4 attempt with nbs=64. s uses the existing checkpoint. Best checkpoints were selected on combined validation mAP50–95. Wall h is the training loop time from the run log.

| Size | Batch | Epochs | Best epoch | Wall h | Stop | Final val AP50 | Final val AP50–95 |
|---|---|---|---|---|---|---|---|
| n | 8 | 174 | 144 | 5.728 | Patience 30 | 0.85497 | 0.63017 |
| s | 8 | 106 | 76 | 6.021 | Patience 30 | 0.86047 | 0.6442 |
| m | 4 | 57 | 56 | 7.012 | Time cap | 0.85856 | 0.64317 |

| Size | Training run | Training log |
|---|---|---|
| n | model_size_benchmark/train/n_batch8 | model_size_benchmark/logs/n_train_batch8.log |
| s | runs/detect/drone4_bal | runs/logs/train_drone4_bal.log |
| m | model_size_benchmark/train/m_batch4 | model_size_benchmark/logs/m_train_batch4.log |

## Held-out results

Existing successful test evaluations were verified and reused. DUT, combined DUT + COCO + Open Images, and the same 1,000-image Det-Fly subset use best.pt, split=test, imgsz=1280, batch=8, conf=0.001, iou=0.7 and CUDA device 0. P and R are at the Ultralytics mean-F1 operating point. False drone boxes at 0.25 come from detector/false_positives.py across the 1,534 COCO + Open Images test images. AP denotes class AP or overall mean AP as indicated. NOT MEASURED includes classes absent from a split.

| Size | DUT AP50 | DUT R | DUT AP50–95 | Combined AP50 | Combined AP50–95 | Det-Fly R | Det-Fly AP50 | False drone boxes | PT FP16 mean ms |
|---|---|---|---|---|---|---|---|---|---|
| n | 0.9759 | 0.9623 | 0.7317 | 0.8601 | 0.6458 | 0.2726 | 0.3051 | 10 | 4.81 |
| s | 0.9792 | 0.9670 | 0.7490 | 0.8608 | 0.6602 | 0.2850 | 0.3036 | 5 | 5.43 |
| m | 0.9816 | 0.9705 | 0.7556 | 0.8598 | 0.6523 | 0.3140 | 0.3427 | 3 | 8.83 |

| Split | Class | Size | Precision | Recall | AP50 | AP50–95 |
|---|---|---|---|---|---|---|
| dut | overall | n | 0.9778 | 0.9623 | 0.9759 | 0.7317 |
| dut | overall | s | 0.9871 | 0.9670 | 0.9792 | 0.7490 |
| dut | overall | m | 0.9909 | 0.9705 | 0.9816 | 0.7556 |
| dut | drone | n | 0.9778 | 0.9623 | 0.9759 | 0.7317 |
| dut | drone | s | 0.9871 | 0.9670 | 0.9792 | 0.7490 |
| dut | drone | m | 0.9909 | 0.9705 | 0.9816 | 0.7556 |
| dut | bird | n | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| dut | bird | s | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| dut | bird | m | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| dut | airplane | n | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| dut | airplane | s | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| dut | airplane | m | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| dut | helicopter | n | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| dut | helicopter | s | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| dut | helicopter | m | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| combined | overall | n | 0.9118 | 0.8064 | 0.8601 | 0.6458 |
| combined | overall | s | 0.9164 | 0.8069 | 0.8608 | 0.6602 |
| combined | overall | m | 0.9172 | 0.8149 | 0.8598 | 0.6523 |
| combined | drone | n | 0.9795 | 0.9555 | 0.9754 | 0.7317 |
| combined | drone | s | 0.9865 | 0.9670 | 0.9790 | 0.7490 |
| combined | drone | m | 0.9892 | 0.9706 | 0.9815 | 0.7555 |
| combined | bird | n | 0.8317 | 0.5905 | 0.6702 | 0.4501 |
| combined | bird | s | 0.8161 | 0.5905 | 0.6641 | 0.4494 |
| combined | bird | m | 0.8266 | 0.6053 | 0.6708 | 0.4480 |
| combined | airplane | n | 0.8917 | 0.7741 | 0.8500 | 0.6671 |
| combined | airplane | s | 0.9138 | 0.7673 | 0.8562 | 0.6836 |
| combined | airplane | m | 0.9160 | 0.7859 | 0.8539 | 0.6745 |
| combined | helicopter | n | 0.9444 | 0.9054 | 0.9448 | 0.7343 |
| combined | helicopter | s | 0.9493 | 0.9028 | 0.9440 | 0.7588 |
| combined | helicopter | m | 0.9372 | 0.8977 | 0.9328 | 0.7311 |
| detfly | overall | n | 0.7375 | 0.2726 | 0.3051 | 0.1133 |
| detfly | overall | s | 0.8420 | 0.2850 | 0.3036 | 0.1192 |
| detfly | overall | m | 0.8396 | 0.3140 | 0.3427 | 0.1350 |
| detfly | drone | n | 0.7375 | 0.2726 | 0.3051 | 0.1133 |
| detfly | drone | s | 0.8420 | 0.2850 | 0.3036 | 0.1192 |
| detfly | drone | m | 0.8396 | 0.3140 | 0.3427 | 0.1350 |
| detfly | bird | n | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| detfly | bird | s | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| detfly | bird | m | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| detfly | airplane | n | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| detfly | airplane | s | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| detfly | airplane | m | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| detfly | helicopter | n | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| detfly | helicopter | s | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| detfly | helicopter | m | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

## Speed and complexity

RTX 3080 desktop GPU (12 GiB), driver 580.173.02, Python 3.12.3, PyTorch 2.14.1+cu130, CUDA 13.0, ultralytics 8.4.215, onnxruntime-gpu 1.32.0, onnx 1.23.2, numpy 2.5.4, ultralytics-thop 2.2.5.

Latency uses detector/latency.py with batch 1 and imgsz=1280. All timed runs use the same 220 images in latency_images.json and its seed-0 order, 20 warmup calls and 200 timed calls. Timing covers predict() including preprocessing and postprocessing. Images are decoded before timing. ONNX uses the existing static square export and CUDA execution provider. PyTorch uses the script’s native rectangular preprocessing. GPU occupancy was checked before each timing and compute PIDs were polled while each process ran.

| Size | Backend | Mean ms | p50 ms | p95 ms | FPS |
|---|---|---|---|---|---|
| n | pytorch_fp32 | 4.81 | 5.03 | 5.39 | 208.01 |
| n | pytorch_fp16 | 4.81 | 5.04 | 5.40 | 207.71 |
| n | onnx_fp32 | 6.60 | 6.91 | 7.14 | 151.48 |
| n | onnx_fp16 | 5.41 | 5.69 | 6.01 | 184.92 |
| s | pytorch_fp32 | 7.13 | 7.36 | 7.64 | 140.33 |
| s | pytorch_fp16 | 5.43 | 5.64 | 6.09 | 184.23 |
| s | onnx_fp32 | 11.17 | 11.45 | 11.81 | 89.51 |
| s | onnx_fp16 | 8.07 | 8.34 | 8.63 | 123.89 |
| m | pytorch_fp32 | 13.60 | 13.85 | 14.08 | 73.55 |
| m | pytorch_fp16 | 8.83 | 9.06 | 9.33 | 113.29 |
| m | onnx_fp32 | 22.04 | 22.31 | 22.65 | 45.38 |
| m | onnx_fp16 | 14.23 | 14.53 | 14.82 | 70.29 |

| Size | Checkpoint parameters | Fused parameters | GFLOPs @ 1280 |
|---|---|---|---|
| n | 2535344 | 2390464 | 22.99 |
| s | 9950960 | 9466728 | 86.03 |
| m | 21779056 | 20352536 | 275.34 |

GFLOPs use Ultralytics get_flops (2×THOP MACs) for a fused NMS-head model at 1280×1280.

## Commands and artifacts

Original training, checkpoint evaluation, false-positive, export and complexity commands and exit codes are in commands.jsonl. Test stages used the following entry points with SIZE set to n, s, or m.

```bash
for stage in eval_dut eval_combined eval_detfly false_positives complexity; do
  .venv/bin/python -u model_size_benchmark/runner.py worker --size "$SIZE" --stage "$stage" --batch 8
done
```

Those commands describe the recorded run. The timing pass uses the exact arguments written in latency_logs/, invoking the original detector/latency.py main with its output root set to this report. Raw new observations are in experiments/latency.json. The export corrects the n early-stop flag using the training log.

```bash
.venv/bin/python -I detector/confusion_matrix_conf.py --weights "$ABSOLUTE_BEST_PT" --conf 0.25 --iou 0.5 --out "$TEMP_DIR"
```

Confusion matrices use the combined validation split, with matching IoU 0.5. Raw counts are in confusion_matrix_{n,s,m}.json. Training curves and PR plots are copies of the per-run Ultralytics plots.

## Conclusion

s sits between n and m in PyTorch FP16 latency and DUT AP50–95. No size is strictly dominated on these two axes. m adds 3.40 ms for +0.66 percentage points of DUT AP50–95 and +2.90 points of Det-Fly recall. Its combined AP50–95 is 0.79 points lower. The DUT gain is small for the latency cost. The Det-Fly gain could matter if that domain is the priority. n saves 0.61 ms and gives up 1.73 points of DUT AP50–95 and 1.24 points of Det-Fly recall. n produced 10 false drone boxes versus 5 for s and 3 for m on the negative set. These are observed tradeoffs from one run per size. The training batch difference also limits what can be attributed to model size alone.
