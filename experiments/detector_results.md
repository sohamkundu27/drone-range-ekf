# Detector results

YOLO26s, 4 classes (drone, bird, airplane, helicopter), imgsz 1280, trained on DUT Anti-UAV + COCO + Open Images (rebalanced, drones ~51% of training images). P/R are Ultralytics' values at the F1-optimal confidence per class.

## Accuracy

| test set | class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| DUT Anti-UAV test (in-domain) | all | 0.987 | 0.967 | 0.979 | 0.749 |
| | drone | 0.987 | 0.967 | 0.979 | 0.749 |
| DUT + COCO + Open Images test | all | 0.916 | 0.807 | 0.861 | 0.660 |
| | drone | 0.987 | 0.967 | 0.979 | 0.749 |
| | bird | 0.816 | 0.591 | 0.664 | 0.449 |
| | airplane | 0.914 | 0.767 | 0.856 | 0.684 |
| | helicopter | 0.949 | 0.903 | 0.944 | 0.759 |
| Det-Fly **PRELIMINARY** (partial: 1,000 of 13,271 images, sequence 010 only) | all | 0.842 | 0.285 | 0.304 | 0.119 |
| | drone | 0.842 | 0.285 | 0.304 | 0.119 |

## False 'drone' detections on images with no drone

Held-out COCO and Open Images test images containing only birds / airplanes / helicopters. 'on object' = the false drone box overlaps a labelled bird/aircraft (IoU >= 0.1).

| images | conf | images | % images with a false drone | false drones / image | on object |
|---|---|---|---|---|---|
| all/airplane | 0.25 | 597 | 0.5% | 0.005 | 0/3 |
| all/airplane | 0.5 | 597 | 0.0% | 0.000 | 0/0 |
| all/any | 0.25 | 1534 | 0.3% | 0.003 | 2/5 |
| all/any | 0.5 | 1534 | 0.1% | 0.001 | 1/1 |
| all/bird | 0.25 | 612 | 0.0% | 0.000 | 0/0 |
| all/bird | 0.5 | 612 | 0.0% | 0.000 | 0/0 |
| all/helicopter | 0.25 | 358 | 0.6% | 0.006 | 2/2 |
| all/helicopter | 0.5 | 358 | 0.3% | 0.003 | 1/1 |
| coco/airplane | 0.25 | 97 | 0.0% | 0.000 | 0/0 |
| coco/airplane | 0.5 | 97 | 0.0% | 0.000 | 0/0 |
| coco/bird | 0.25 | 112 | 0.0% | 0.000 | 0/0 |
| coco/bird | 0.5 | 112 | 0.0% | 0.000 | 0/0 |
| openimages/airplane | 0.25 | 500 | 0.6% | 0.006 | 0/3 |
| openimages/airplane | 0.5 | 500 | 0.0% | 0.000 | 0/0 |
| openimages/bird | 0.25 | 500 | 0.0% | 0.000 | 0/0 |
| openimages/bird | 0.5 | 500 | 0.0% | 0.000 | 0/0 |
| openimages/helicopter | 0.25 | 358 | 0.6% | 0.006 | 2/2 |
| openimages/helicopter | 0.5 | 358 | 0.3% | 0.003 | 1/1 |

## Latency (DESKTOP GPU: RTX 3080, batch 1, 1920x1080 input, imgsz 1280)

End-to-end `predict()` time including pre/post-processing. Not representative of embedded hardware.

| backend | mean ms | p50 ms | p95 ms | FPS | inference-only ms |
|---|---|---|---|---|---|
| pytorch_fp32 | 7.0 | 7.3 | 7.4 | 142.0 | 5.2 |
| pytorch_fp16 | 5.4 | 5.6 | 5.7 | 186.6 | 3.5 |
| onnx_fp32 | 11.1 | 11.4 | 11.5 | 90.0 | 9.0 |
| onnx_fp16 | 8.0 | 8.4 | 8.4 | 124.3 | 5.9 |

## Box errors on matched true positives (DUT test) -> simulator noise

2196 of 2245 test drones matched at conf >= 0.25 (IoU >= 0.3). σ = robust (1.4826·MAD); std includes outliers; tail = fraction beyond 3 robust σ.

| true width px | n matched | recall | centre σ px (std) | log-width σ (std) | log-width bias | tail centre / width |
|---|---|---|---|---|---|---|
| 0-8 | 0 | - | too few | | | |
| 8-12 | 3 | 1.00 | too few | | | |
| 12-16 | 12 | 1.00 | too few | | | |
| 16-24 | 126 | 0.96 | 0.88 (1.03) | 0.094 (0.119) | +0.073 | 0.4% / 4.8% |
| 24-32 | 230 | 0.98 | 0.89 (1.02) | 0.073 (0.097) | +0.034 | 1.1% / 3.9% |
| 32-48 | 543 | 0.99 | 1.06 (1.29) | 0.047 (0.081) | +0.007 | 2.2% / 7.7% |
| 48-64 | 228 | 0.97 | 1.02 (1.55) | 0.047 (0.066) | +0.004 | 5.0% / 6.6% |
| 64-96 | 193 | 0.97 | 1.18 (2.53) | 0.045 (0.062) | +0.009 | 4.9% / 4.1% |
| 96-160 | 236 | 0.96 | 1.42 (3.33) | 0.030 (0.059) | +0.012 | 6.4% / 5.5% |
| 160-inf | 625 | 0.99 | 2.90 (9.82) | 0.029 (0.067) | +0.000 | 13.7% / 13.1% |
