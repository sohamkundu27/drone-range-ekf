# Detector results — NEW YOLO26s (Det-Fly frames added to training)

Same layout as `presentation_assets/detector_results.md`; the OLD s value is given next to each NEW value. P/R in the accuracy table are Ultralytics' values at the F1-optimal confidence per class (as before). Det-Fly rows use P/R/F1 at conf 0.25 with 95 % block-bootstrap CIs (see DETFLY_RETRAIN_RESULTS.md).

## Accuracy

| test set | class | P new (old) | R new (old) | mAP50 new (old) | mAP50-95 new (old) |
|---|---|---|---|---|---|
| DUT Anti-UAV test (in-domain) | all | 0.985 (0.987) | 0.973 (0.967) | 0.980 (0.979) | 0.743 (0.749) |
| | drone | 0.985 (0.987) | 0.973 (0.967) | 0.980 (0.979) | 0.743 (0.749) |
| DUT + COCO + Open Images test | all | 0.915 (0.916) | 0.826 (0.807) | 0.861 (0.861) | 0.662 (0.660) |
| | drone | 0.982 (0.987) | 0.972 (0.967) | 0.980 (0.979) | 0.743 (0.749) |
| | bird | 0.822 (0.816) | 0.623 (0.591) | 0.674 (0.664) | 0.464 (0.449) |
| | airplane | 0.920 (0.914) | 0.786 (0.767) | 0.849 (0.856) | 0.682 (0.684) |
| | helicopter | 0.935 (0.949) | 0.921 (0.903) | 0.941 (0.944) | 0.757 (0.759) |

## Det-Fly (drone only, conf 0.25; 020 = held out, same site as the 010 training frames)

| set | model | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Det-Fly 020 (6,912 frames) | new s | 0.924 [0.905, 0.942] | 0.711 [0.649, 0.768] | 0.804 [0.759, 0.844] | 0.752 [0.692, 0.804] | 0.358 [0.327, 0.389] |
| Det-Fly 020 (6,912 frames) | old s | 0.741 [0.682, 0.798] | 0.444 [0.375, 0.506] | 0.555 [0.489, 0.613] | 0.497 [0.417, 0.572] | 0.271 [0.224, 0.322] |
| Det-Fly 010 test block (1,590) | new s | 0.790 [0.673, 0.905] | 0.762 [0.633, 0.889] | 0.776 [0.656, 0.891] | 0.677 [0.510, 0.842] | 0.343 [0.199, 0.519] |
| Det-Fly 010 test block (1,590) | old s | 0.854 [0.780, 0.924] | 0.667 [0.524, 0.796] | 0.749 [0.643, 0.842] | 0.673 [0.527, 0.806] | 0.278 [0.212, 0.352] |
| Det-Fly preliminary 1,000 (old slide) | old s | 0.770 | 0.292 | 0.423 | 0.304 | 0.119 |

## False 'drone' detections on images with no drone (conf 0.25 / 0.5)

| images | conf | images | % with a false drone new (old) | false drones new (old) | on object new (old) |
|---|---|---|---|---|---|
| all/airplane | 0.25 | 597 | 0.8% (0.5%) | 5 (3) | 2/5 (0/3) |
| all/airplane | 0.5 | 597 | 0.0% (0.0%) | 0 (0) | 0/0 (0/0) |
| all/any | 0.25 | 1534 | 0.5% (0.3%) | 9 (5) | 6/9 (2/5) |
| all/any | 0.5 | 1534 | 0.1% (0.1%) | 1 (1) | 1/1 (1/1) |
| all/bird | 0.25 | 612 | 0.0% (0.0%) | 0 (0) | 0/0 (0/0) |
| all/bird | 0.5 | 612 | 0.0% (0.0%) | 0 (0) | 0/0 (0/0) |
| all/helicopter | 0.25 | 358 | 1.1% (0.6%) | 5 (2) | 5/5 (2/2) |
| all/helicopter | 0.5 | 358 | 0.3% (0.3%) | 1 (1) | 1/1 (1/1) |
| coco/airplane | 0.25 | 97 | 1.0% (0.0%) | 1 (0) | 1/1 (0/0) |
| coco/airplane | 0.5 | 97 | 0.0% (0.0%) | 0 (0) | 0/0 (0/0) |
| coco/bird | 0.25 | 112 | 0.0% (0.0%) | 0 (0) | 0/0 (0/0) |
| coco/bird | 0.5 | 112 | 0.0% (0.0%) | 0 (0) | 0/0 (0/0) |
| openimages/airplane | 0.25 | 500 | 0.8% (0.6%) | 4 (3) | 1/4 (0/3) |
| openimages/airplane | 0.5 | 500 | 0.0% (0.0%) | 0 (0) | 0/0 (0/0) |
| openimages/bird | 0.25 | 500 | 0.0% (0.0%) | 0 (0) | 0/0 (0/0) |
| openimages/bird | 0.5 | 500 | 0.0% (0.0%) | 0 (0) | 0/0 (0/0) |
| openimages/helicopter | 0.25 | 358 | 1.1% (0.6%) | 5 (2) | 5/5 (2/2) |
| openimages/helicopter | 0.5 | 358 | 0.3% (0.3%) | 1 (1) | 1/1 (1/1) |

## Latency (DESKTOP GPU: RTX 3080, batch 1, 1920x1080 input, imgsz 1280) — REUSED

Not re-measured: the new s has the identical YOLO26s architecture and input size, so the old measurement (`experiments/latency.json`) applies. Not representative of embedded hardware.

| backend | mean ms | p50 ms | p95 ms | FPS | inference-only ms |
|---|---|---|---|---|---|
| pytorch_fp32 | 7.0 | 7.3 | 7.4 | 142.0 | 5.2 |
| pytorch_fp16 | 5.4 | 5.6 | 5.7 | 186.6 | 3.5 |
| onnx_fp32 | 11.1 | 11.4 | 11.5 | 90.0 | 9.0 |
| onnx_fp16 | 8.0 | 8.4 | 8.4 | 124.3 | 5.9 |

## Box errors on matched true positives (DUT test) -> simulator noise (NEW s; old in brackets)

2202 of 2245 test drones matched at conf >= 0.25 (IoU >= 0.3) [old: 2196 of 2245]. σ = robust (1.4826·MAD); std includes outliers; tail = fraction beyond 3 robust σ.

| true width px | n matched | recall | centre σ px (std) | log-width σ (std) | log-width bias | tail centre / width |
|---|---|---|---|---|---|---|
| 0-8 | 0 | - | too few | | | |
| 8-12 | 3 | 1.00 | too few | | | |
| 12-16 | 12 | 1.00 | too few | | | |
| 16-24 | 126 [126] | 0.96 | 0.92 (1.04) [0.88] | 0.092 (0.106) [0.094] | +0.086 [+0.073] | 0.4% / 2.4% |
| 24-32 | 232 [230] | 0.99 | 0.84 (1.06) [0.89] | 0.071 (0.091) [0.073] | +0.034 [+0.034] | 3.2% / 4.7% |
| 32-48 | 541 [543] | 0.99 | 0.99 (1.27) [1.06] | 0.051 (0.076) [0.047] | +0.007 [+0.007] | 2.9% / 6.3% |
| 48-64 | 228 [228] | 0.97 | 1.02 (1.52) [1.02] | 0.041 (0.062) [0.047] | +0.008 [+0.004] | 4.6% / 8.3% |
| 64-96 | 195 [193] | 0.98 | 1.07 (2.49) [1.18] | 0.034 (0.059) [0.045] | +0.004 [+0.009] | 6.9% / 6.2% |
| 96-160 | 238 [236] | 0.97 | 1.51 (3.77) [1.42] | 0.030 (0.071) [0.030] | +0.009 [+0.012] | 8.4% / 12.2% |
| 160-inf | 627 [625] | 0.99 | 2.81 (12.95) [2.90] | 0.030 (0.072) [0.029] | +0.002 [+0.000] | 15.8% / 14.4% |
