# Detector results — NEW YOLO26s (Det-Fly frames added to training)

Same layout as `presentation_assets/detector_results.md`; the OLD s value is given next to each NEW value. P/R in the accuracy table are Ultralytics' values at the F1-optimal confidence per class (as before). Det-Fly rows use P/R/F1 at conf 0.25 with 95 % block-bootstrap CIs (see DETFLY_RETRAIN_RESULTS.md).

## Accuracy

| test set | class | P new (old) | R new (old) | mAP50 new (old) | mAP50-95 new (old) |
|---|---|---|---|---|---|
| DUT Anti-UAV test (in-domain) | all | n/a (n/a) | n/a (n/a) | n/a (n/a) | n/a (n/a) |
| DUT + COCO + Open Images test | all | n/a (0.916) | n/a (0.807) | n/a (0.861) | n/a (0.660) |
| | drone | n/a (0.987) | n/a (0.967) | n/a (0.979) | n/a (0.749) |
| | bird | n/a (0.816) | n/a (0.591) | n/a (0.664) | n/a (0.449) |
| | airplane | n/a (0.914) | n/a (0.767) | n/a (0.856) | n/a (0.684) |
| | helicopter | n/a (0.949) | n/a (0.903) | n/a (0.944) | n/a (0.759) |

## Det-Fly (drone only, conf 0.25; 020 = held out, same site as the 010 training frames)

| set | model | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Det-Fly 020 (6,912 frames) | new s | n/a | n/a | n/a | n/a | n/a |
| Det-Fly 020 (6,912 frames) | old s | 0.741 [0.682, 0.798] | 0.444 [0.375, 0.506] | 0.555 [0.489, 0.613] | 0.497 [0.417, 0.572] | 0.271 [0.224, 0.322] |
| Det-Fly 010 test block (1,590) | new s | n/a | n/a | n/a | n/a | n/a |
| Det-Fly 010 test block (1,590) | old s | 0.854 [0.780, 0.924] | 0.667 [0.524, 0.796] | 0.749 [0.643, 0.842] | 0.673 [0.527, 0.806] | 0.278 [0.212, 0.352] |
| Det-Fly preliminary 1,000 (old slide) | old s | 0.770 | 0.292 | 0.423 | 0.304 | 0.119 |

## False 'drone' detections on images with no drone (conf 0.25 / 0.5)

| images | conf | images | % with a false drone new (old) | false drones new (old) | on object new (old) |
|---|---|---|---|---|---|

## Latency (DESKTOP GPU: RTX 3080, batch 1, 1920x1080 input, imgsz 1280) — REUSED

Not re-measured: the new s has the identical YOLO26s architecture and input size, so the old measurement (`experiments/latency.json`) applies. Not representative of embedded hardware.

| backend | mean ms | p50 ms | p95 ms | FPS | inference-only ms |
|---|---|---|---|---|---|
| pytorch_fp32 | 7.0 | 7.3 | 7.4 | 142.0 | 5.2 |
| pytorch_fp16 | 5.4 | 5.6 | 5.7 | 186.6 | 3.5 |
| onnx_fp32 | 11.1 | 11.4 | 11.5 | 90.0 | 9.0 |
| onnx_fp16 | 8.0 | 8.4 | 8.4 | 124.3 | 5.9 |
