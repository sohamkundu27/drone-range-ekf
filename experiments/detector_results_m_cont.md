# Detector results — YOLO26m continued training

YOLO26m, 4 classes (drone, bird, airplane, helicopter), imgsz 1280. The retrained m (Det-Fly added; `runs/m_detfly_new`) stopped at its 7 h time cap after 49 epochs with its best epoch = its last (49). This run continues from that best.pt as a NEW run (`runs/m_detfly_cont`) for a 2.0 h time cap: 14 more epochs (63 in total), 2.01 h. Same data yaml and splits (12,008 train / 4,693 val), batch 4, seed 0, patience 30, default augmentation. Test sets and scripts identical to DETFLY_RETRAIN_RESULTS.md.

The best epoch was chosen on VALIDATION fitness only (Ultralytics best.pt, val mAP50-95); each test split was run once on that checkpoint.

## Side by side

DUT test = DUT Anti-UAV test (2,200 images, drones only). All-class and bird = DUT + COCO + Open Images test (all4_test). P/R there are Ultralytics' values at the F1-optimal confidence per class (as in detector_results.md).

| model | DUT drone P | DUT drone R | DUT drone mAP50 | DUT drone mAP50-95 | all-class mAP50 | all-class mAP50-95 | bird R | bird mAP50 | images with a false drone (of 1,534) |
|---|---|---|---|---|---|---|---|---|---|
| m, 49 epochs (retrained) | 0.991 | 0.975 | 0.988 | 0.755 | 0.865 | 0.657 | 0.630 | 0.682 | 2 |
| m, continued | 0.986 | 0.975 | 0.982 | 0.756 | 0.867 | 0.663 | 0.651 | 0.693 | 7 |
| s (retrained) | 0.985 | 0.973 | 0.980 | 0.743 | 0.861 | 0.662 | 0.623 | 0.674 | 8 |

Det-Fly (drone only; P/R at conf 0.25 with IoU >= 0.5; 95% block-bootstrap CI, 100-frame chunks, 1,000 resamples). 020 = 6,912 frames, a later part of the same site, never trained or validated on.

| model | Det-Fly set | P @0.25 [95% CI] | R @0.25 [95% CI] | mAP50 [95% CI] | mAP50-95 [95% CI] |
|---|---|---|---|---|---|
| m, 49 epochs (retrained) | 020 (all frames) | 0.939 [0.924, 0.953] | 0.729 [0.661, 0.785] | 0.775 [0.713, 0.826] | 0.391 [0.361, 0.421] |
| m, continued | 020 (all frames) | 0.934 [0.920, 0.948] | 0.731 [0.667, 0.786] | 0.775 [0.714, 0.828] | 0.394 [0.363, 0.424] |
| s (retrained) | 020 (all frames) | 0.924 [0.905, 0.942] | 0.711 [0.649, 0.768] | 0.752 [0.692, 0.804] | 0.358 [0.327, 0.389] |
| m, 49 epochs (retrained) | 010 test block | 0.800 [0.687, 0.911] | 0.762 [0.637, 0.885] | 0.687 [0.542, 0.845] | 0.339 [0.201, 0.512] |
| m, continued | 010 test block | 0.825 [0.721, 0.923] | 0.785 [0.671, 0.896] | 0.732 [0.596, 0.875] | 0.344 [0.211, 0.519] |
| s (retrained) | 010 test block | 0.790 [0.673, 0.905] | 0.762 [0.633, 0.889] | 0.677 [0.510, 0.842] | 0.343 [0.199, 0.519] |

## Accuracy (continued m), detector_results.md format

| test set | class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| DUT Anti-UAV test (in-domain) | all | 0.986 | 0.975 | 0.982 | 0.756 |
| | drone | 0.986 | 0.975 | 0.982 | 0.756 |
| DUT + COCO + Open Images test | all | 0.915 | 0.830 | 0.867 | 0.663 |
| | drone | 0.988 | 0.972 | 0.982 | 0.756 |
| | bird | 0.831 | 0.651 | 0.693 | 0.467 |
| | airplane | 0.898 | 0.793 | 0.860 | 0.689 |
| | helicopter | 0.944 | 0.903 | 0.935 | 0.741 |
| Det-Fly 020 (all 6,912 frames), conf 0.25 | drone | 0.934 | 0.731 | 0.775 | 0.394 |
| Det-Fly 010 test block, conf 0.25 | drone | 0.825 | 0.785 | 0.732 | 0.344 |

Latency: unchanged (same YOLO26m architecture); not re-measured.

## Training

| model | epochs run | best epoch (val) | best val mAP50-95 | wall-clock h | stop | final lr (regular / 3x head groups) | schedule factor at stop (floor = lrf 0.01) |
|---|---|---|---|---|---|---|---|
| m, 49 epochs (retrained) | 49 | 49 | 0.6367 | 7.01 | time cap / epoch limit | 0.000302 / 0.000906 (lr0 0.01) | 0.030 |
| m, continued | 14 (epochs 50-63 overall) | 14 (= epoch 63 overall) | 0.6400 | 2.01 | time cap (2.0 h) | 0.000161 / 0.000484 (lr0 0.002) | 0.081 |
| s (retrained) | 106 | 92 | 0.6409 | 7.01 | time cap / epoch limit | - | - |

Validation mAP50-95 per epoch of the continued run (start checkpoint: 0.6367):

| epoch (continued run) | overall epoch | val mAP50 | val mAP50-95 |
|---|---|---|---|
| 1 | 50 | 0.8570 | 0.6175 |
| 2 | 51 | 0.8585 | 0.6077 |
| 3 | 52 | 0.8647 | 0.6222 |
| 4 | 53 | 0.8504 | 0.6089 |
| 5 | 54 | 0.8459 | 0.5998 |
| 6 | 55 | 0.8524 | 0.6091 |
| 7 | 56 | 0.8564 | 0.6186 |
| 8 | 57 | 0.8584 | 0.6264 |
| 9 | 58 | 0.8644 | 0.6303 |
| 10 | 59 | 0.8579 | 0.6292 |
| 11 | 60 | 0.8567 | 0.6289 |
| 12 | 61 | 0.8625 | 0.6321 |
| 13 | 62 | 0.8619 | 0.6385 |
| 14 | 63 | 0.8622 | 0.6400 |

## Recipe and checks

* Start: `runs/m_detfly_new/weights/best.pt` (SHA-256 `68b679a66185742257e2b6912b4d043b5daedb98a1219e3ce9f0ae31e81df47e`), epoch 49 = best and last epoch of that run (max val mAP50-95 in its results.csv).
* Applied (from the run's args.yaml and log): epochs=150, time=2.0, optimizer=MuSGD, lr0=0.002, lrf=0.01, momentum=0.937, warmup_epochs=1.0, warmup_bias_lr=0.0, warmup_momentum=0.8, batch=4, nbs=64, imgsz=1280, seed=0, patience=30, close_mosaic=10, mosaic=1.0, workers=12.
* Log: `optimizer: MuSGD(lr=0.002, momentum=0.937) with parameter groups 124 weight(decay=0.0), 136 weight(decay=0.0005), 136 bias(decay=0.0)`. With `optimizer=auto` (original run) Ultralytics ignores lr0, uses 0.01 and also sets warmup_bias_lr = 0; the continued run sets MuSGD explicitly with lr0 = 0.002, and warmup_bias_lr = 0 to keep that behaviour (the default 0.1 would be 50x lr0 for the bias groups during warmup). Momentum 0.937 is the default and what the original run used after warmup.
* Final learning rate: with `time` set, Ultralytics re-fits the epoch count E and the linear LR schedule to the time budget after every epoch. The linear schedule gives the last epoch (index E-1) the factor lrf + (1 - lrf)/E, so the LR never reaches the lrf floor inside training: the floor is where the schedule would be after one more epoch. Here E = 14: logged factor 0.081 = 0.01 + 0.99/14 = 0.081 (floor 0.01). Original run: factor 0.030, consistent with E = 50 (0.01 + 0.99/50 = 0.030), i.e. the same effect, not a run cut short before its schedule ended.
* close_mosaic = 10 (default) counts back from the re-fitted epoch count, so mosaic augmentation was switched off from epoch 5 of 14 (re-fitted) of this run.
* best.pt = epoch 14 of the continued run, the max-val-fitness epoch (verified against the checkpoint).
* best.pt SHA-256 `e3bbe45671958ec7be379459d5d54f584d7bd2ed9376419e389aee5cb0234c04` (weights not committed).

## Plots

`detfly_retrain/m_cont/`: results.png (training curves), BoxPR_curve.png, confusion_matrix.png, confusion_matrix_normalized.png (all from the continued run's validation set).
