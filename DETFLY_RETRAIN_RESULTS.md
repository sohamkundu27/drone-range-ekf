# Det-Fly retrain — OLD vs NEW YOLO26 n / s / m

> **Read first (choices made without asking, per instructions):**
> * **020 is the same site as 010** (same hill, buildings, sports courts and dusk light in sampled frames; both 3840×2160). It is treated as a *later part of the same recording*, not an independent cross-site test. Det-Fly as a whole is still a different dataset/camera from the DUT training data.
> * 13,270 Det-Fly frames: 010 = 6,358, 020 = 6,912. Frame 020/0207614 is missing from the source release (its XML exists, the JPEG does not); 020_0201689 has no UAV box (kept as a background frame), so 020 has 6,911 drone boxes.
> * **020 drones are much smaller** than the Det-Fly training frames: median √(w·h) after letterboxing to 1280 is 16.6 px on 020 vs 34.6 px on the 010 train frames and 28.7 px on the 010 test block (`presentation_assets_v2/dataset_stats.md`), so 020 is also a harder, smaller-object test.
> * The XML `<folder>` field marks Det-Fly's own authors' split (most 010 frames 'train', later ones 'test'); it is not used here — the split is the time-block split in `detfly_split.json`.
> * Split (`detfly_split.json`): 010 cut at 60 %/75 % of its frames, 200-frame buffers dropped from the end of the train and val blocks, train and val subsampled with stride 2 → 1,808 train / 377 val frames; 010 test = last 25 % (1,590 frames, all kept); 020 = 6,912 test frames, never trained or validated on.
> * Protocol: Ultralytics val, imgsz 1280, batch 8, conf 0.001, NMS IoU 0.7 (as detector/eval.py), drone class only. P/R/F1 at confidence ≥ 0.25 with IoU ≥ 0.5 matching (recomputed from the validator's own matches). 95 % CI = block bootstrap, 100-frame time-ordered chunks, 1,000 resamples, percentile.
> * NEW = same recipe as the old run (saved `args.yaml` of the old s run, as used by the old n/m benchmark), with 1,808 Det-Fly train frames added to training and 377 Det-Fly val frames added to validation (early stopping / best.pt selection therefore also sees Det-Fly val; the old runs did not).
> * Decision rule (stated, not acted on): *new is clearly better if Det-Fly 020 recall is up beyond the CI AND DUT mAP50 and false-drone count are not clearly worse.* Applied here as: new 020 recall@0.25 above the old CI upper bound and new CI lower bound above the old point; DUT-test drone mAP50 drop < 0.01; false-drone images increase ≤ 5 (of 1,534).

## Plain-language summary

* **s**: new model not finished yet.
* **n**: new model not finished yet.
* **m**: new model not finished yet.
* No new model finished yet; old-model numbers are in DETFLY_FULL_RESULTS.md.

## (c) Det-Fly drone, conf 0.25 (020 = later part of the same site; 010 test = held-out block)

| model | Det-Fly set | frames | P @0.25 [95% CI] | R @0.25 [95% CI] | F1 @0.25 [95% CI] | mAP50 [95% CI] | mAP50-95 [95% CI] |
|---|---|---|---|---|---|---|---|
| old n | 020 (all frames) | 6912 | 0.688 [0.606, 0.762] | 0.460 [0.392, 0.521] | 0.551 [0.486, 0.612] | 0.498 [0.416, 0.575] | 0.255 [0.211, 0.303] |
| old n | 010 test block | 1590 | 0.797 [0.708, 0.871] | 0.650 [0.514, 0.782] | 0.716 [0.601, 0.820] | 0.666 [0.519, 0.803] | 0.273 [0.208, 0.344] |
| **new n** | 020 (all frames) | | pending | pending | pending | pending | pending |
| **new n** | 010 test block | | pending | pending | pending | pending | pending |
| old s | 020 (all frames) | 6912 | 0.741 [0.682, 0.798] | 0.444 [0.375, 0.506] | 0.555 [0.489, 0.613] | 0.497 [0.417, 0.572] | 0.271 [0.224, 0.322] |
| old s | 010 test block | 1590 | 0.854 [0.780, 0.924] | 0.667 [0.524, 0.796] | 0.749 [0.643, 0.842] | 0.673 [0.527, 0.806] | 0.278 [0.212, 0.352] |
| **new s** | 020 (all frames) | | pending | pending | pending | pending | pending |
| **new s** | 010 test block | | pending | pending | pending | pending | pending |
| old m | 020 (all frames) | | pending | pending | pending | pending | pending |
| old m | 010 test block | | pending | pending | pending | pending | pending |
| **new m** | 020 (all frames) | | pending | pending | pending | pending | pending |
| **new m** | 010 test block | | pending | pending | pending | pending | pending |

## (a) DUT test — all 4 classes (all4_test: DUT + COCO + Open Images test; eval.py protocol)

P/R are Ultralytics' values at the max-F1 confidence (as in all earlier eval_*.json).

| model | class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| old n | **overall** | 0.912 | 0.806 | 0.860 | 0.646 |
| old n | drone | 0.979 | 0.956 | 0.975 | 0.732 |
| old n | bird | 0.832 | 0.591 | 0.670 | 0.450 |
| old n | airplane | 0.892 | 0.774 | 0.850 | 0.667 |
| old n | helicopter | 0.944 | 0.905 | 0.945 | 0.734 |
| new n | all | pending | | | |
| old s | **overall** | 0.916 | 0.807 | 0.861 | 0.660 |
| old s | drone | 0.987 | 0.967 | 0.979 | 0.749 |
| old s | bird | 0.816 | 0.591 | 0.664 | 0.449 |
| old s | airplane | 0.914 | 0.767 | 0.856 | 0.684 |
| old s | helicopter | 0.949 | 0.903 | 0.944 | 0.759 |
| new s | all | pending | | | |
| old m | all | pending | | | |
| new m | all | pending | | | |

DUT-only test (2,200 images, drones only; `dut_anti_uav.yaml`):

| model | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| old n | 0.978 | 0.962 | 0.976 | 0.732 |
| new n | pending | | | |
| old s | 0.987 | 0.967 | 0.979 | 0.749 |
| new s | pending | | | |
| old m | pending | | | |
| new m | pending | | | |

## (b) False drones (bird/airplane/helicopter-only test images with ≥ 1 drone detection at conf 0.25)

| model | images with a false drone | of | false drones total |
|---|---|---|---|
| old n | 9 | 1534 | 10 |
| new n | pending | | |
| old s | 5 | 1534 | 5 |
| new s | pending | | |
| old m | pending | | |
| new m | pending | | |

## (d) Training facts

| model | epochs run | best epoch | wall-clock h | batch | time cap h | stop | train images (DUT drone / negatives / Det-Fly) |
|---|---|---|---|---|---|---|---|
| old n | 174 | 144 | 5.73 | 8 | 7.0 | early stopping (patience) | 5,200 / 5,000 / 0 |
| new n | pending | | | | | | |
| old s | 106 | 76 | 6.02 | 8 | 7.0 | early stopping (patience) | 5,200 / 5,000 / 0 |
| new s | pending | | | | | | |
| old m | 57 | 56 | 7.01 | 4 | 7.0 | time cap / epoch limit | 5,200 / 5,000 / 0 |
| new m | pending | | | | | | |

Recipe differences, honestly: old n, s and new n, s: batch 8; old m and new m: batch 4 (old m ran out of memory at batch 8; nbs = 64 keeps the effective batch at 64 via gradient accumulation). All runs: epochs 150, time cap 7 h unless marked SHORTENED, patience 30, imgsz 1280, seed 0, optimizer auto, default augmentation, COCO-pretrained start. With `time` set, Ultralytics re-fits the epoch count and LR schedule to the wall-clock budget, so a larger training set means fewer epochs in the same time. Old m hit the 7 h time cap (it did not early-stop).

## Weights (never committed)

| model | path | SHA-256 |
|---|---|---|
| old n | `model_size_benchmark/train/n_batch8/weights/best.pt` | `1888527f1b8f5fd55c977d56fe99a68dd00f86b23571ebd828782a6d7164e8a9` |
| old s | `runs/drone4_bal_backup.aVl2ZP/best.pt` | `cf148f9782974d94c5c1e380b674af78de6d5aaa66f9e89b5f1139ac31639900` |
| old m | `model_size_benchmark/train/m_batch4/weights/best.pt` | `f2c85e52e453d26c7ac5c2e9695e1366e3f347c86d67adba3c3e3f6d51dd3b53` |

All numbers: `detfly_retrain/summary.json`, per-run JSONs in `detfly_retrain/eval/` and `detfly_retrain/train_<size>.json`. Produced by `scripts/run_retrain_queue.py`.
