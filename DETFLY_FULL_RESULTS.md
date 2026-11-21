# Det-Fly full-release results — OLD n / s / m models

> **Read first (choices made without asking, per instructions):**
> * **020 is the same site as 010** (same hill, buildings, sports courts and dusk light in sampled frames; both 3840×2160). It is treated as a *later part of the same recording*, not an independent cross-site test. Det-Fly as a whole is still a different dataset/camera from the DUT training data.
> * 13,270 Det-Fly frames: 010 = 6,358, 020 = 6,912. Frame 020/0207614 is missing from the source release (its XML exists, the JPEG does not); 020_0201689 has no UAV box (kept as a background frame), so 020 has 6,911 drone boxes.
> * Split (`detfly_split.json`): 010 cut at 60 %/75 % of its frames, 200-frame buffers dropped from the end of the train and val blocks, train and val subsampled with stride 2 → 1,808 train / 377 val frames; 010 test = last 25 % (1,590 frames, all kept); 020 = 6,912 test frames, never trained or validated on.
> * Protocol: Ultralytics val, imgsz 1280, batch 8, conf 0.001, NMS IoU 0.7 (as detector/eval.py), drone class only. P/R/F1 at confidence ≥ 0.25 with IoU ≥ 0.5 matching (recomputed from the validator's own matches). 95 % CI = block bootstrap, 100-frame time-ordered chunks, 1,000 resamples, percentile.

## Old models on full Det-Fly (drone class)

| model | Det-Fly set | frames | P @0.25 [95% CI] | R @0.25 [95% CI] | F1 @0.25 [95% CI] | mAP50 [95% CI] | mAP50-95 [95% CI] |
|---|---|---|---|---|---|---|---|
| old n | 020 (all frames) | | pending | pending | pending | pending | pending |
| old n | 010 test block | | pending | pending | pending | pending | pending |
| old s | 020 (all frames) | | pending | pending | pending | pending | pending |
| old s | 010 test block | | pending | pending | pending | pending | pending |
| old m | 020 (all frames) | | pending | pending | pending | pending | pending |
| old m | 010 test block | | pending | pending | pending | pending | pending |

## Sanity check: old s on the 1,000-frame preliminary slice

pending

## Old weights

| model | weights | SHA-256 |
|---|---|---|
| old n | `model_size_benchmark/train/n_batch8/weights/best.pt` | `1888527f1b8f5fd55c977d56fe99a68dd00f86b23571ebd828782a6d7164e8a9` |
| old s | `runs/drone4_bal_backup.aVl2ZP/best.pt` | `cf148f9782974d94c5c1e380b674af78de6d5aaa66f9e89b5f1139ac31639900` |
| old m | `model_size_benchmark/train/m_batch4/weights/best.pt` | `f2c85e52e453d26c7ac5c2e9695e1366e3f347c86d67adba3c3e3f6d51dd3b53` |

Produced by `scripts/run_retrain_queue.py` (stage old_eval) → `detector/new_eval_detfly.py`; JSONs in `detfly_retrain/eval/old_<size>_<set>.json`.
