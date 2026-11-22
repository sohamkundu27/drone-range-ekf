# STATUS — Det-Fly retrain: DONE

Finished Sun 2026-11-22 00:18 CST (deadline Sun 22:00 CST). Every stage ran and every result is pushed.

**Read first:** `presentation_assets_v2/NUMBERS.md` (every slide number OLD → NEW with source), then
`DETFLY_RETRAIN_RESULTS.md` (old vs new n/s/m) and `DETFLY_FULL_RESULTS.md` (old models on full Det-Fly).

## Result in one table (Det-Fly 020 = 6,912 held-out frames, conf 0.25, 95 % block-bootstrap CI)

| model | 020 recall old → new | 020 mAP50 old → new | DUT drone mAP50 old → new | false-drone images /1,534 old → new | decision rule |
|---|---|---|---|---|---|
| n | 0.460 [0.392, 0.521] → 0.684 [0.612, 0.746] | 0.498 → 0.730 | 0.976 → 0.976 | 9 → 7 | met |
| s | 0.444 [0.375, 0.506] → 0.711 [0.649, 0.768] | 0.497 → 0.752 | 0.979 → 0.980 | 5 → 8 | met |
| m | 0.485 [0.409, 0.557] → 0.729 [0.661, 0.785] | 0.541 → 0.775 | 0.982 → 0.988 | 3 → 2 | met |

Decision rule, stated only (no model adopted): *new is clearly better if Det-Fly 020 recall is up beyond the CI
AND DUT mAP50 and false-drone count are not clearly worse.* "Not clearly worse" was set by me as: DUT drone mAP50
drop < 0.01 and ≤ 5 more false-drone images.

## Stages

split → old n/s/m eval → train+eval new s → slide assets v2 + sim (from new s) → train+eval new n →
train+eval new m → final assets/report. All exited 0; no batch reductions, no OOM, no resumes, no failed pushes.

| run | epochs | best epoch | hours | batch | stop |
|---|---|---|---|---|---|
| new s | 106 | 92 | 7.01 | 8 | 7 h time cap (old s: early stop at 106, best 76, 6.02 h) |
| new n | 182 | 158 | 7.00 | 8 | 7 h time cap (old n: early stop at 174, best 144, 5.73 h) |
| new m | 49 | 49 | 7.01 | 4 | 7 h time cap (old m: time cap at 57, best 56) |

## Caveats to say out loud in the talk

* **020 is the same site as 010** (same hill, buildings, sports courts, dusk light): it is a *later part of the same
  recording*, not an independent cross-site test. Det-Fly itself is still a different dataset/camera from DUT.
* **020 drones are much smaller** than the Det-Fly training frames (median 16.6 px vs 34.6 px at the 1280 input).
* **One frame is missing** from the source release (020/0207614: XML but no JPEG); 020_0201689 has no drone box.
* **The old "preliminary" 1,000-frame Det-Fly slide number (R 0.285) is superseded**: on all 6,912 frames of 020
  the old s has R 0.444 (conf 0.25). 500 of the preliminary frames are now training data, so the new models are not
  scored on that slice.
* **Recipe mismatches**: m uses batch 4 (old and new; batch 8 runs out of memory), n/s batch 8. All runs had a 7 h
  budget; with more training images the new runs got fewer epochs (m: 49 vs 57) and all three new runs hit the time
  cap instead of early-stopping. **New m was still improving at the cap** (best epoch = last epoch), so it is if
  anything under-trained. New runs chose best.pt on a val set that includes 377 Det-Fly frames; old runs did not.
* **On the 010 test block, precision went down while recall went up** (s: P 0.854 → 0.790, R 0.667 → 0.762; m: P
  0.913 → 0.800); the 010-test CIs are wide (1,590 frames, 16 chunks). On 020 both P and R went up.
* **New s false drones rose 5 → 8 images** of 1,534 (n and m went down).
* **Sim with the new noise**: the measured noise barely changed (σ within ~±10 %), so sim numbers move slightly;
  NaN runs 321 → 315 of 9,600 (NaN bug kept as is). The weaving + size-prior ANEES jumps 3.9 → 22.5 because of ONE
  run with an exploding (finite) NEES; without it 3.6 → 3.8 (`experiments/new_results/COMPARISON.md`).

## Not done / reused

* Latency not re-measured (identical architecture and input size); old numbers reused and labelled as such.
* Numbers that exist only in figures (e.g. the 68th-percentile actual error on `uncertainty_vs_motion.png`, slide 20)
  must be re-read from the new figure; they are not in any JSON.
* The old model's view of the two Det-Fly miss frames is given in the image title only (both models miss both
  frames), not as separate images, to stay within the 5 committed example images.
* `PR_curve.png` / `confusion_matrix.png` in `presentation_assets_v2/` are on the OLD val split (1:1 comparable with
  the old slides); the versions on the new val split (incl. Det-Fly val) are `*_newval.png`.
* Weights are not committed: `runs/{n,s,m}_detfly_new/weights/best.pt` (SHA-256 in `detfly_retrain/weights_sha256.json`).
