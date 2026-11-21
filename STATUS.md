# STATUS — Det-Fly retrain queue

Updated: Sat 2026-11-21 02:37 CST. Deadline: Sun 2026-11-22 22:00 CST.

**Stage: old_eval (old n/s/m on full Det-Fly + DUT)** — old_n: false drones


## Finished stages

* split

## Outputs

* `DETFLY_FULL_RESULTS.md` (old n/s/m on full Det-Fly)
* `DETFLY_RETRAIN_RESULTS.md` (old vs new per size)
* `detfly_split.json` (split definition)
* `detfly_retrain/` (all JSON results)
* `presentation_assets_v2/NUMBERS.md` (slide numbers, written last)

## Notes / anything odd

* none

## Caveats to say out loud (kept up to date by the agent)

* **020 is the same site as 010** (same hill, buildings, sports courts, dusk light; both 3840×2160): it is a
  later part of the same recording / same location, not an independent cross-site test.
* **020 drones are much smaller** than the Det-Fly training frames: median √(w·h) after letterboxing to 1280 is
  16.6 px on 020 vs 34.6 px on the 010 train frames and 28.7 px on the 010 test block
  (`presentation_assets_v2/dataset_stats.md`). 020 is the harder set.
* One Det-Fly frame is missing from the source release (020/0207614: XML but no JPEG); 020_0201689 has no drone box.
* The old 1,000-frame "preliminary" Det-Fly slice lies inside the new 010 training block (500 of its frames are
  new training data), so new models are not scored on it.
* Recipe differences: old/new m use batch 4 (old m ran out of memory at batch 8), n/s batch 8; with `time` set,
  Ultralytics fits the epoch count to the 7 h budget, so the larger new training set gets fewer epochs in the
  same time. New runs select best.pt / early-stop on a val set that includes 377 Det-Fly frames; old runs did not.
* Latency is not re-measured (identical architecture); old numbers reused.
