# NUMBERS — every slide number / asset, OLD → NEW

Read this first. NEW = YOLO26s retrained with 1,808 Det-Fly 010 frames added (same recipe otherwise). OLD = the deployed s from the earlier slides. Det-Fly 020 is the same site as the 010 training frames (later part of the same recording); say so on the slide. Sim numbers are SIMULATION ONLY; latency is DESKTOP GPU.

| number | OLD | NEW | source (file / command) |
|---|---|---|---|
| Training: epochs run (slide 6) | 106 | n/a | `detfly_retrain/old_training_facts.json`, `detfly_retrain/train_s.json` |
| Training: best epoch | 76 | n/a | same |
| Training: wall-clock h | 6.02 | n/a | same (results.csv time column) |
| Training: stop reason | early stopping (patience) | n/a | training log |
| Training images (DUT drone / negatives / Det-Fly) | 5,200 / 5,000 / 0 | 5,200 / 5,000 / 1,808 | `presentation_assets_v2/dataset_stats.md` |
| Drone share of training images (slide 5) | 51.0% | 58.3% | `presentation_assets_v2/dataset_stats.md` Table 1 |
| DUT test: P (overall, slide 7) | n/a | n/a | `detfly_retrain/eval/old_s_dut_test.json` → `new_s_dut_test.json` (detector/eval.py) |
| DUT test: R (overall, slide 7) | n/a | n/a | `detfly_retrain/eval/old_s_dut_test.json` → `new_s_dut_test.json` (detector/eval.py) |
| DUT test: mAP50 (overall, slide 7) | n/a | n/a | `detfly_retrain/eval/old_s_dut_test.json` → `new_s_dut_test.json` (detector/eval.py) |
| DUT test: mAP50-95 (overall, slide 7) | n/a | n/a | `detfly_retrain/eval/old_s_dut_test.json` → `new_s_dut_test.json` (detector/eval.py) |
| DUT+COCO+OI test: P (overall, slide 7) | 0.916 | n/a | `detfly_retrain/eval/old_s_all4_test.json` → `new_s_all4_test.json` (detector/eval.py) |
| DUT+COCO+OI test: R (overall, slide 7) | 0.807 | n/a | `detfly_retrain/eval/old_s_all4_test.json` → `new_s_all4_test.json` (detector/eval.py) |
| DUT+COCO+OI test: mAP50 (overall, slide 7) | 0.861 | n/a | `detfly_retrain/eval/old_s_all4_test.json` → `new_s_all4_test.json` (detector/eval.py) |
| DUT+COCO+OI test: mAP50-95 (overall, slide 7) | 0.660 | n/a | `detfly_retrain/eval/old_s_all4_test.json` → `new_s_all4_test.json` (detector/eval.py) |
| all4 test drone: R / mAP50 | 0.967 / 0.979 | n/a / n/a | same |
| all4 test bird: R / mAP50 | 0.591 / 0.664 | n/a / n/a | same |
| all4 test airplane: R / mAP50 | 0.767 / 0.856 | n/a / n/a | same |
| all4 test helicopter: R / mAP50 | 0.903 / 0.944 | n/a / n/a | same |
| False-drone images at conf 0.25 (of 1,534) (slide 7) | n/a | n/a | `detfly_retrain/eval/*_false_positives.json` (detector/false_positives.py) |
| False drones total at conf 0.25 | n/a | n/a | same |
| Det-Fly PRELIMINARY 1,000 (old slide 8): P / R / mAP50 (Ultralytics max-F1) | 0.842 / 0.285 / 0.304 | not applicable (500 of these frames are now training data) | `detfly_retrain/eval/old_s_prelim.json` |
| Det-Fly 020 (held out, 6,912 frames): P@0.25 [95% CI] (slide 8) | 0.741 [0.682, 0.798] | n/a | `detfly_retrain/eval/old_s_test020.json` → `new_s_test020.json` (detector/new_eval_detfly.py) |
| Det-Fly 020 (held out, 6,912 frames): R@0.25 [95% CI] (slide 8) | 0.444 [0.375, 0.506] | n/a | `detfly_retrain/eval/old_s_test020.json` → `new_s_test020.json` (detector/new_eval_detfly.py) |
| Det-Fly 020 (held out, 6,912 frames): F1@0.25 [95% CI] (slide 8) | 0.555 [0.489, 0.613] | n/a | `detfly_retrain/eval/old_s_test020.json` → `new_s_test020.json` (detector/new_eval_detfly.py) |
| Det-Fly 020 (held out, 6,912 frames): mAP50 [95% CI] (slide 8) | 0.497 [0.417, 0.572] | n/a | `detfly_retrain/eval/old_s_test020.json` → `new_s_test020.json` (detector/new_eval_detfly.py) |
| Det-Fly 020 (held out, 6,912 frames): mAP50-95 [95% CI] (slide 8) | 0.271 [0.224, 0.322] | n/a | `detfly_retrain/eval/old_s_test020.json` → `new_s_test020.json` (detector/new_eval_detfly.py) |
| Det-Fly 010 test block (1,590): P@0.25 [95% CI] (slide 8) | 0.854 [0.780, 0.924] | n/a | `detfly_retrain/eval/old_s_test010.json` → `new_s_test010.json` (detector/new_eval_detfly.py) |
| Det-Fly 010 test block (1,590): R@0.25 [95% CI] (slide 8) | 0.667 [0.524, 0.796] | n/a | `detfly_retrain/eval/old_s_test010.json` → `new_s_test010.json` (detector/new_eval_detfly.py) |
| Det-Fly 010 test block (1,590): F1@0.25 [95% CI] (slide 8) | 0.749 [0.643, 0.842] | n/a | `detfly_retrain/eval/old_s_test010.json` → `new_s_test010.json` (detector/new_eval_detfly.py) |
| Det-Fly 010 test block (1,590): mAP50 [95% CI] (slide 8) | 0.673 [0.527, 0.806] | n/a | `detfly_retrain/eval/old_s_test010.json` → `new_s_test010.json` (detector/new_eval_detfly.py) |
| Det-Fly 010 test block (1,590): mAP50-95 [95% CI] (slide 8) | 0.278 [0.212, 0.352] | n/a | `detfly_retrain/eval/old_s_test010.json` → `new_s_test010.json` (detector/new_eval_detfly.py) |
| Box noise 16-24 px: centre σ px / log-width σ (slide 10) | 0.88 / 0.094 | n/a | `experiments/detector_noise.json` → `presentation_assets_v2/detector_noise.json` (detector/measure_noise.py) |
| Box noise 24-32 px: centre σ px / log-width σ (slide 10) | 0.89 / 0.073 | n/a | `experiments/detector_noise.json` → `presentation_assets_v2/detector_noise.json` (detector/measure_noise.py) |
| Box noise 32-48 px: centre σ px / log-width σ (slide 10) | 1.06 / 0.047 | n/a | `experiments/detector_noise.json` → `presentation_assets_v2/detector_noise.json` (detector/measure_noise.py) |
| Box noise: matched / GT drones | 2196 / 2245 | n/a / n/a | same |
| Latency pytorch_fp16 mean ms (slide 9, DESKTOP GPU) | 5.4 | 5.4 (reused: same architecture) | `experiments/latency.json` |
| Latency pytorch_fp32 mean ms (slide 9, DESKTOP GPU) | 7.0 | 7.0 (reused: same architecture) | `experiments/latency.json` |
| Latency onnx_fp16 mean ms (slide 9, DESKTOP GPU) | 8.0 | 8.0 (reused: same architecture) | `experiments/latency.json` |
| Latency onnx_fp32 mean ms (slide 9, DESKTOP GPU) | 11.1 | 11.1 (reused: same architecture) | `experiments/latency.json` |
| Weaving + tight prior: 2σ coverage (slide 22) | 94.9% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_tight_prior] |
| Weaving + tight prior: ANEES range (slide 22) | 1.07 | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_tight_prior] |
| Straight + size prior: 2σ coverage ('33 %', slide 23) | 33.3% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_straight_size_prior] |
| Straight + size prior: RMS rel range error | 42.5% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_straight_size_prior] |
| Straight + size prior: reported σ/r ('~7 %', slide 20) | 6.7% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_straight_size_prior] |
| Weaving + size prior: RMS rel range error | 12.5% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_size_prior] |
| Weaving + size prior: reported σ/r ('4 %', slide 20) | 3.7% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_size_prior] |
| Weaving + size prior: 2σ coverage | 77.7% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_size_prior] |
| Weaving + size prior: RMS rel err at 10-20m (slide 21) | 5.5% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_size_prior] |
| Weaving + size prior: RMS rel err at 20-40m (slide 21) | 6.2% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_size_prior] |
| Weaving + size prior: RMS rel err at 40-80m (slide 21) | 6.1% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_size_prior] |
| Weaving + size prior: RMS rel err at 80-120m (slide 21) | 13.1% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_size_prior] |
| Weaving + size prior: RMS rel err at 120-160m (slide 21) | 31.4% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_size_prior] |
| Weaving + size prior: RMS rel err at 160-200m (slide 21) | 66.9% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_size_prior] |
| Straight + wrong prior: bias ('27 % short', slide 23) | -27.1% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_straight_wrong_prior] |
| Weaving + wrong prior: bias | -7.8% | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [A_approach_weaving_wrong_prior] |
| Model-truth weaving 40m: ANEES range (slide 22) | 1.80 | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [E_model_truth_weaving_40m] |
| Model-truth weaving 40m: ANEES pos | 3.86 | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [E_model_truth_weaving_40m] |
| Model-truth weaving 80m: ANEES range (slide 22) | 2.62 | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [E_model_truth_weaving_80m] |
| Model-truth weaving 80m: ANEES pos | 4.83 | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [E_model_truth_weaving_80m] |
| Model-truth weaving 150m: ANEES range (slide 22) | 7.24 | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [E_model_truth_weaving_150m] |
| Model-truth weaving 150m: ANEES pos | 10.01 | n/a | `experiments/results_sim.json` → `experiments/new_results/results_sim.json` [E_model_truth_weaving_150m] |
| NaN runs (of 9,600) (slide 23) | 321 | n/a | `experiments/nan_runs.json` → `experiments/new_results/nan_runs.json` |

## Assets (same file names as presentation_assets/)

| asset | what changed | produced by |
|---|---|---|
| PR_curve.png, confusion_matrix.png | new s validated on the OLD val split (COCO + DUT + Open Images val, 4,316 images) so they compare 1:1 with the old ones; `*_newval.png` = on the new val split incl. 377 Det-Fly frames | `scripts/new_final_assets.sh` (Ultralytics val plots) |
| results.png | training curves of the new s run (`runs/s_detfly_new/results.png`) | Ultralytics training |
| confusion_matrix_conf025{,_counts}.png/.json | conf 0.25, NMS IoU 0.5, old val split | `detector/confusion_matrix_conf.py --weights runs/s_detfly_new/weights/best.pt --out presentation_assets_v2` |
| dut_good_small/medium/large.jpg | same DUT images as before, new model's boxes (old conf/IoU in the title) | `detector/new_presentation_examples.py` (`examples.json`) |
| detfly_missed_1/2.jpg | Det-Fly 020 frames the NEW model misses; title says what the old model did | same |
| dataset_stats.md/.json | composition incl. Det-Fly train/val/test | `scripts/new_dataset_stats.py` |
| detector_results.md | new s numbers with old s in brackets | `scripts/new_slide_assets.py` |
| detector_noise.json/.png | box noise re-measured with new s on DUT test | `detector/measure_noise.py --weights runs/s_detfly_new/weights/best.pt` |
| model_size_table.md/.json | old vs new n/s/m | `scripts/new_slide_assets.py` |
| rmse_vs_range, nees_time, range_error_envelopes, uncertainty_vs_motion, consistency_model_truth .png; results_sim.md; nan_runs.md | sim rerun with the new noise (also in `experiments/new_results/`) | `scripts/new_sim_shadow.sh` |

Numbers read off figures only (not in any JSON), e.g. the 68th-percentile actual error on `uncertainty_vs_motion.png` (slide 20), must be re-read from the new figure.
