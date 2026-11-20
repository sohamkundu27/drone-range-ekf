# Training details

Evidence: local code, configuration, checkpoints, dataset-conversion outputs, and run outputs only. No README/presentation narrative or external documentation was used. No training or evaluation was rerun.

Source notation (all paths relative to the repository root):

- `A:key` = key in `runs/detect/drone4_bal/args.yaml`.
- `L:line` = `runs/logs/train_drone4_bal.log:line`.
- `C:line` = `runs/detect/drone4_bal/results.csv:line`.
- `E:line` = `experiments/training_details_evidence.json:line`. This companion file contains **new read-only audit extracts**, not a historical log: checkpoint metadata, saved loader-cache counts, manifest counts, and file hashes. Extraction methods are at `E:2–13`.
- `U/` = `.venv/lib/python3.12/site-packages/ultralytics/`. The installed source reports `8.4.215`, matching the training log (`U/__init__.py:3`; `L:1`). Source-derived behavior below is identified separately from saved settings. An immutable training-time environment/source snapshot is **NOT FOUND** [Q1].
- Log line numbers count newline characters, not carriage-return progress-bar updates.

## 1. Model

| Item | Recorded value | Source |
|---|---|---|
| Architecture | YOLO26s detection model; scale `s`; 3 input channels; `reg_max=1`; detection strides 8, 16, 32. Model YAML has `end2end=true` and the checkpoint retains the one-to-one head. | `L:29–30`; `E:97–113`; `U/cfg/models/26/yolo26.yaml:8–14,20–52` |
| Training model size | 260 layers; 9,950,960 parameters and gradients. | `L:30` |
| Final fused evaluation model | 120 layers; 9,466,728 parameters. | `L:694–696` |
| Initialization file | Root-level `yolo26s.pt`; `pretrained=true`; no project checkpoint resume. | `A:model,pretrained,resume`; `detector/train.py:64–75` |
| Pretrained from what | The initialization checkpoint names `/home/lq/codes/ultralytics/ultralytics/cfg/datasets/coco.yaml` as its training dataset and contains 80 class names, including airplane and bird. Its preceding initialization file is recorded as `yolo26s-objv1-150.pt`. This supports COCO pretraining metadata, but the preceding checkpoint, complete upstream training history/dataset snapshot, and original download URL are **NOT FOUND** [Q2]. | `E:16–59` (extracted from `yolo26s.pt`) |
| Transfer | Original `nc=80` replaced by `nc=4`; 708/708 weight items transferred; 2/4 classifier rows remapped by class name. The matching names are bird and airplane. | `L:3,32–33`; `A:cls_remap`; `E:56–59,115–119` |
| Classes | `0: drone`, `1: bird`, `2: airplane`, `3: helicopter`; four classes, `single_cls=false`. | `data/yolo/drone4_bal.yaml:11–15`; `A:single_cls` |
| Selected detector | `runs/detect/drone4_bal/weights/best.pt`; SHA-256 `cf148f9782974d94c5c1e380b674af78de6d5aaa66f9e89b5f1139ac31639900`. | `L:687,692,694`; `E:75–80` |
| Final-epoch detector | `runs/detect/drone4_bal/weights/last.pt`; distinct from `best.pt`. Both files exist and have stripped optimizer state. | `L:691–692`; `E:75–80,132–137` |

The model architecture supports two training branches; this does **not** establish NMS-free evaluation. The recorded evaluation configuration follows the NMS path described in section 8 (`A:nms`; `U/engine/validator.py:237–238,253–264`).

## 2. Hyperparameters

“Explicit” below means passed by the project training script or its command-line arguments. The call does not pass optimizer, LR, momentum, weight decay, warmup, or loss/augmentation overrides (`detector/train.py:47–76`).

| Setting | Saved setting and effective behavior | Source |
|---|---|---|
| Optimizer | Saved `auto`, inherited rather than explicitly chosen in the project call. Log confirms automatic selection of **MuSGD**, with base LR `0.01` and constructor momentum `0.9`. | `A:optimizer`; `L:44–45`; `detector/train.py:71–76` |
| Initial LR | Saved `lr0=0.01`; auto-selection explicitly ignores that input and independently chooses `0.01`. Local MuSGD setup gives selected classifier parameter groups 3× LR: base rates `0.03` and `0.01` before warmup/scheduling. | `A:lr0`; `L:44–45`; `U/engine/trainer.py:1232–1234,1272–1284` |
| Final LR fraction | `lrf=0.01`. Planned terminal base rates are therefore `0.0003` for boosted groups and `0.0001` for the other groups; these are schedule targets, not the rates reached at early stopping. | `A:lrf`; `U/engine/trainer.py:270–276,1283` |
| Observed LR, first completed epoch | `lr/pg0,2,4,6=0.00999216`; `lr/pg1,3,5,7=0.00333072` during warmup. | `C:1–2` |
| Observed LR, last completed epoch | At epoch 106: `lr/pg0,2,4,6=0.00485081`; `lr/pg1,3,5,7=0.00161694`. | `C:1,107` |
| LR schedule | `cos_lr=false`. Installed code uses linear multiplier `max(1 - epoch / epochs, 0) * (1 - lrf) + lrf`. With the time budget enabled, the epoch horizon and scheduler are recomputed after each epoch. The final displayed horizon is 124. | `A:cos_lr,time`; `U/engine/trainer.py:270–276,690–695`; `L:683`; `E:141` |
| Warmup | `warmup_epochs=3.0`, `warmup_momentum=0.8`. Saved `warmup_bias_lr=0.1`, but auto-optimizer setup changes it to `0.0`; the checkpoint confirms `0.0`. LR interpolates from zero to each group's scheduled initial LR. | `A:warmup_epochs,warmup_momentum,warmup_bias_lr`; `U/engine/trainer.py:278–281,524–542,1234`; `E:93` |
| Momentum | Saved/configured target `0.937`; auto-optimizer constructor logs `0.9`. The warmup code subsequently interpolates each group's momentum from `0.8` to configured `0.937`. Do not describe `0.9` as a verified constant throughout training. | `A:momentum,warmup_momentum`; `L:45`; `U/engine/trainer.py:541–542`; `U/optim/muon.py:238–248` |
| Weight decay | `0.0005`; optimizer log reports decay `0.0005` for weight groups and `0.0` for normalization weights/biases. | `A:weight_decay`; `L:45` |
| Batch size | Explicit `8`. `nbs=64`; local code gives accumulation target `round(64/8)=8` batches per optimizer step, approached during warmup. | `A:batch,nbs`; `detector/train.py:72`; `U/engine/trainer.py:325–326,528` |
| Epochs requested | Explicit `150`, together with explicit `time=7.0` hours; this is a time-limited run with an adaptive epoch horizon. | `A:epochs,time`; `L:2,50`; `detector/train.py:72–73` |
| Image size | Explicit `1280`. | `A:imgsz`; `detector/train.py:72` |
| Patience | Explicit `30` epochs without improvement. | `A:patience`; `detector/train.py:73`; `L:687–688` |
| Seed | Explicit `0`. | `A:seed`; `detector/train.py:75` |
| Deterministic flag | Explicit `false` (installed default is `true`). | `A:deterministic`; `detector/train.py:75`; `U/cfg/default.yaml:30` |
| Other execution settings | Explicit GPU `device=0`, `workers=12`; inherited `amp=true`, `cache=false`, `fraction=1.0`, `freeze=null`, `compile=false`. AMP checks passed. | `A:device,workers,amp,cache,fraction,freeze,compile`; `detector/train.py:72–75`; `L:34–35,47–48` |

Final optimizer tensors and per-step LR/momentum traces are **NOT FOUND** [Q3]. Both checkpoints have been stripped; the CSV retains epoch-level LR values (`L:691–692`; `E:80,137`; `C:1`).

## 3. Loss

The run actually logs `box_loss`, `cls_loss`, and `l1_loss`, for both training and validation (`C:1`; `L:52`). No project loss override is passed (`detector/train.py:71–76`).

| Term | Gain | Implementation evidence |
|---|---|---|
| Box overlap | `box=7.5` | CIoU: weighted `(1 - CIoU)` in `U/utils/loss.py:140–143`; gain applied at `U/utils/loss.py:483`; recorded value `A:box`. |
| Class prediction | `cls=0.5` | `BCEWithLogitsLoss`, summed and normalized by target scores: `U/utils/loss.py:366,462–468,484`; recorded value `A:cls`. |
| Box distance | `dfl=1.5`, applied to **L1** for this model | `reg_max=1` (`E:101`; `L:29`), so the implementation takes its normalized L1 branch, not DFL: `U/utils/loss.py:125,150–162,374–375,485`; recorded value `A:dfl`. |

The detection criterion selects `E2ELoss` when the one-to-one head exists (`U/nn/tasks.py:651–653`; `E:112`). Its local implementation combines one-to-many and one-to-one losses: initial branch weights `0.8` and `0.2`; one-to-many weight decays toward `0.1`, and one-to-one uses the complement toward `0.9`. Assignment settings are `tal_topk=10` and `tal_topk=7, tal_topk2=1`, respectively. The displayed loss components come from the one-to-one branch, not the weighted total. These are installed-code settings, not keys separately recorded in the run YAML (`U/utils/loss.py:1327–1356`; epoch update at `U/engine/trainer.py:650–651`).

Class-frequency reweighting is disabled by `cls_pw=0.0`; distillation is unconfigured (`distill_model=null`). Pose/keypoint/angle/depth gains present in the generic YAML do not establish additional detection loss terms (`A:cls_pw,distill_model`; `U/engine/trainer.py:1009–1011`; `U/cfg/default.yaml:102–114`; `U/utils/loss.py:374–375,483–485`).

## 4. Augmentation

**Explicit project augmentation changes: none in the saved training call.** All settings below are omitted from `model.train(...)` and match the installed defaults. The recorded values come from the final run's `args.yaml`, not from the initialization checkpoint's pretraining hyperparameters (`detector/train.py:71–76`; `U/cfg/default.yaml:32–40,116–133`; comparison at `E:450–476`).

| Inherited setting | Recorded value | Source / applicability |
|---|---|---|
| `mosaic` | `1.0` | `A:mosaic` |
| `close_mosaic` | `10` | `A:close_mosaic`; actual closure behavior below |
| `mixup` | `0.0` | `A:mixup` |
| `cutmix` | `0.0` | `A:cutmix` |
| `copy_paste` | `0.0` | `A:copy_paste`; disabled |
| `copy_paste_mode` | `flip` | `A:copy_paste_mode`; inactive with zero probability |
| `hsv_h`, `hsv_s`, `hsv_v` | `0.015`, `0.7`, `0.4` | `A:hsv_h,hsv_s,hsv_v` |
| `degrees` | `0.0` | `A:degrees` |
| `translate` | `0.1` | `A:translate` |
| `scale` | `0.5` | `A:scale` |
| `shear` | `0.0` | `A:shear` |
| `perspective` | `0.0` | `A:perspective` |
| `flipud`, `fliplr` | `0.0`, `0.5` | `A:flipud,fliplr` |
| `bgr` | `0.0` | `A:bgr`; channel-swap augmentation disabled |
| `multi_scale` | `0.0` | `A:multi_scale`; disabled |
| `rect` | `false` | `A:rect`; training batching setting |
| `augment` | `false` | `A:augment`; inference/test-time augmentation flag, not a switch disabling training transforms (`U/engine/validator.py:235`) |
| `auto_augment` | `randaugment` | `A:auto_augment`; classification-only setting, not used by this detection transform pipeline (`U/cfg/default.yaml:132`) |
| `erasing` | `0.4` | `A:erasing`; classification-only setting (`U/cfg/default.yaml:133`) |

The detection pipeline constructs Mosaic, RandomPerspective, CopyPaste, MixUp, CutMix, optional Albumentations, HSV, and vertical/horizontal flips with these gains; BGR handling is in the final format transform (`U/data/augment.py:2874–2916`; `U/data/dataset.py:348–358`). `dropout=0.0`, `overlap_mask=true`, and `mask_ratio=4` also appear in the generic args, but are classification/segmentation settings, not extra active augmentations here (`A:dropout,overlap_mask,mask_ratio`; `U/cfg/default.yaml:44–49`).

`close_mosaic=10` did **not** produce a final ten-epoch no-mosaic phase. Closure is gated by the estimated end of the run; training stopped at `106/124` before that gate. There is no `Closing dataloader mosaic` message (`L:683,687`; `U/engine/trainer.py:512–515,1195–1201`; full-log scan `E:489–493`).

Historical activation of the optional Albumentations transforms is **NOT FOUND** [Q4]. No Albumentations setup appears in the run's initialization log, and no package is present in the inspected environment (`L:34–50`; `E:477`). The installed wrapper can silently skip an unavailable import; its Blur/MedianBlur/ToGray/CLAHE defaults therefore must not be asserted as active (`U/data/augment.py:2145–2155,2168–2202`).

## 5. Data

Shared output format: `class cx cy width height`, normalized, written to six decimal places; images are hard-linked or copied. The pixel-coordinate converters clip boxes to the image and drop nonpositive width/height boxes (`scripts/_common.py:25–50`). Class order is defined at `scripts/_common.py:21–22`.

| Dataset | Conversion/filtering and split rule | Source |
|---|---|---|
| DUT Anti-UAV | VOC XML; assert class `UAV`, map to drone `0`; use XML dimensions; preserve source `train`, `val`, `test` directories; skip missing JPEGs. | `scripts/convert_dut.py:19–37`; `scripts/_common.py:35–42` |
| COCO | Read `instances_train2017.json` and `instances_val2017.json` from the annotation ZIP. Keep category `16 → bird 1`, `5 → airplane 2`; exclude an entire image if a relevant annotation is `iscrowd`; other categories receive no labels. Convert COCO `x,y,w,h` boxes. `train2017`: validation if `(image_id * 2654435761) % 100 < 10`, otherwise training. `val2017 → test`. Failed image downloads are skipped. | `scripts/download_coco_subset.py:30,49–51,60–94,99–106` |
| Open Images | Roots `/m/015p6 → bird 1`, `/m/0cmf2 → airplane 2`, `/m/09ct_ → helicopter 3`, including their hierarchy descendants. Exclude images containing relevant `IsGroupOf`/`IsDepiction` boxes or generic Aircraft `/m/0k5j` / Rocket `/m/09rvcxw`. Use normalized bounds directly, dropping nonpositive boxes. Preserve source train/validation/test as train/val/test. Write only successful downloads. | `scripts/download_openimages_subset.py:38–44,47–71,104–111,129–146` |
| Det-Fly | VOC XML `UAV → drone 0`; require matching XML; use actual JPEG dimensions and count XML-size discrepancies; prefix image IDs with sequence; write only to `test`. | `scripts/convert_detfly.py:23–48` |

COCO's optional image cap defaults to `--max-per-class 0` (all eligible images); the capped branch uses seed `0`. Open Images defaults to `--cap-train 3000` for bird/airplane, uncapped training helicopters, and `--cap-eval 500`; it shuffles sorted IDs with seed `0`, selects helicopter then airplane then bird, and accounts for already-selected multi-class images (`scripts/download_coco_subset.py:56,61,72–78`; `scripts/download_openimages_subset.py:93–94,101,113–126`). **The historical converter/download command lines, actual cap overrides, and rejection/download-failure summaries are NOT FOUND** [Q5]. These defaults describe the code, not a recovered execution command.

The Open Images script specifically reads `oidv6-train-annotations-bbox.csv`, `validation-annotations-bbox.csv`, `test-annotations-bbox.csv`, `oidv7-class-descriptions-boxable.csv`, and `bbox_labels_600_hierarchy.json` (`scripts/download_openimages_subset.py:38–42,64–66`).

Full converted-directory counts, before training rebalancing and loader duplicate-label removal. “Instances” is the sum of the listed class counts; omitted classes have no rows in the saved count map (`scripts/dataset_stats.py:29–53`).

| Dataset / split | Images | Raw instances | Class instances | Source |
|---|---:|---:|---|---|
| DUT train | 5,200 | 5,243 | drone 5,243; 3 empty-label images | `figures/dataset_stats.json:158–163` |
| DUT val | 2,600 | 2,620 | drone 2,620 | `figures/dataset_stats.json:186–191` |
| DUT test | 2,200 | 2,245 | drone 2,245 | `figures/dataset_stats.json:214–219` |
| COCO train | 5,324 | 10,758 | bird 6,215; airplane 4,543 | `figures/dataset_stats.json:2–8` |
| COCO val | 604 | 1,178 | bird 673; airplane 505 | `figures/dataset_stats.json:31–37` |
| COCO test | 209 | 376 | bird 233; airplane 143 | `figures/dataset_stats.json:60–66` |
| Open Images train | 8,065 | 15,772 | bird 7,744; airplane 5,280; helicopter 2,748 | `figures/dataset_stats.json:270–277` |
| Open Images val | 1,112 | 1,569 | bird 824; airplane 611; helicopter 134 | `figures/dataset_stats.json:300–307` |
| Open Images test | 1,325 | 1,766 | bird 783; airplane 592; helicopter 391 | `figures/dataset_stats.json:330–337` |
| Det-Fly train / val | 0 / 0 | 0 / 0 | No converted training/validation split; test-only converter | `E:427–428`; `scripts/convert_detfly.py:42–44` |
| Det-Fly test | 1,000 | 1,000 | drone 1,000 | `figures/dataset_stats.json:118–123`; `E:340–348` |

The final training set is a **selected manifest**, not all of the converted training directories: `data/yolo/drone4_bal_train.txt` (`A:data`; `data/yolo/drone4_bal.yaml:1–2`). Rebalancing keeps all DUT training images. For COCO/Open Images it assigns a primary class by most boxes, ranks median normalized `sqrt(width*height)` ascending and background smoothness descending, and selects by the sum of those ranks. Smoothness is the fraction of unboxed pixels with Sobel magnitude below `20`, after grayscale resizing to longest side `256`; fewer than `100` background pixels gives zero. Quotas are bird `2,000`, airplane `1,800`, helicopter `1,200` images. These quotas are image counts by primary class, not instance counts (`scripts/rebalance_train.py:37–38,46–65,72–107`).

| Source in final training manifest | Images | Raw instances | Loader instances | Loader class instances | Source |
|---|---:|---:|---:|---|---|
| DUT | 5,200 | 5,243 | 5,243 | drone 5,243 | `E:216–225,373–382` |
| COCO | 2,159 | 5,617 | 5,617 | bird 3,592; airplane 2,025 | `E:205–214,384–393` |
| Open Images | 2,841 | 7,266 | 7,265 | bird 3,304; airplane 2,250; helicopter 1,711 | `E:227–236,395–404` |
| Det-Fly | 0 | 0 | 0 | Excluded | `E:406–415`; `scripts/rebalance_train.py:120` |
| Total | 10,200 | 18,126 | 18,125 | drone 5,243; bird 6,896; airplane 4,275; helicopter 1,711 | Sum of rows; loader total `E:193–202` |

The manifest has `10,200` unique paths and matches the saved training-cache image set (`E:366–371`). The balance output records `5,000` non-drone-class images and drone-source fraction `0.5098039215686274`; the `5,200` DUT images include the three empty-label images (`experiments/train_balance_stats.json:2,35–37`; `E:195,218`).

Loader accounting matters: one duplicate bird label is removed from Open Images training, three from validation, and five from test (`L:38,41–43`; `runs/logs/eval_all.log:42–46`). Effective Open Images validation is **1,112 images / 1,566 instances** (bird 821, airplane 611, helicopter 134); effective test is **1,325 / 1,761** (bird 778, airplane 592, helicopter 391). COCO/DUT counts are unchanged (`E:254–285,303–334`). Thus the final combined validation total is **4,316 / 5,364**, and combined test is **3,734 / 4,382**, matching the logs (`L:698–702`; `runs/logs/eval_all.log:50–54`).

Checkpoint selection uses **`drone4_bal.yaml` → `val`**, the union of COCO val + DUT val + Open Images val. Held-out detector tests use its separate **`test`** union and, separately, Det-Fly test. Rebalancing changes only training; validation/test lists are copied unchanged from `drone4.yaml` (`A:val,split`; `data/yolo/drone4_bal.yaml:3–10`; `scripts/rebalance_train.py:115–120`; `detector/run_eval_all.sh:10–13`).

The per-dataset YAML files intentionally alias **all three keys** (`train`, `val`, `test`) to that dataset's test directory; they are evaluation wrappers, not the training split definition (`data/yolo/dut_anti_uav.yaml:2–4`; `data/yolo/coco.yaml:2–4`; `data/yolo/openimages.yaml:2–4`; `data/yolo/detfly.yaml:2–4`; `scripts/make_data_yaml.py:38–42`). Det-Fly is excluded from the merged training/validation lists (`scripts/make_data_yaml.py:15,32–42`).

Only **1,000 local Det-Fly test images, all sequence prefix `010`**, are represented; the saved evaluation is tagged `detfly_partial` (`E:419–428`; `experiments/eval_detfly_partial.json:2–6,17`). The acquisition/subsampling procedure that produced this partial set and coverage of the complete dataset are **NOT FOUND** [Q6].

“Held out” above is established for this project's detector fitting/checkpoint selection. A cross-source image/near-duplicate audit, DUT sequence-level independence audit, and proof that these test images were unseen in upstream pretraining or model selection are **NOT FOUND** [Q7]. Also, DUT test later supplies detector-noise estimates to the simulator; it is not held out from that downstream calibration (`detector/measure_noise.py:27,59–91,113–119`; `sim/noise.py:48,54–67`).

## 6. Training run

The training script contains this full-run invocation, which matches the final saved arguments (`detector/train.py:3–5`; `A:data,model,imgsz,epochs,batch,workers,time,patience,name`). The verbatim historical shell invocation is **NOT FOUND** [Q11].

```bash
python detector/train.py --data data/yolo/drone4_bal.yaml --model yolo26s.pt --imgsz 1280 \
    --epochs 150 --batch 8 --workers 12 --time 7 --patience 30 --name drone4_bal
```

| Item | Result | Source |
|---|---|---|
| Final run | `runs/detect/drone4_bal`; requested 150 epochs, 7-hour budget, patience 30, `resume=false`. | `A:save_dir,epochs,time,patience,resume`; `L:2,49–50` |
| Epochs completed | **106**, CSV epochs 1–106. Final progress is `106/124` because time mode changed the horizon. | `C:2–107`; `L:683,690` |
| Best epoch | **76**. | `L:687`; `C:77` |
| Selection metric | Validation **mean box AP over IoU 0.50–0.95**, across classes. Installed fitness weights are `[0,0,0,1]` for `[P,R,mAP50,mAP50-95]`; checkpoint `fitness=0.64647`, equal to its `metrics/mAP50-95(B)`. | `U/utils/metrics.py:1001,1015–1018,1214–1216`; `U/engine/trainer.py:851–852,961–964`; `E:121–129` |
| Best epoch CSV metrics | `P=0.90775`, `R=0.79662`, `mAP50=0.8609`, `mAP50-95=0.64647`. | `C:77` |
| Early stopping | Explicit log: no improvement for 30 epochs, best at 76; stopped at 106. The 7-hour budget was not exhausted. | `L:687–690`; `A:time` |
| Elapsed time | Log: **6.021 hours**. Final CSV cumulative `time=21676.4` seconds (6 h 1 min 16.4 s, arithmetic conversion). CSV time is measured from `train_time_start`. | `L:690`; `C:107`; `U/engine/trainer.py:1034–1042` |
| End-of-run validation | Reloaded `best.pt`; combined validation 4,316 images / 5,364 instances. Rounded output `P=0.909`, `R=0.797`, `mAP50=0.861`, `mAP50-95=0.646`. Keep this distinct from the CSV's epoch-76 row. | `L:694–702` |
| Allocation warnings | The log contains 106 CUDA allocation/OOM warnings, including the last epoch, but completes training and final validation. | `E:489–493`; `L:684–704` |

Total end-to-end project time including download, conversion, setup, aborted training, export, and evaluation is **NOT FOUND** [Q8]; the logged training duration does not measure that scope.

Other artifacts are separate runs: `runs/detect/smoke` requests one epoch on `_smoke/smoke.yaml` (`runs/detect/smoke/args.yaml`, keys `data,epochs,name`). `runs/logs/train_drone4_bal_aborted_patience15.log:2` records an earlier request for 100 epochs and patience 15; it ends during epoch-5 validation (`:78–80`). Do not combine that attempt's settings or elapsed work with the final run. Its exact termination cause is **NOT FOUND** [Q9].

## 7. Hardware and software

| Item | Evidence-backed value | Source |
|---|---|---|
| Training GPU | NVIDIA GeForce RTX 3080; CUDA device 0; reported memory 11,911 MiB. | `L:1`; `A:device` |
| PyTorch | `2.14.1+cu130` during training/evaluation; requirements pin `torch==2.14.1`. | `L:1`; `runs/logs/eval_all.log:1`; `requirements.txt:5` |
| CUDA build | Local PyTorch build reports `cuda='13.0'`, consistent with logged `+cu130`. This is the PyTorch build's CUDA version. | `.venv/lib/python3.12/site-packages/torch/version.py:4–6`; `L:1` |
| NVIDIA driver / system CUDA toolkit | **NOT FOUND** [Q10]. The requirements comment is not a driver or toolkit version record. | Checked `runs/logs/` and `requirements.txt:1–2` |
| Ultralytics | `8.4.215` in the final run; unpinned in requirements. | `L:1,695`; `requirements.txt:7`; `U/__init__.py:3` |
| Python | `3.12.3`. | `L:1`; `.venv/pyvenv.cfg:3` |
| CPU reported during export | 12th Gen Intel Core i9-12900K. | `runs/logs/eval_rest2.log:134` |
| Complete reproducible environment | **NOT FOUND** [Q1]: requirements leave packages other than torch unpinned; no historical lockfile/package freeze/source snapshot found. Current local package source is available and was read. | `requirements.txt:5–20`; current source fingerprints `E:478–486` |

## 8. Evaluation

The saved pipeline invokes these commands (`detector/run_eval_all.sh:7,11–13`):

```bash
python detector/eval.py --weights runs/detect/drone4_bal/weights/best.pt --data data/yolo/dut_anti_uav.yaml --tag dut_test
python detector/eval.py --weights runs/detect/drone4_bal/weights/best.pt --data data/yolo/drone4_bal.yaml --tag all4_test
python detector/eval.py --weights runs/detect/drone4_bal/weights/best.pt --data data/yolo/detfly.yaml --tag detfly_partial
```

The script calls `YOLO(weights).val(...)` with default `split=test`, `imgsz=1280`, `batch=8`, `conf=0.001`, `device=0`, `plots=True`, `exist_ok=True`; it does not pass `iou` (`detector/eval.py:22–33`). Installed validator code resolves omitted/null `iou` to **0.7** (`U/engine/validator.py:143–146`). These are **code-resolved settings**. The output JSON confirms weights/data/split/imgsz but does not serialize confidence, IoU, batch, or precision; a verbatim historical shell invocation and independent serialized runtime evaluation arguments are **NOT FOUND** [Q11] (`detector/eval.py:40–46`; `experiments/eval_all4_test.json:2–6`).

| Saved test result | Images / loader instances | mAP50 | mAP50–95 | Source |
|---|---|---:|---:|---|
| DUT test | 2,200 / 2,245 | 0.9791816086744017 | 0.7490165422829068 | `runs/logs/eval_all.log:7–8`; `experiments/eval_dut_test.json:7–8` |
| Combined four-class test | 3,734 / 4,382 | 0.8608131353848335 | 0.6602235416663433 | `runs/logs/eval_all.log:50–54`; `experiments/eval_all4_test.json:7–8` |
| Partial Det-Fly test | 1,000 / 1,000 | 0.3036273349803137 | 0.11916425568779325 | `E:340–348`; `experiments/eval_detfly_partial.json:7–8` |

The AP matching grid is `0.50…0.95` in ten steps; it is distinct from NMS IoU (`U/models/yolo/detect/val.py:59,173–183`). Saved P/R values are selected at **one confidence index maximizing the smoothed mean F1 across classes**, not independently optimized per-class confidence and not necessarily confidence 0.25. The script's docstring says “per class,” but the installed implementation uses `smooth(f1_curve.mean(0), 0.1).argmax()` (`detector/eval.py:8,37–42`; `U/utils/metrics.py:890–894`). The numeric F1-optimal confidence thresholds used for these reports are **NOT FOUND** [Q12].

Validation-set plots come from the final training validation of `best.pt` on **`drone4_bal.yaml`, `split=val`**, with `plots=true`, `imgsz=1280`, saved `conf=null`, `iou=null`, `max_det=300`. The installed code resolves those thresholds to **conf 0.001 / NMS IoU 0.7**; training validation uses batch `2 × 8 = 16` (`A:split,plots,imgsz,conf,iou,max_det,batch`; `L:694–704`; `U/engine/trainer.py:319–324,1053–1066`; `U/engine/validator.py:143–146`). The plots include `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, and confusion matrices; `results.png` plots the epoch CSV (`U/utils/metrics.py:884–888`; `U/engine/trainer.py:1044–1046`; `L:704`).

The presentation's `PR_curve.png`, `confusion_matrix.png`, and `results.png` are byte-identical to `runs/detect/drone4_bal/BoxPR_curve.png`, `confusion_matrix.png`, and `results.png`, respectively. In particular, the presentation confusion matrix copy is the raw-count file, not `confusion_matrix_normalized.png` (`E:430–448`). These are validation/training plots, not the test plots under `runs/val/{dut_test,all4_test,detfly_partial}` (`runs/logs/eval_all.log:10,56,113`).

The separate **confidence 0.25 confusion matrix** is generated by:

```bash
python -I detector/confusion_matrix_conf.py
```

Defaults select `best.pt` and `drone4_bal.yaml`, then explicitly call `model.val(data=..., split="val", imgsz=1280, batch=8, conf=0.25, iou=0.5, plots=True)` (`detector/confusion_matrix_conf.py:3,80–92`). Saved settings confirm `conf=0.25`, `nms_iou=0.5`, `match_iou=0.5`, and validation split (`presentation_assets/confusion_matrix_conf025.json:47–57`). The matrix is `[predicted][true]`, with background as the final row/column; normalized plotting divides by true-class column totals (`presentation_assets/confusion_matrix_conf025.json:2–9`; `detector/confusion_matrix_conf.py:37–44`). Ground-truth matching is class-agnostic; the local implementation uses strict `IoU > 0.5` and `confidence > 0.25` at its filtering step (`U/utils/metrics.py:457–470`). Outputs are `presentation_assets/confusion_matrix_conf025.png`, `confusion_matrix_conf025_counts.png`, and `confusion_matrix_conf025.json` (`detector/confusion_matrix_conf.py:100–109`). Its mAP is not used for the low-confidence test metrics (`presentation_assets/confusion_matrix_conf025.json:59`).

NMS behavior is code-backed: `nms=null` does not satisfy `nms is False`, so the validator sets `end2end=False` and passes the configured IoU into NMS. Do not characterize these evaluations as NMS-free solely because the architecture is YOLO26 (`A:nms`; `U/engine/validator.py:237–238,253–264`; `U/models/yolo/detect/val.py:173–183`).

Other test-derived quantities use separate scripts/settings:

- False drone alarms: `python detector/false_positives.py --weights runs/detect/drone4_bal/weights/best.pt`; COCO/Open Images test images, `imgsz=1280`, prediction confidence `0.25`, reports at `0.25` and `0.5`, object-overlap attribution IoU `0.1`. Prediction NMS IoU is not explicitly passed (`detector/run_eval_all.sh:16`; `detector/false_positives.py:43–75`).
- Detector-noise calibration: `python detector/measure_noise.py --weights runs/detect/drone4_bal/weights/best.pt`; DUT test, `imgsz=1280`, confidence `0.25`, GT-match IoU `0.3`; that `0.3` is a matching threshold, not the prediction NMS IoU, which is not explicitly passed (`detector/run_eval_all.sh:19`; `detector/measure_noise.py:27,45–50,64–83`).

For both prediction scripts, the installed predictor resolves omitted NMS IoU to **0.5** (`U/engine/predictor.py:140,159–160`; `U/utils/__init__.py:977`). Noise-calibration confidence and matching IoU are independently saved in `experiments/detector_noise.json:3–5`; prediction NMS IoU is not serialized there.

The earlier full evaluation log terminates in a false-positive prediction OOM **after all three AP result JSONs were written** (`runs/logs/eval_all.log:37,104,140–198`). Later logs record completion of the remaining evaluation/export steps (`runs/logs/eval_rest2.log:10,132,141,157`). This failure does not mean the saved test AP runs failed.

## 9. Open questions

The concrete gaps are provenance/reproducibility, exact historical invocation records, partial Det-Fly coverage, and independence of the evaluation data. The split definitions and reported metrics above are supported; stronger claims require the missing evidence below.

Multi-seed repeat runs, confidence intervals for detector metrics, and a documented detector hyperparameter/model-search history are **NOT FOUND** [Q13]. The retained detector runs are the smoke test and final run, plus the earlier aborted-run log (`runs/detect/smoke/args.yaml`, key `name`; `A:name`; `runs/logs/train_drone4_bal_aborted_patience15.log:2`). No statistical error bars should be inferred from the single final CSV.

Every value marked **NOT FOUND**:

1. **Q1 — Historical complete environment lock/package freeze and immutable training-time source snapshot.** Current version strings and local source exist; the archived full environment does not.
2. **Q2 — Initialization checkpoint's original download URL, preceding `yolo26s-objv1-150.pt` artifact, complete upstream training history, and exact upstream dataset snapshot.** The local `yolo26s.pt` metadata does identify COCO.
3. **Q3 — Unstripped final optimizer tensors and per-step LR/momentum traces.** Epoch-level LRs remain in the CSV.
4. **Q4 — Historical activation/version of optional Albumentations transforms.** Do not count their optional defaults as confirmed active augmentations.
5. **Q5 — Historical dataset converter/downloader command lines, actual cap overrides, and rejection/download-failure summaries.** Current scripts and resulting counts exist.
6. **Q6 — Acquisition/subsampling procedure and full-dataset coverage for the partial Det-Fly test set.** Only the local sequence-010 subset is established.
7. **Q7 — Cross-source image/near-duplicate audit, DUT sequence-independence audit, and proof of no upstream pretraining/model-selection exposure to project test images.**
8. **Q8 — Total wall-clock time for the complete project pipeline, including data preparation, setup, aborted work, exports, and evaluation.** Final training elapsed time is recorded.
9. **Q9 — Exact termination cause of the earlier patience-15 training attempt.**
10. **Q10 — Training-time NVIDIA driver version and system CUDA toolkit version.** PyTorch's CUDA build version is recorded separately.
11. **Q11 — Verbatim historical training/evaluation shell invocations and independent serialized runtime evaluation arguments.** Saved scripts/settings and output associations exist; test JSONs omit confidence/IoU/batch/precision.
12. **Q12 — Numeric F1-optimal confidence thresholds underlying the saved test P/R values.** Their selection algorithm is available.
13. **Q13 — Detector multi-seed repeats, metric confidence intervals, and documented model/hyperparameter-search history.**
