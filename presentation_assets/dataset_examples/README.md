# Perseus dataset examples

Generated using Pillow on CPU only. No model was loaded, no inference or training was run, and no GPU was used. Datasets, labels and existing runs were opened only for reading. All repository writes for this task are inside this directory.

## Selection and rendering

Test-image selection seed: **20261120**, using Python `random.Random`. For each dataset, collect sorted test JPEGs with exactly one relevant target (drone for DUT/Det-Fly, bird or airplane for COCO, helicopter for Open Images). Sort by target normalized bounding-box area, retain the middle 50% by rank, sort that pool by path, and sample eight without replacement. The RNG is initialized separately with the same seed for each dataset. Visually review those candidates and choose a clear, ordinary scene; the final choice is manual, not purely random. This avoids size extremes but does not establish a statistical difficulty ranking. No detector scores or success/failure results were used. Candidate lists and final indices are saved in `manifest.json`.

Every box in each selected converted YOLO label file is drawn in green with its class name. These are ground-truth annotations for the project’s four-class subset, not predictions or all categories from the original datasets. All four selected files have one box. Images retain aspect ratio, with no upscaling and a maximum width of 1600 px. DUT additionally has a 400 px zoom inset from the original image, with the box and class name repeated.

## dut_anti_uav

- Output: [dut_anti_uav_example.jpg](dut_anti_uav_example.jpg)
- Exact image source: `/home/soham/github/drone-range-ekf/data/yolo/dut_anti_uav/images/test/00887.jpg`
- Exact annotation source: `/home/soham/github/drone-range-ekf/data/yolo/dut_anti_uav/labels/test/00887.txt`
- Split: **test**. Seed: **20261120**.
- Dimensions: 1920 × 1080 → 1600 × 900.
- Why selected: Small drone above a treeline with campus context; visible in the inset, with a moderate annotated area rather than an extreme size.
- License, as recorded in [DATA.md](../../DATA.md): The project GitHub repository is Apache-2.0; DATA.md states that its README gives no separate data license. Cite Zhao et al., “Vision-based Anti-UAV Detection and Tracking,” IEEE T-ITS 2022 (arXiv:2205.10851). Separate dataset license: NOT FOUND in DATA.md.

## coco

- Output: [coco_example.jpg](coco_example.jpg)
- Exact image source: `/home/soham/github/drone-range-ekf/data/yolo/coco/images/test/000000284279.jpg`
- Exact annotation source: `/home/soham/github/drone-range-ekf/data/yolo/coco/labels/test/000000284279.txt`
- Split: **test**. Seed: **20261120**.
- Dimensions: 640 × 512 → 640 × 512.
- Why selected: One recognizable bird beside a feeder, with ordinary background detail and a moderate annotated area.
- License, as recorded in [DATA.md](../../DATA.md): Annotations: CC BY 4.0 (COCO Consortium). Images retain their individual copyrights and must follow Flickr Terms of Use; the consortium does not own them. This is not a blanket CC BY license for the photographs. Individual image license/photographer attribution was not checked.

The local test split originates from COCO **val2017**, as documented in DATA.md; it is not the official unlabeled COCO test split.

## openimages

- Output: [openimages_example.jpg](openimages_example.jpg)
- Exact image source: `/home/soham/github/drone-range-ekf/data/yolo/openimages/images/test/4bc6d10752652309.jpg`
- Exact annotation source: `/home/soham/github/drone-range-ekf/data/yolo/openimages/labels/test/4bc6d10752652309.txt`
- Split: **test**. Seed: **20261120**.
- Dimensions: 1024 × 683 → 1024 × 683.
- Why selected: One helicopter in side view with its rotor and skids visible; moderate annotated area within the helicopter subset.
- License, as recorded in [DATA.md](../../DATA.md): Annotations: CC BY 4.0 (Google LLC). Images are listed as CC BY 2.0, with the dataset disclaimer that individual image license status must be verified. Individual image license/photographer attribution was not checked.

## detfly

- Output: [detfly_example.jpg](detfly_example.jpg)
- Exact image source: `/home/soham/github/drone-range-ekf/data/yolo/detfly/images/test/010_0100612.jpg`
- Exact annotation source: `/home/soham/github/drone-range-ekf/data/yolo/detfly/labels/test/010_0100612.txt`
- Split: **test**. Seed: **20261120**.
- Dimensions: 3840 × 2160 → 1600 × 900.
- Why selected: Drone above textured agricultural ground, illustrating the requested ground-background scenario. Selected for scene content, not a measured model failure.
- License, as recorded in [DATA.md](../../DATA.md): Project GitHub repository: MIT; the downloaded Hugging Face subset card also says MIT, as recorded in DATA.md. Cite Zheng et al., “Air-to-Air Visual Detection of Micro-UAVs: An Experimental Evaluation of Deep Learning,” RA-L 2021.

This is the available provisional **1,000-image sequence-010 subset**, held out from training. Full 13,271-image Det-Fly dataset: **NOT FOUND locally**, according to DATA.md. This example is not claimed to be a confirmed detector miss.

## Real training mosaic

- Output: [mosaic_example.jpg](mosaic_example.jpg)
- Exact source: `/home/soham/github/drone-range-ekf/runs/detect/drone4_bal/train_batch1.jpg`
- Split: **train**; actual Ultralytics training-batch visualization, containing augmented mosaic samples.
- Provenance: **copied from the real training run, not rebuilt**.
- Run configuration: `/home/soham/github/drone-range-ekf/runs/detect/drone4_bal/args.yaml`; recorded seed **0**, `mosaic: 1.0`, `deterministic: false`. No new seed or augmentation was applied to the copy.
- Selection: visually compared `train_batch0.jpg`, `train_batch1.jpg`, and `train_batch2.jpg`. Batch 1 has especially readable large helicopter and airplane boxes alongside the small drone boxes; batch 0 has more crowded labels and batch 2 is dominated by small drones. Some training crops and small boxes remain, as in the original.
- The original numeric labels are retained: **0 drone, 1 bird, 2 airplane, 3 helicopter**.
- Kept at its original **1920 × 1920** resolution, byte-for-byte, to preserve the requested real-run copy and readability. The 1600 px width cap is applied to the four Part 1 dataset examples.
- The mixed training imagery is subject to the DUT, COCO and Open Images license notes above. Det-Fly is held out and is not training data.

## Verification and reproduction

All five JPEGs were opened successfully. The four rendered examples were visually inspected for boxes, labels and scene content. SHA-256 hashes of the selected source images and label files were checked before and after rendering and remained identical. The copied mosaic SHA-256 matches the original: `cf29613c7225fa06a30101a061377eb508896d40c92752f9dbf7b9ef2227da9a`.

Render the same selected examples and copy the same mosaic from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python presentation_assets/dataset_examples/generate.py
```

This writes the five JPEGs and `manifest.json` only in this directory. It does not regenerate this README. The script uses Pillow and the system DejaVu Sans font; it does not import PyTorch or Ultralytics.

All requested example images, annotation files, run arguments and saved training mosaics were found. License limitations and the unavailable full Det-Fly set are noted above.
