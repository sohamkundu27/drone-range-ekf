# Datasets

Classes (YOLO ids): `0 drone`, `1 bird`, `2 airplane`, `3 helicopter`.
Raw downloads live in `data/raw/<dataset>/`; converted data in `data/yolo/<dataset>/{images,labels}/{train,val,test}`.
Images are hard-linked from `raw/` into `yolo/`, so the converted copy uses no extra disk.
`data/` is git-ignored. This file and the scripts in `scripts/` are the record of how the data was built.

Status as of 2026-11-17:

| dataset | role | status | source |
|---|---|---|---|
| DUT Anti-UAV (detection) | drone train/val/test | **downloaded, converted** | official GitHub → Google Drive |
| Drone-vs-Bird (WOSDETC) | drone train/val/test | **annotations only; videos need a signed agreement (manual)** | official GitHub + email request |
| Det-Fly | **held-out cross-dataset test only** | **partial: 1,000 of 13,271 images** (full set needs a manual OneDrive download) | official OneDrive (manual) / Hugging Face copy |
| COCO 2017 (bird, airplane) | hard negatives | **downloaded, converted** | official cocodataset.org |
| Open Images V7 (helicopter, bird, fixed-wing aircraft) | hard negatives | **downloaded, converted** | official Google/CVDF bucket |

---

## 1. DUT Anti-UAV — detection subset

- Project page: https://github.com/wangdongdut/DUT-Anti-UAV
- Paper: Zhao et al., "Vision-based Anti-UAV Detection and Tracking", IEEE T-ITS 2022 (arXiv:2205.10851).
- **License:** the GitHub repo is Apache-2.0. The README states no separate data license, so cite the paper.
- Downloaded from the Google Drive links in the official README (2026-11-17), with `gdown`:
  - train `1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj` (train.zip, 744,616,230 B)
  - val `1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n` (val.zip, 372,283,691 B)
  - test `1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0` (test.zip, 271,153,425 B)
- Format: Pascal VOC XML, single class `UAV`. 10,000 images (5200/2600/2200), 10,109 boxes (one has zero area and is dropped).
- Conversion: `scripts/convert_dut.py`. The official split is kept.

```bash
mkdir -p data/raw/dut_anti_uav/zips && cd data/raw/dut_anti_uav/zips
for id in 1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj 1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n 1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0; do gdown "$id"; done
cd .. && mkdir -p extracted && for z in zips/*.zip; do unzip -q -o "$z" -d extracted; done
```

## 2. Drone-vs-Bird (WOSDETC challenge) — MANUAL STEP REQUIRED

- Project page: https://github.com/wosdetc/challenge (challenge site: https://wosdetc2023.wordpress.com/drone-vs-bird-detection-challenge/)
- **License / access:** the videos are released only after you **sign a data usage agreement**
  (research use). Request them by emailing **wosdetc@googlegroups.com**.
  Read the agreement before using the data. I did not request it on your behalf.
  Re-uploads of these videos exist on third-party sites. I did **not** use them, because
  they would bypass the agreement.
- What is already here: the **public annotations** (77 files, one per video) from
  `https://github.com/wosdetc/challenge/tree/master/annotations`, in `data/raw/drone_vs_bird/annotations/`.
  They cover 106,485 frames and 119,259 boxes. **Only `drone` is annotated: birds are NOT labelled**,
  despite the challenge's name.
- Steps once you have access:
  1. Put the videos in `data/raw/drone_vs_bird/videos/`, named like the annotation files
     (e.g. `gopro_001.mp4` ↔ `annotations/gopro_001.txt`). Check frame counts against the annotations; the converter warns on mismatch.
  2. `python -I scripts/convert_drone_vs_bird.py --stride 10`. This splits by video, about 70/15/15
     (56/12/9 videos), and keeps every 10th frame, giving about 10.7k frames.
  3. `python -I scripts/make_data_yaml.py`, then re-run `dataset_stats.py` and `sanity_check.py`.
- **Caveat:** these videos contain unlabelled birds. Training on them together with the `bird`
  class teaches the model that some birds are background. Options: (a) accept it, since
  drones are the priority; (b) use Drone-vs-Bird frames only for drone recall and evaluation;
  (c) pseudo-label birds with the trained model and review them by hand.

## 3. Det-Fly — held-out test set (never used for training) — MANUAL STEP FOR THE FULL SET

- Project page: https://github.com/Jake-WU/Det-Fly
- Paper: Zheng et al., "Air-to-Air Visual Detection of Micro-UAVs: An Experimental Evaluation of Deep Learning", RA-L 2021.
- **License:** the GitHub repo is MIT. Cite the paper.
- **Official download (manual, browser):** OneDrive folders from the README:
  - Annotations: https://westlakeu-my.sharepoint.com/:f:/g/personal/zhengye_westlake_edu_cn/Em9kWPCMYm1KpFHnhVOTn9UBQfQ25m0lH67xIcULuYXYHw?e=AWZfiw
  - JPEGImages: https://westlakeu-my.sharepoint.com/:f:/g/personal/zhengye_westlake_edu_cn/EqFYguroD9lEnVDTRkyoOJQBrHvndTQGa5f8EQurGRFUqQ?e=DXPQxN
  - (Baidu alternative: Annotations https://pan.baidu.com/s/1v30h53xu0PmgwnarAaFK7A code `4dcc`; JPEGImages https://pan.baidu.com/s/1A6D9j_PzcWLeYe9GQttypQ code `qjyt`)
  - Each folder has subfolders `010` (6,358 items) and `020` (6,913 items), 13,271 images in total.
    Scripted download does not work: the anonymous SharePoint API refuses folder listing
    (over the 5,000-item list-view threshold, and access is denied on the v2 API). Use the **Download** button
    in a browser on each folder (OneDrive zips it), then unzip so you get
    `data/raw/detfly/extracted/JPEGImages/{010,020}/*.jpg` and `data/raw/detfly/extracted/Annotations/{010,020}/*.xml`,
    and re-run `python -I scripts/convert_detfly.py`.
- **What is here now (provisional):** the Hugging Face copy https://huggingface.co/datasets/DestinyLove/Det-Fly
  (`Det-Fly/Det-Fly.zip`, 2,138,809,573 B, sha256 `6af21e53…f8f9d109` matches HF's LFS hash; card says MIT, links to the official repo).
  **It is incomplete:** it holds only `JPEGImages/010` with 1,000 images (`0100001`–`0101000`) and 11,358 of the 13,271 XMLs.
  Those 1,000 images, all matched to XMLs, are converted as a provisional `detfly/test`.
  Sequence `010` is only part of the dataset's scenes and views, so cross-dataset numbers
  on this subset are preliminary.
- Format: VOC XML, class `UAV`, 3840×2160 images. Conversion: `scripts/convert_detfly.py`
  (reads the real JPEG size and cross-checks it against the XML: 0 mismatches).

## 4. COCO 2017 — bird + airplane hard negatives

- https://cocodataset.org. **License** (from the official terms of use): annotations CC BY 4.0 (COCO Consortium);
  "The COCO Consortium does not own the copyright of the images. Use of the images must abide by the
  Flickr Terms of Use."
- Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip (252,907,541 B).
- Images are fetched individually from `http://images.cocodataset.org/{train2017,val2017}/<file>`,
  only for images containing bird (cat 16) or airplane (cat 5). Images where a bird or airplane is `iscrowd` are skipped.
- Splits: train2017 → train/val (deterministic 90/10 by image id); val2017 → test.
- Script: `scripts/download_coco_subset.py` (downloads and converts).

## 5. Open Images V7 — helicopter, bird, fixed-wing aircraft hard negatives

- https://storage.googleapis.com/openimages/web/index.html. **License** (from the official site):
  "The annotations are licensed by Google LLC under CC BY 4.0 license. The images are listed as having a
  CC BY 2.0 license. Note: while we tried to identify images that are licensed under a Creative Commons
  Attribution license, we make no representations or warranties regarding the license status of each image
  and you should verify the license for each image yourself."
- Annotation files (box annotations are unchanged from V6 to V7):
  - https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv (2.26 GB)
  - https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv
  - https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv
  - https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv
  - https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json
- Images: `https://open-images-dataset.s3.amazonaws.com/{train,validation,test}/<ImageID>.jpg` (official CVDF bucket).
- Mapping: `Bird` and its 17 subclasses (Eagle, Duck, Owl, Parrot, …) → bird. **`Fixed-wing aircraft`
  (`/m/0cmf2`; Open Images has no class named "Airplane")** → airplane. `Helicopter` → helicopter.
- Filters: drop images with any relevant `IsGroupOf` or `IsDepiction` box, and images containing generic
  `Aircraft` or `Rocket` boxes, which would otherwise become unlabelled objects.
- Caps: train has 3,000 images per class for bird and airplane, and all helicopter images (2,222). Val and test have up to 500 per class.
- Splits: OI train → train, OI validation → val, OI test → test.
- Script: `scripts/download_openimages_subset.py` (`--dry-run` prints counts only).

---

## Converted counts and box sizes

Generated by `python -I scripts/dataset_stats.py` (also saved to `figures/dataset_stats.md`):

| dataset | split | images | bg imgs | drone | bird | airplane | helicopter | med w px | med h px | med sqrt(wh) px | <16px | <32px | med @640 | med @1280 | typical res |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coco | train | 5324 | 0 | 0 | 6215 | 4543 | 0 | 89.0 | 63.1 | 75.8 | 16% | 29% | 77.5 | 155.0 | 640x480 |
| coco | val | 604 | 0 | 0 | 673 | 505 | 0 | 104.9 | 64.1 | 84.0 | 13% | 27% | 86.1 | 172.1 | 640x480 |
| coco | test | 209 | 0 | 0 | 233 | 143 | 0 | 89.8 | 68.3 | 81.4 | 16% | 28% | 81.4 | 162.7 | 640x427 |
| coco | all | 6137 | 0 | 0 | 7121 | 5191 | 0 | 90.0 | 63.3 | 76.7 | 16% | 29% | 78.4 | 156.8 | 640x480 |
| detfly | test | 1000 | 0 | 1000 | 0 | 0 | 0 | 138.0 | 109.0 | 122.0 | 0% | 0% | 20.3 | 40.7 | 3840x2160 |
| detfly | all | 1000 | 0 | 1000 | 0 | 0 | 0 | 138.0 | 109.0 | 122.0 | 0% | 0% | 20.3 | 40.7 | 3840x2160 |
| dut_anti_uav | train | 5200 | 3 | 5243 | 0 | 0 | 0 | 43.0 | 23.0 | 31.1 | 7% | 52% | 10.4 | 20.9 | 1920x1080 |
| dut_anti_uav | val | 2600 | 0 | 2620 | 0 | 0 | 0 | 42.0 | 23.0 | 30.8 | 7% | 53% | 10.3 | 20.6 | 1920x1080 |
| dut_anti_uav | test | 2200 | 0 | 2245 | 0 | 0 | 0 | 55.0 | 30.0 | 40.9 | 4% | 38% | 14.5 | 29.0 | 1920x1080 |
| dut_anti_uav | all | 10000 | 3 | 10108 | 0 | 0 | 0 | 44.0 | 24.0 | 32.3 | 6% | 49% | 10.8 | 21.7 | 1920x1080 |
| openimages | train | 8065 | 0 | 0 | 7744 | 5280 | 2748 | 252.2 | 174.4 | 208.9 | 2% | 8% | 130.5 | 261.1 | 1024x683 |
| openimages | val | 1112 | 0 | 0 | 824 | 611 | 134 | 584.3 | 335.5 | 475.0 | 0% | 1% | 296.8 | 593.7 | 1024x683 |
| openimages | test | 1325 | 0 | 0 | 783 | 592 | 391 | 681.8 | 369.6 | 509.2 | 1% | 2% | 318.4 | 636.7 | 1024x683 |
| openimages | all | 10502 | 0 | 0 | 9351 | 6483 | 3273 | 310.8 | 211.2 | 266.4 | 2% | 7% | 166.5 | 332.9 | 1024x683 |

Box size columns are in ORIGINAL image pixels. `med @640` and `med @1280` give the median sqrt(w·h)
after letterboxing the long side to 640 or 1280, which is what the detector sees.
