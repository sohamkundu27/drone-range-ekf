# drone-range-ekf

One-week project: **monocular drone detection + an EKF that tracks the drone's 3D position and
range from a moving camera**, evaluated in a Python simulation.

1. **Detector** (PyTorch / Ultralytics YOLO): 4 classes, `drone`, `bird`, `airplane`, `helicopter`.
   Birds and aircraft are hard negatives, so the tracker is not fed confusers.
   Trained on DUT Anti-UAV (+ Drone-vs-Bird once access is granted), COCO and Open Images.
   Evaluated in-domain and on **Det-Fly as a held-out cross-dataset test**.
2. **Simulation** (NumPy): a moving pinhole camera observes a drone along known trajectories.
   Detections are synthesised with noise fitted to the real detector (pixel jitter, box-size noise,
   misses, clutter).
3. **EKF** (NumPy): nearly-constant-velocity state `[p, v]` in the world frame. Measurements are pixel
   centroid + apparent box size, which is what makes range observable from one camera.
   Consistency is checked with NEES/NIS against chi-square bounds.

Status: environment, data pipeline and detector tooling are done and smoke-tested.
`sim/` and `ekf/` are documented stubs. No full training run has been started.

## Layout

```
data/                (git-ignored)
  raw/<dataset>/       original downloads (zips, CSVs, images)
  yolo/<dataset>/      YOLO format: images|labels/{train,val,test}; drone4.yaml, detfly.yaml, ...
scripts/             data download/conversion, stats, sanity plots (see DATA.md)
detector/            train.py, eval.py, export_onnx.py, latency.py
sim/                 camera.py, trajectories.py, noise.py   (stubs)
ekf/                 ekf.py, nees.py                         (stubs)
experiments/         experiment scripts and result JSONs
figures/             plots; figures/sanity/ = GT-box visualisations (git-ignored, regenerate)
runs/                Ultralytics training/val output (git-ignored)
DATA.md              sources, licenses, manual steps, counts
```

## Reproduce

All commands run from the repo root. Scripts that read downloaded data are run with `python -I`
(isolated mode, so nothing in the data directories can be imported).

### 1. Environment (tested: Ubuntu, Python 3.12, RTX 3080, NVIDIA driver 580 / CUDA 13.0)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True NVIDIA GeForce RTX 3080
```
On Windows, use `.venv\Scripts\activate`. Pick the CUDA wheel (`cu126`/`cu130`) that matches
`nvidia-smi`'s "CUDA Version".

### 2. Data (about 13 GB in data/raw; see DATA.md for licenses and manual steps)

```bash
# DUT Anti-UAV (Google Drive, official links)
mkdir -p data/raw/dut_anti_uav/zips && (cd data/raw/dut_anti_uav/zips && \
  for id in 1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj 1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n 1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0; do gdown "$id"; done)
mkdir -p data/raw/dut_anti_uav/extracted && for z in data/raw/dut_anti_uav/zips/*.zip; do unzip -q -o "$z" -d data/raw/dut_anti_uav/extracted; done
python -I scripts/convert_dut.py

# COCO bird/airplane (annotations zip, then per-image download)
mkdir -p data/raw/coco/zips && aria2c -x8 -d data/raw/coco/zips http://images.cocodataset.org/annotations/annotations_trainval2017.zip
python -I scripts/download_coco_subset.py

# Open Images V7 helicopter/bird/fixed-wing aircraft
mkdir -p data/raw/openimages/csv && cd data/raw/openimages/csv && \
  for u in https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv \
           https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv \
           https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv \
           https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv \
           https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json; do aria2c -x8 "$u"; done; cd -
python -I scripts/download_openimages_subset.py            # --dry-run to see counts only

# Det-Fly (held-out test). Full set = manual OneDrive download (DATA.md §3).
# Provisional partial copy (1,000 images) from Hugging Face:
mkdir -p data/raw/detfly/zips && aria2c -x8 -d data/raw/detfly/zips -o Det-Fly.zip \
  https://huggingface.co/datasets/DestinyLove/Det-Fly/resolve/main/Det-Fly/Det-Fly.zip
mkdir -p data/raw/detfly/extracted && unzip -q -o data/raw/detfly/zips/Det-Fly.zip -d data/raw/detfly/extracted
python -I scripts/convert_detfly.py

# Drone-vs-Bird: annotations are public; videos need a signed agreement (DATA.md §2)
mkdir -p data/raw/drone_vs_bird/annotations && curl -s https://api.github.com/repos/wosdetc/challenge/contents/annotations \
  | python3 -c "import sys,json;[print(x['download_url']) for x in json.load(sys.stdin)]" \
  | aria2c -j8 -d data/raw/drone_vs_bird/annotations -i -
python -I scripts/convert_drone_vs_bird.py --dry-run       # without --dry-run once videos are in data/raw/drone_vs_bird/videos/

# Data YAMLs (drone4.yaml = training union, detfly.yaml = held-out test, one per dataset)
python -I scripts/make_data_yaml.py
```

### 3. Checks

```bash
python -I scripts/sanity_check.py --n 8     # -> figures/sanity/<ds>_<split>_<k>.jpg and <ds>_crops.jpg
python -I scripts/dataset_stats.py          # -> figures/dataset_stats.{md,json}
python detector/train.py --smoke            # 1 epoch, 96/32-image subset, imgsz 1280, GPU
```

### 4. Detector

```bash
python detector/train.py --data data/yolo/drone4.yaml --model yolo26s.pt --imgsz 1280 --epochs 100 --batch 8
python detector/eval.py --weights runs/detect/drone4/weights/best.pt --data data/yolo/drone4.yaml    # in-domain test
python detector/eval.py --weights runs/detect/drone4/weights/best.pt --data data/yolo/detfly.yaml    # cross-dataset
python detector/export_onnx.py --weights runs/detect/drone4/weights/best.pt --imgsz 1280
python detector/latency.py --weights runs/detect/drone4/weights/best.pt --imgsz 1280
```

**Image size:** drones are small. The median DUT drone is about 32 px (sqrt(w·h)) in a 1920×1080 frame, which
becomes about **11 px at imgsz 640** and about 22 px at 1280, and 49% of DUT boxes are under 32 px.
Train at `--imgsz 1280`. For the 4K Det-Fly frames, consider SAHI-style tiling at inference
(Det-Fly's median drone of 122 px in 4K shrinks to about 20 px at 640). At 1280, batch 8, YOLO26s used
about 10.2 GB of the 3080's 12 GB in the smoke test. Drop to `--batch 6` if a full run hits OOM.

### 5. Simulation + EKF

To do: see the docstrings in `sim/*.py` and `ekf/*.py`.
