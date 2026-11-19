presentation_assets/  -  images for the slides (about 2 MB in total)

Detector: YOLO26s, 4 classes (drone, bird, airplane, helicopter), imgsz 1280,
run runs/detect/drone4_bal (stopped early at epoch 106, best epoch 76).


TRAINING PLOTS (copied unchanged from runs/detect/drone4_bal/, made by Ultralytics)

results.png
    Training curves per epoch: train/val box, class and L1 losses (YOLO26 has no DFL loss),
    and validation precision, recall, mAP50 and mAP50-95. The validation losses and mAP
    flatten after about epoch 60-75 while the training losses keep falling; training
    stopped early at epoch 106 (patience 30) and kept the best epoch, 76.

PR_curve.png
    Precision-recall curve per class, for the best weights on the VALIDATION split
    (COCO + DUT Anti-UAV + Open Images val). Copied from BoxPR_curve.png.

confusion_matrix.png
    Confusion matrix (counts) for the best weights on the same VALIDATION split.

    Note: the numbers in these three plots are VALIDATION numbers. The tables in README.md,
    experiments/detector_results.md and the slides are TEST-split numbers (e.g. DUT test
    drone mAP50 0.979), so the plots and the tables will not show identical values.


EXAMPLE DETECTIONS (drawn by detector/presentation_examples.py; rerun with
    python -I detector/presentation_examples.py
 green = ground-truth box, red = detector box with its confidence. Frames are resized to
 1600 px wide. Small drones get a "zoom" inset, top right, cut from the full-resolution frame.)

dut_good_small.jpg
    DUT Anti-UAV test image 02151.jpg (1920x1080). Small drone, 28 px wide, in front of a
    building. Detected with conf 0.74, IoU 0.89 with the ground truth.

dut_good_medium.jpg
    DUT Anti-UAV test image 01727.jpg (1920x1080). Drone 56 px wide against sky above a
    rooftop. Conf 0.84, IoU 0.91.

dut_good_large.jpg
    DUT Anti-UAV test image 01049.jpg (1280x720). Large close hexacopter, 289 px wide.
    Conf 0.90, IoU 0.95.

    How these were chosen: from a fixed random sample (seed 0) of 300 DUT test images, keep
    frames with one drone, one predicted drone box, conf >= 0.7 and IoU >= 0.7 (256 frames
    qualified), sort them by drone width and take the 10th, 50th and 90th percentile.
    They are picked to show GOOD detections; they are not representative of failures.

detfly_missed_1.jpg
    Det-Fly (PRELIMINARY: partial set, 1,000 images of sequence 010) frame 010_0100136.jpg
    (3840x2160). Drone seen from above against grass next to a running track, 186 px wide
    (62 px at the 1280 detector input, so size is not the problem). NOT detected. The
    detector instead put a false drone box (conf 0.35) on two people on the track (red).

detfly_missed_2.jpg
    Det-Fly (PRELIMINARY) frame 010_0100900.jpg (3840x2160). Drone seen from above against
    farmland, 87 px wide (29 px at 1280 input). NOT detected, and no other drone boxes.

    How these were chosen: from a fixed random sample (seed 0) of 300 Det-Fly images, a miss
    = no predicted drone box at conf >= 0.25 overlapping the ground truth (IoU >= 0.1).
    200 of the 300 sampled frames were misses (Det-Fly recall is 0.285 overall). The two
    shown are the misses at 1/6 and 5/6 of that list, so they come from different parts
    of the sequence.


SOURCES AND LICENSES (see DATA.md)

DUT Anti-UAV: Zhao et al., "Vision-based Anti-UAV Detection and Tracking", IEEE T-ITS 2022
    (arXiv:2205.10851). Repository license Apache-2.0; no separate data license stated.
Det-Fly: Zheng et al., "Air-to-Air Visual Detection of Micro-UAVs: An Experimental
    Evaluation of Deep Learning", RA-L 2021. Repository license MIT.
Cite both papers on any slide that shows these images.
