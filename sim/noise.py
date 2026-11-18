"""Synthetic detector measurement noise, fitted to the real detector where possible (stub).

Planned contents:
- Pixel-centroid noise: zero-mean Gaussian with sigma that grows as the box gets smaller.
  Sigma will be fitted to detector residuals measured on the DUT / Det-Fly test sets.
- Box-size noise: multiplicative (log-normal) noise on w_px / h_px. This dominates the
  range error, since range ~ 1 / box size.
- Missed detections: probability p_miss(box size) that rises for tiny targets.
- False positives / clutter: Poisson count per frame with uniform pixel location, plus
  "bird-like" confusers near the target to test gating.
- Latency and frame-drop jitter between the image timestamp and when the detection
  is delivered.
- Camera ego-pose noise (IMU/GNSS-like), so the filter does not get perfect ego-motion.
All functions take an explicit numpy Generator for reproducibility.
"""
