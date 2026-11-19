"""Check which execution provider ONNX Runtime really uses for the exported detector.

    python detector/check_onnx_providers.py --onnx runs/detect/drone4_bal/weights/best_fp32.onnx --out experiments/onnx_providers_fp32.json
    python detector/check_onnx_providers.py --onnx runs/detect/drone4_bal/weights/best_fp16.onnx --out experiments/onnx_providers_fp16.json

onnxruntime-gpu needs the CUDA 13 libraries (cuBLAS, cuDNN). In this venv they come from the
nvidia-* pip wheels that PyTorch installs, and they are only loaded into the process when
torch is imported. So:
  * import onnxruntime alone  -> CUDA provider fails to load -> silent fallback to CPU
  * import torch first        -> CUDA provider is active
detector/latency.py (through Ultralytics) imports torch first, so its ONNX numbers are GPU numbers.

Each case runs in a fresh subprocess so the two do not affect each other.
Writes experiments/onnx_providers.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The test program run in a subprocess: optionally import torch, build a session, time session.run.
PROGRAM = """
import sys, time, json
if sys.argv[2] == "1":
    import torch
import numpy as np, onnxruntime as ort
s = ort.InferenceSession(sys.argv[1], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
i = s.get_inputs()[0]
x = np.random.rand(*i.shape).astype(np.float16 if "16" in i.type else np.float32)
for _ in range(10):
    s.run(None, {i.name: x})
t = time.perf_counter()
for _ in range(30):
    s.run(None, {i.name: x})
print(json.dumps({"available": ort.get_available_providers(), "active": s.get_providers(),
                  "session_run_ms": (time.perf_counter() - t) / 30 * 1000}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "runs/detect/drone4_bal/weights/best_fp16.onnx"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "onnx_providers.json"))
    args = ap.parse_args()
    out = {"onnx": str(Path(args.onnx).relative_to(ROOT)) if Path(args.onnx).is_absolute() else args.onnx,
           "note": "session_run_ms = raw ONNX Runtime inference on a 1x3x1280x1280 input, including the "
                   "host-to-device copy; no pre/post-processing."}
    for torch_first in (False, True):
        r = subprocess.run([sys.executable, "-c", PROGRAM, args.onnx, "1" if torch_first else "0"],
                           capture_output=True, text=True, check=True)
        res = json.loads(r.stdout.strip().splitlines()[-1])
        out["torch_imported_first" if torch_first else "onnxruntime_alone"] = res
        print("torch imported first:" if torch_first else "onnxruntime alone:  ", res)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
