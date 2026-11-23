"""Evaluate the continued YOLO26m (runs/m_detfly_cont/weights/best.pt) with exactly the jobs, scripts and
splits used for the n/s/m table in DETFLY_RETRAIN_RESULTS.md (scripts/run_retrain_queue.py eval_jobs):

  Det-Fly 020 and 010-test   detector/new_eval_detfly.py   (drone only, P/R at conf 0.25, block-bootstrap CI)
  all4_test, dut_test        detector/eval.py              (data/yolo/drone4_bal.yaml, data/yolo/dut_anti_uav.yaml)
  false drones               detector/false_positives.py

    .venv/bin/python -I detector/cont_eval_m.py

best.pt is the epoch with the highest VALIDATION fitness (chosen by Ultralytics during training); nothing
here looks at test results to pick a checkpoint. Each test job runs once: it is skipped if its output exists.
Outputs: detfly_retrain/eval/cont_m_<set>.json, logs/eval_cont_m_<set>.log. GPU jobs hold /tmp/gpu.lock.
"""
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
W = str(ROOT / "runs" / "m_detfly_cont" / "weights" / "best.pt")
TAG = "cont_m"
EV = ROOT / "detfly_retrain" / "eval"

JOBS = [
    ("test020", [PY, "detector/new_eval_detfly.py", "--weights", W, "--model-tag", TAG, "--set", "test020"]),
    ("test010", [PY, "detector/new_eval_detfly.py", "--weights", W, "--model-tag", TAG, "--set", "test010"]),
    ("all4_test", [PY, "detector/eval.py", "--weights", W, "--data", "data/yolo/drone4_bal.yaml",
                   "--tag", f"{TAG}_all4_test", "--out", str(EV / f"{TAG}_all4_test.json"),
                   "--project", str(ROOT / "runs" / "new_val")]),
    ("dut_test", [PY, "detector/eval.py", "--weights", W, "--data", "data/yolo/dut_anti_uav.yaml",
                  "--tag", f"{TAG}_dut_test", "--out", str(EV / f"{TAG}_dut_test.json"),
                  "--project", str(ROOT / "runs" / "new_val")]),
    ("false_positives", [PY, "detector/false_positives.py", "--weights", W,
                         "--out", str(EV / f"{TAG}_false_positives.json")]),
]


def main():
    failed = []
    for name, cmd in JOBS:
        out = EV / f"{TAG}_{name}.json"
        if out.exists():
            print("skip (already evaluated once):", out.relative_to(ROOT), flush=True)
            continue
        log = ROOT / "logs" / f"eval_{TAG}_{name}.log"
        shell = (f"flock /tmp/gpu.lock {' '.join(shlex.quote(c) for c in cmd)} 2>&1 | tee {shlex.quote(str(log))} "
                 f"> /dev/null; exit ${{PIPESTATUS[0]}}")
        print("RUN", name, flush=True)
        rc = subprocess.run(["bash", "-c", shell], cwd=ROOT).returncode
        if rc != 0 or not out.exists():
            failed.append(name)
            print(f"FAILED {name} (exit {rc}); see {log}", flush=True)
    if failed:
        sys.exit(f"failed: {failed}")
    print("all evaluations done")


if __name__ == "__main__":
    main()
