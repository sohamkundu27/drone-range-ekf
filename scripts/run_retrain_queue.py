"""Sequential, restart-safe GPU queue for the Det-Fly retrain experiment.

    tmux new-window -t perseus -n queue "cd <repo> && .venv/bin/python scripts/run_retrain_queue.py"

Stages (each idempotent: skipped when its output/marker exists; rerun after a crash to continue):
  split -> old_eval (old n/s/m: Det-Fly 020 + 010-test [+ prelim for s], DUT all4 + DUT-only, false drones)
  -> train_s, eval_new_s -> train_n, eval_new_n -> train_m, eval_new_m -> final (slide assets + sim).
Every GPU job runs under `flock /tmp/gpu.lock` and waits while another process holds the GPU.
All job output is tee'd to logs/. After every stage the reports are regenerated and pushed
(git pull --rebase --autostash; git add <explicit paths>; commit; push; never force).
"""
import csv
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
LOGS = ROOT / "logs"
MARK = ROOT / "runs" / "retrain_markers"
EV = ROOT / "detfly_retrain" / "eval"
NOTES = ROOT / "detfly_retrain" / "notes.json"
TZ = ZoneInfo("America/Chicago")
DEADLINE = dt.datetime(2026, 11, 22, 22, 0, tzinfo=TZ)
OLD_W = {"s": "runs/drone4_bal_backup.aVl2ZP/best.pt",
         "n": "model_size_benchmark/train/n_batch8/weights/best.pt",
         "m": "model_size_benchmark/train/m_batch4/weights/best.pt"}
BATCH = {"s": 8, "n": 8, "m": 4}
ORDER = ["s", "n", "m"]
OLD_CAP_H, EVAL_H, FINAL_H, SAFETY_H = 7.0, 0.75, 3.0, 1.5
STATE = {"stage": "starting", "detail": ""}


def now():
    return dt.datetime.now(TZ)


def log(msg):
    line = f"[{now():%a %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOGS / "queue.log", "a") as f:
        f.write(line + "\n")


def note(msg):
    notes = json.loads(NOTES.read_text()) if NOTES.exists() else []
    if msg not in [n["note"] for n in notes]:
        notes.append({"time": f"{now():%Y-%m-%d %H:%M %Z}", "note": msg})
        NOTES.parent.mkdir(parents=True, exist_ok=True)
        NOTES.write_text(json.dumps(notes, indent=1) + "\n")
    log("NOTE " + msg)


def done(name):
    return (MARK / f"{name}.done").exists()


def mark(name):
    MARK.mkdir(parents=True, exist_ok=True)
    (MARK / f"{name}.done").write_text(f"{now():%Y-%m-%d %H:%M:%S %Z}\n")


def gpu_busy():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        return [p for p in out if p.strip()]
    except Exception:
        return []


def run(cmd, logname, gpu=True, poll=None):
    """Run cmd (list) with output tee'd to logs/<logname>.log. Returns exit code."""
    if gpu:
        while (busy := gpu_busy()):
            STATE["detail"] = f"waiting: GPU held by pid {','.join(busy)}"
            write_status()
            log(f"GPU busy ({busy}); waiting 5 min")
            time.sleep(300)
    q = " ".join(shlex.quote(c) for c in cmd)
    shell = (f"{'flock /tmp/gpu.lock ' if gpu else ''}{q} 2>&1 | tee -a {LOGS / (logname + '.log')}; "
             f"exit ${{PIPESTATUS[0]}}")
    log(f"RUN {logname}: {q}")
    p = subprocess.Popen(["bash", "-c", shell], cwd=ROOT, stdout=subprocess.DEVNULL)
    last_push = time.time()
    while p.poll() is None:
        time.sleep(60)
        if poll:
            poll()
            if time.time() - last_push > 2 * 3600:
                push("STATUS: periodic progress update")
                last_push = time.time()
    log(f"EXIT {logname}: {p.returncode}")
    return p.returncode


# ---------------------------------------------------------------- git
def allowed_files():
    files = [ROOT / f for f in ("STATUS.md", "DETFLY_FULL_RESULTS.md", "DETFLY_RETRAIN_RESULTS.md", "detfly_split.json",
                                 "scripts/convert_detfly.py", "scripts/build_detfly_split.py",
                                 "scripts/run_retrain_queue.py", "detector/eval.py")]
    files += list((ROOT / "scripts").glob("new_*")) + list((ROOT / "detector").glob("new_*.py"))
    for d in ("detfly_retrain", "presentation_assets_v2", "experiments/new_results"):
        files += [p for p in (ROOT / d).rglob("*") if p.suffix in (".md", ".json", ".csv", ".png", ".jpg")]
    out = [p for p in files if p.is_file() and p.stat().st_size < 5 * 2**20]
    jpgs = [p for p in out if p.suffix == ".jpg"]
    if len(jpgs) > 5:
        note(f"more than 5 example jpgs found ({len(jpgs)}); only the first 5 (sorted) are committed")
        out = [p for p in out if p.suffix != ".jpg"] + sorted(jpgs)[:5]
    return [str(p.relative_to(ROOT)) for p in out]


def git(*a):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    with open(LOGS / "git.log", "a") as f:
        f.write(f"$ git {' '.join(a)}\n{r.stdout}{r.stderr}\n")
    return r.returncode


def push(msg):
    write_status()
    for attempt in range(3):
        lock = os.open("/tmp/git.lock", os.O_CREAT | os.O_RDWR)
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            git("add", "--", *allowed_files())
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
                git("commit", "-q", "-m", f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
            if git("pull", "--rebase", "--autostash", "origin", "main") != 0:
                git("rebase", "--abort")
                raise RuntimeError("pull --rebase failed")
            if git("push", "origin", "main") == 0:
                log(f"PUSHED: {msg}")
                return True
            raise RuntimeError("push failed")
        except Exception as e:
            log(f"push attempt {attempt + 1} failed: {e}")
            time.sleep(60)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.close(lock)
    note(f"git push failed 3 times at '{msg}' (see logs/git.log); results are committed locally")
    return False


# ---------------------------------------------------------------- status
def train_progress(size):
    run = ROOT / "runs" / f"{size}_detfly_new"
    csvp = run / "results.csv"
    if not csvp.exists():
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(csvp))]
    if not rows:
        return None
    el = float(rows[-1]["time"])
    cap = json.loads((MARK / f"cap_{size}.json").read_text())["time_h"] if (MARK / f"cap_{size}.json").exists() else None
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    per = el / len(rows)
    eta = (now() + dt.timedelta(hours=max(0.0, cap - el / 3600))) if cap else None
    return (f"{size}: epoch {rows[-1]['epoch']} done, {el / 3600:.2f} h elapsed, {per / 60:.1f} min/epoch, "
            f"time cap {cap} h, latest val mAP50 {float(rows[-1]['metrics/mAP50(B)']):.3f}, best val mAP50-95 "
            f"{float(best['metrics/mAP50-95(B)']):.3f} (epoch {best['epoch']}), "
            f"ETA (time cap, earlier if early stopping) {eta:%a %H:%M %Z}" if eta else "")


def write_status(final=False):
    marks = sorted(p.stem for p in MARK.glob("*.done")) if MARK.exists() else []
    notes = json.loads(NOTES.read_text()) if NOTES.exists() else []
    lines = ["# STATUS — Det-Fly retrain queue", "",
             f"Updated: {now():%a %Y-%m-%d %H:%M %Z}. Deadline: {DEADLINE:%a %Y-%m-%d %H:%M %Z}.", "",
             f"**Stage: {STATE['stage']}**" + (f" — {STATE['detail']}" if STATE["detail"] else ""), ""]
    for s in ORDER:
        if (pr := train_progress(s)):
            lines.append(f"* Training {pr}")
    lines += ["", "## Finished stages", ""] + [f"* {m}" for m in marks] + [""]
    lines += ["## Outputs", "", "* `DETFLY_FULL_RESULTS.md` (old n/s/m on full Det-Fly)",
              "* `DETFLY_RETRAIN_RESULTS.md` (old vs new per size)", "* `detfly_split.json` (split definition)",
              "* `detfly_retrain/` (all JSON results)", "* `presentation_assets_v2/NUMBERS.md` (slide numbers, written last)",
              "", "## Notes / anything odd", ""]
    lines += [f"* {n['time']}: {n['note']}" for n in notes] or ["* none"]
    extra = ROOT / "detfly_retrain" / "caveats.md"
    if extra.exists():
        lines += ["", extra.read_text().strip()]
    (ROOT / "STATUS.md").write_text("\n".join(lines) + "\n")


def poll_status():
    write_status()


# ---------------------------------------------------------------- stages
def report():
    rp = ROOT / "scripts" / "new_retrain_report.py"
    if rp.exists():
        if run([PY, str(rp)], "report", gpu=False) != 0:
            note("report generation failed (see logs/report.log)")
    else:
        note("scripts/new_retrain_report.py missing at report time; reports not regenerated")


def stage_split():
    if done("split"):
        return
    STATE.update(stage="split", detail="")
    if run([PY, "scripts/build_detfly_split.py"], "split", gpu=False) != 0:
        raise SystemExit("split failed")
    mark("split")
    push("Det-Fly retrain: split definition (detfly_split.json)")


def eval_jobs(tag, w, size, prelim=False):
    jobs = [("test020", None), ("test010", None)] + ([("prelim", None)] if prelim else [])
    for st, _ in jobs:
        out = EV / f"{tag}_{st}.json"
        if not out.exists():
            STATE["detail"] = f"{tag}: Det-Fly {st}"
            write_status()
            if run([PY, "detector/new_eval_detfly.py", "--weights", w, "--model-tag", tag, "--set", st],
                   f"eval_{tag}_{st}") != 0:
                note(f"Det-Fly eval {tag} {st} failed (see logs/eval_{tag}_{st}.log)")
    for st, data in (("all4_test", "data/yolo/drone4_bal.yaml"), ("dut_test", "data/yolo/dut_anti_uav.yaml")):
        out = EV / f"{tag}_{st}.json"
        if not out.exists():
            STATE["detail"] = f"{tag}: {st}"
            write_status()
            if run([PY, "detector/eval.py", "--weights", w, "--data", data, "--tag", f"{tag}_{st}",
                    "--out", str(out), "--project", str(ROOT / "runs" / "new_val")], f"eval_{tag}_{st}") != 0:
                note(f"{st} eval {tag} failed")
    out = EV / f"{tag}_false_positives.json"
    if not out.exists():
        STATE["detail"] = f"{tag}: false drones"
        write_status()
        if run([PY, "detector/false_positives.py", "--weights", w, "--out", str(out)], f"eval_{tag}_fp") != 0:
            note(f"false-positive eval {tag} failed")


def stage_old_eval():
    if done("old_eval"):
        return
    STATE.update(stage="old_eval (old n/s/m on full Det-Fly + DUT)")
    for s in ORDER:
        w = ROOT / OLD_W[s]
        if not w.exists():
            note(f"old {s} weights not found at {OLD_W[s]}; skipped")
            continue
        eval_jobs(f"old_{s}", str(w), s, prelim=(s == "s"))
        report()
        push(f"Det-Fly retrain: old {s} evaluated on full Det-Fly / DUT")
    mark("old_eval")
    report()
    push("Det-Fly retrain: old n/s/m on full Det-Fly (DETFLY_FULL_RESULTS.md)")


def stage_train(size):
    if done(f"train_{size}"):
        return True
    STATE.update(stage=f"train_{size}", detail="")
    capf = MARK / f"cap_{size}.json"
    if capf.exists():
        cap = json.loads(capf.read_text())["time_h"]
    else:
        later = ORDER[ORDER.index(size) + 1:]
        remaining = (DEADLINE - now()).total_seconds() / 3600
        avail = remaining - len(later) * (OLD_CAP_H + EVAL_H) - EVAL_H - FINAL_H - SAFETY_H
        cap = round(min(OLD_CAP_H, avail), 2)
        if cap < 1.0:
            note(f"{size}: only {avail:.1f} h available before the deadline after reserving time for the rest; "
                 f"new {size} NOT trained (SKIPPED)")
            mark(f"train_{size}")
            mark(f"skipped_{size}")
            return False
        if cap < OLD_CAP_H:
            note(f"new {size} is SHORTENED: time cap {cap} h instead of the old {OLD_CAP_H} h (deadline)")
        MARK.mkdir(parents=True, exist_ok=True)
        capf.write_text(json.dumps({"time_h": cap, "set_at": f"{now():%Y-%m-%d %H:%M %Z}",
                                    "remaining_h_at_start": round(remaining, 2)}) + "\n")
    if (ROOT / "runs" / f"{size}_detfly_new" / "weights" / "last.pt").exists():
        note(f"new {size}: training resumed from last.pt after an interruption")
    write_status()
    push(f"STATUS: start training new {size} (time cap {cap} h)")
    rc = run([PY, "detector/new_train_detfly.py", "--size", size, "--batch", str(BATCH[size]), "--time", str(cap),
              "--log", str(LOGS / f"train_{size}.log")], f"train_{size}", poll=poll_status)
    if rc != 0 or not (ROOT / "detfly_retrain" / f"train_{size}.json").exists():
        note(f"training new {size} exited with code {rc}; rerun the queue to resume")
        raise SystemExit(f"train {size} failed")
    tj = json.loads((ROOT / "detfly_retrain" / f"train_{size}.json").read_text())
    if tj.get("batch_reductions"):
        note(f"new {size}: Ultralytics reduced the batch after CUDA OOM: {tj['batch_reductions']}")
    mark(f"train_{size}")
    return True


def stage_eval_new(size):
    if done(f"eval_new_{size}") or done(f"skipped_{size}"):
        return
    STATE.update(stage=f"eval_new_{size}")
    eval_jobs(f"new_{size}", str(ROOT / "runs" / f"{size}_detfly_new" / "weights" / "best.pt"), size)
    mark(f"eval_new_{size}")
    report()
    push(f"Det-Fly retrain: new {size} trained and evaluated (DETFLY_RETRAIN_RESULTS.md)")


def stage_final():
    if done("final"):
        return
    STATE.update(stage="final (slide assets v2 + sim with new noise)", detail="")
    write_status()
    sh = ROOT / "scripts" / "new_final_assets.sh"
    if sh.exists():
        rc = run(["bash", str(sh)], "final_assets")
        if rc != 0:
            note(f"final asset script exited with {rc} (see logs/final_assets.log); partial assets pushed")
    else:
        note("scripts/new_final_assets.sh missing; slide assets v2 not generated")
    report()
    mark("final")
    STATE.update(stage="DONE", detail="")
    push("Det-Fly retrain: slide assets v2, sim with new noise, final report")


def main():
    LOGS.mkdir(exist_ok=True)
    log(f"queue start, pid {os.getpid()}")
    try:
        stage_split()
        stage_old_eval()
        for s in ORDER:
            stage_train(s)
            stage_eval_new(s)
        stage_final()
        STATE.update(stage="DONE")
        push("STATUS: DONE")
        log("queue finished")
    except SystemExit as e:
        STATE.update(stage=f"STOPPED: {e}")
        push(f"STATUS: queue stopped ({e})")
        raise


if __name__ == "__main__":
    main()
