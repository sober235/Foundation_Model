#!/usr/bin/env python
# scripts/gpu_queue.py
"""Run the brain multi-disease nnU-Net trainings on whatever GPUs are idle (spec 2026-09-29 §4, M6, M14).

Jobs go out fold by fold (fold 0 of every disease first). Every POLL_SECONDS the queue looks for GPUs with no compute
process and almost no memory in use, and starts the next jobs there, one per GPU, never more than MAX_JOBS at once.
A card somebody else uses is never touched. Control files in the log directory: `skip_<dataset id>` drops the jobs of
that dataset that have not started; `stop` lets the running jobs finish and starts nothing new.

  cd <worktree> && PYTHONNOUSERSITE=1 PYTHONPATH=. setsid nohup ~/anaconda3/envs/nvgen/bin/python scripts/gpu_queue.py \
      --diseases glioma metastasis infarct --folds 0 1 2 3 4 > logs/brain_disease/queue.log 2>&1 < /dev/null &
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain_detector_train import idle_gpus, query_busy_pids, query_nvidia_smi  # noqa: E402
from anatobind.infer.knee import NNUNET_ROOT  # noqa: E402
from anatobind.nnunet.brain_disease import CONFIG, DISEASES, TRAINER, fold_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs" / "brain_disease"
MAX_JOBS = 6
N_PROC_DA = 6
POLL_SECONDS = 60


def job_list(diseases, folds):
    """Fold-major order: fold 0 of every disease, then fold 1, ..."""
    return [(d, int(f)) for f in folds for d in diseases]


def log_path(log_dir, job, trainer):
    return Path(log_dir) / f"{DISEASES[job[0]]['name']}_{trainer}_fold{job[1]}.log"


def refusal(results_root, log_dir, job, trainer):
    """Why this job must not start (its outputs exist and would be overwritten), or None."""
    r, l = fold_dir(results_root, job[0], job[1], trainer), log_path(log_dir, job, trainer)
    if r.exists():
        return f"result folder {r} exists"
    if l.exists():
        return f"log {l} exists"
    return None


def command(job, gpu, trainer, log, env_sh, n_proc_da=N_PROC_DA):
    cmd = ["nnUNetv2_train", str(DISEASES[job[0]]["id"]), CONFIG, str(int(job[1])), "-tr", trainer, "--npz"]
    inner = (f"{{ source {shlex.quote(str(env_sh))} && export nnUNet_n_proc_DA={int(n_proc_da)} && "
             f"CUDA_VISIBLE_DEVICES={int(gpu)} nice -n 19 {' '.join(shlex.quote(c) for c in cmd)}; }} "
             f"> {shlex.quote(str(log))} 2>&1")
    return ["bash", "-c", inner]


def skipped_diseases(log_dir):
    return {d for d, spec in DISEASES.items() if (Path(log_dir) / f"skip_{spec['id']}").exists()}


def plan_launches(pending, running_gpus, idle, max_jobs):
    """[(job, gpu)] to start now: pending order is kept, a GPU that runs one of our jobs is never reused, and at most
    max_jobs run at once."""
    free = [g for g in idle if g not in running_gpus]
    room = max(0, max_jobs - len(running_gpus))
    return list(zip(pending, free))[:room]


def say(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def idle_now(gpus):
    """Idle GPUs among gpus, or None when nvidia-smi cannot be queried or read right now: the queue then starts
    nothing in this round and asks again at the next poll."""
    try:
        return idle_gpus(query_nvidia_smi(), gpus, query_busy_pids())
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        say(f"GPU query failed ({type(e).__name__}: {e}); nothing starts in this round")
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Queue nnU-Net trainings of the brain disease detectors on idle GPUs")
    ap.add_argument("--diseases", nargs="+", choices=sorted(DISEASES), required=True)
    ap.add_argument("--folds", nargs="+", type=int, required=True)
    ap.add_argument("--trainer", default=TRAINER)
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)), help="GPUs the queue may consider")
    ap.add_argument("--max-jobs", type=int, default=MAX_JOBS)
    ap.add_argument("--dry-run", action="store_true", help="print the first round of launches and exit")
    a = ap.parse_args(argv)

    results_root = Path(os.environ.get("nnUNet_results") or NNUNET_ROOT / "results")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending, running = job_list(a.diseases, a.folds), {}
    say(f"queue of {len(pending)} jobs: {pending}")
    while True:
        for gpu, (job, proc) in list(running.items()):
            if proc.poll() is not None:
                say(f"finished {job} on GPU {gpu} with exit code {proc.returncode}")
                del running[gpu]
        if (LOG_DIR / "stop").exists() and pending:
            say(f"stop file found: {len(pending)} jobs will not start: {pending}")
            pending = []
        skip = skipped_diseases(LOG_DIR)
        if any(j[0] in skip for j in pending):
            say(f"skip file found: dropping {[j for j in pending if j[0] in skip]}")
            pending = [j for j in pending if j[0] not in skip]
        if not pending and not running:
            say("queue empty, nothing running: done")
            return 0
        if pending and len(running) < a.max_jobs:
            for job, gpu in plan_launches(pending, set(running), idle_now(a.gpus) or [], a.max_jobs):
                pending.remove(job)
                why = refusal(results_root, LOG_DIR, job, a.trainer)
                if why:
                    say(f"refused {job}: {why}")
                    continue
                cmd = command(job, gpu, a.trainer, log_path(LOG_DIR, job, a.trainer), REPO / "scripts/nnunet_env.sh")
                if a.dry_run:
                    say(f"would launch {job} on GPU {gpu}: {cmd[2]}")
                    continue
                proc = subprocess.Popen(cmd, cwd=str(REPO), start_new_session=True, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                running[gpu] = (job, proc)
                say(f"launched {job} on GPU {gpu} (pid {proc.pid})")
        if a.dry_run:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
