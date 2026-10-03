#!/usr/bin/env python
# scripts/brain_anatomy_train.py
"""Launch the two S4 trainings on idle GPUs, one job per card (spec 2026-10-02 A9, A10, A15):
Dataset907 3d_fullres fold 0 (student, nnUNetTrainer_250epochs_NoMirroring: its classes have a side, spec A17) and
Dataset908 2d fold 0 (brain outline, nnUNetTrainer_250epochs).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_train.py --jobs student outline --gpus 0 1 2 3 4 5 6 7

A job whose result folder or log exists is refused (nnU-Net would start over inside the folder); a card with a
compute process or more than IDLE_MEM_THRESHOLD_MIB in use is never touched."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain_detector_train import idle_gpus, query_busy_pids, query_nvidia_smi  # noqa: E402
from anatobind.infer.brain_anatomy import OUTLINE, STUDENT  # noqa: E402
from anatobind.infer.knee import NNUNET_ROOT  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs" / "brain_anatomy"
JOBS = {"student": {**STUDENT, "name": "Dataset907_BrainAnatomyFLAIR"},       # id, config and trainer: the inference chain's
        "outline": {**OUTLINE, "name": "Dataset908_FastMRIBrainOutline"}}
FOLD = 0


def result_dir(results_root, job, fold=FOLD):
    j = JOBS[job]
    return Path(results_root) / j["name"] / f"{j['trainer']}__nnUNetPlans__{j['config']}" / f"fold_{fold}"


def log_path(log_dir, job, fold=FOLD):
    j = JOBS[job]
    return Path(log_dir) / f"{j['name']}_{j['config']}_{j['trainer']}_fold{fold}.log"


def refusal(results_root, log_dir, job):
    r, l = result_dir(results_root, job), log_path(log_dir, job)
    if r.exists():
        return f"result folder {r} exists"
    if l.exists():
        return f"log {l} exists"
    return None


def train_command(job, fold=FOLD):
    j = JOBS[job]
    return ["nnUNetv2_train", str(j["id"]), j["config"], str(fold), "-tr", j["trainer"]]


def launch_command(job, gpu, repo_root, log_dir):
    cmd = " ".join(shlex.quote(p) for p in train_command(job))
    chain = (f"source {shlex.quote(str(Path(repo_root) / 'scripts/nnunet_env.sh'))} && "
             f"CUDA_VISIBLE_DEVICES={int(gpu)} nice -n 19 {cmd} > {shlex.quote(str(log_path(log_dir, job)))} 2>&1")
    return ["setsid", "bash", "-c", chain]


def plan(jobs, gpus, results_root, log_dir):
    """[(job, gpu)] for the jobs that may start now, and the reasons the others do not."""
    starts, reasons = [], {}
    free = list(gpus)
    for job in jobs:
        why = refusal(results_root, log_dir, job)
        if why:
            reasons[job] = why
        elif not free:
            reasons[job] = "no idle GPU left"
        else:
            starts.append((job, free.pop(0)))
    return starts, reasons


def main(argv=None):
    ap = argparse.ArgumentParser(description="Launch the S4 trainings on idle GPUs")
    ap.add_argument("--jobs", nargs="+", choices=sorted(JOBS), default=["student", "outline"])
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)), help="candidate cards; only idle ones are used")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    idle = idle_gpus(query_nvidia_smi(), a.gpus, query_busy_pids())
    starts, reasons = plan(a.jobs, idle, NNUNET_ROOT / "results", LOG_DIR)
    for job, why in reasons.items():
        print(f"not started: {job}: {why}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for job, gpu in starts:
        cmd = launch_command(job, gpu, REPO, LOG_DIR)
        if a.dry_run:
            print(f"would launch {job} on GPU {gpu}: {' '.join(shlex.quote(c) for c in cmd)}")
            continue
        p = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print(f"launched {job} on GPU {gpu} (pid {p.pid}); log {log_path(LOG_DIR, job)}; results {result_dir(NNUNET_ROOT / 'results', job)}")
    return 0 if starts or not a.jobs else 1


if __name__ == "__main__":
    sys.exit(main())
