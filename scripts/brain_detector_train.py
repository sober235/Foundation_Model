#!/usr/bin/env python
# scripts/brain_detector_train.py
"""Launch nnU-Net v2 training for the brain small-lesion detector (Dataset903), one fold per idle GPU.

  source scripts/nnunet_env.sh
  PYTHONPATH=. python scripts/brain_detector_train.py --config 2d --folds 0 1 2 3 4 --gpus 0 1 2 3 --trainer nnUNetTrainer_250epochs
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.infer.knee import NNUNET_ROOT  # noqa: E402

DATASET_ID = "903"
IDLE_MEM_THRESHOLD_MIB = 1000


def idle_gpus(nvidia_smi_csv, candidates, busy_pids):
    """Parse `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits` output.

    Keeps candidates whose used memory is below IDLE_MEM_THRESHOLD_MIB and that have
    no entry in busy_pids (gpu index -> list of pids running compute on it).
    """
    used_mem = {}
    for line in nvidia_smi_csv.strip().splitlines():
        if not line.strip():
            continue
        idx_str, mem_str = line.split(",")
        used_mem[int(idx_str.strip())] = int(mem_str.strip())

    idle = []
    for gpu in candidates:
        if used_mem.get(gpu, IDLE_MEM_THRESHOLD_MIB) >= IDLE_MEM_THRESHOLD_MIB:
            continue
        if busy_pids.get(gpu):
            continue
        idle.append(gpu)
    return idle


def assign(folds, gpus):
    """Round-robin fold-to-GPU assignment; more folds than GPUs queue behind the first ones."""
    return [(f, gpus[i % len(gpus)]) for i, f in enumerate(folds)]


def train_command(config, fold, trainer):
    """nnU-Net v2 training command for one fold."""
    return ["nnUNetv2_train", DATASET_ID, config, str(fold), "-tr", trainer, "--npz"]


def existing_logs(config, trainer, folds, log_dir):
    """Target log paths for these folds that already exist under log_dir (would be silently
    overwritten by the chain's `>` redirect if launched)."""
    return [p for p in (log_dir / f"{config}_{trainer}_fold{f}.log" for f in folds) if p.exists()]


def existing_results(results_root, config, trainer, folds):
    """Existing nnU-Net fold output directories for these folds, in fold order (`nnUNetv2_train` without
    `--c` starts fresh inside an existing fold folder and overwrites checkpoint_best, checkpoint_final and
    validation/)."""
    base = Path(results_root) / f"Dataset{DATASET_ID}_FastMRIBrainSmallLesion" / f"{trainer}__nnUNetPlans__{config}"
    return [p for p in (base / f"fold_{f}" for f in folds) if p.exists()]


def query_nvidia_smi():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout


def query_busy_pids():
    """Map gpu index -> list of pids with a compute process on that GPU."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    uuid_out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    uuid_to_index = {}
    for line in uuid_out.strip().splitlines():
        if not line.strip():
            continue
        idx_str, uuid = line.split(",")
        uuid_to_index[uuid.strip()] = int(idx_str.strip())

    busy = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = line.split(",")
        gpu = uuid_to_index.get(uuid.strip())
        if gpu is None:
            continue
        busy.setdefault(gpu, []).append(int(pid.strip()))
    return busy


def build_chain(config, gpu, fold_list, trainer, repo_root, log_dir):
    """One setsid bash -c chain that runs fold_list sequentially, pinned to gpu."""
    parts = [f"source {shlex.quote(str(repo_root / 'scripts/nnunet_env.sh'))}"]
    for fold in fold_list:
        cmd = train_command(config, fold, trainer)
        log_path = log_dir / f"{config}_{trainer}_fold{fold}.log"
        cmd_str = " ".join(shlex.quote(part) for part in cmd)
        parts.append(f"CUDA_VISIBLE_DEVICES={gpu} nice -n 19 {cmd_str} > {shlex.quote(str(log_path))} 2>&1")
    chain = " && ".join(parts)
    return ["setsid", "bash", "-c", chain]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Launch nnU-Net training for Dataset903, one fold per idle GPU")
    ap.add_argument("--config", choices=("2d", "3d_fullres"), required=True, help="nnU-Net configuration")
    ap.add_argument("--folds", nargs="+", type=int, required=True, help="Folds to train")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3], help="Candidate GPU indices (default: 0 1 2 3)")
    ap.add_argument("--trainer", required=True, help="nnU-Net trainer class name")
    ap.add_argument("--dry-run", action="store_true", help="Print the chains without launching")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    log_dir = repo_root / "logs" / "brain_detector"
    log_dir.mkdir(parents=True, exist_ok=True)

    clobbered = existing_logs(args.config, args.trainer, args.folds, log_dir)
    if clobbered:
        names = ", ".join(str(p) for p in clobbered)
        print(f"Refusing to start: log file(s) already exist and would be overwritten: {names}", file=sys.stderr)
        return 1

    results_root = Path(os.environ.get("nnUNet_results") or NNUNET_ROOT / "results")
    existing = existing_results(results_root, args.config, args.trainer, args.folds)
    if existing:
        names = ", ".join(str(p) for p in existing)
        print(f"Refusing to start: nnU-Net output folder(s) already exist and would be overwritten: {names}", file=sys.stderr)
        return 1

    csv = query_nvidia_smi()
    busy_pids = query_busy_pids()
    idle = idle_gpus(csv, args.gpus, busy_pids)

    n_needed = min(len(args.gpus), len(args.folds))
    if len(idle) < n_needed:
        busy = [g for g in args.gpus if g not in idle]
        print(f"Refusing to start: need {n_needed} idle GPUs among {args.gpus}, found {len(idle)} idle ({idle}); busy: {busy}", file=sys.stderr)
        return 1

    assignment = assign(args.folds, idle)
    folds_by_gpu = {}
    for fold, gpu in assignment:
        folds_by_gpu.setdefault(gpu, []).append(fold)

    chains = {gpu: build_chain(args.config, gpu, fold_list, args.trainer, repo_root, log_dir) for gpu, fold_list in folds_by_gpu.items()}

    if args.dry_run:
        for gpu, chain in chains.items():
            print(f"GPU {gpu}: {' '.join(chain)}")
        return 0

    for gpu, chain in chains.items():
        print(f"Launching GPU {gpu}: folds {folds_by_gpu[gpu]}")
        subprocess.Popen(chain, start_new_session=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
