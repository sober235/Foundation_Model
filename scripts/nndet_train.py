#!/usr/bin/env python
# scripts/nndet_train.py
"""Launch nnDetection training (with its sweep) of one Task903 fold on one idle GPU (spec 2026-09-28 brain-nndet §6).

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/nndet_train.py --fold 0 --gpus 3 6 7
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain_detector_train import idle_gpus, query_busy_pids, query_nvidia_smi  # noqa: E402
from anatobind.nndet.brain_task import MODEL_ID, PLAN_ID, TASK_ID, TASK_NAME, read_splits_pkl, splits_payload  # noqa: E402

FM = Path("/data2/congcong/data/FM_data")
DET_DATA = FM / "derived/nndet"
DET_MODELS = FM / "derived/nndet_models"
NNUNET_SPLITS = FM / "derived/nnunet/preprocessed/Dataset903_FastMRIBrainSmallLesion/splits_final.json"
REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs/brain_nndet"


def train_dir(det_models, fold):
    return Path(det_models) / TASK_NAME / MODEL_ID / f"fold{fold}"


def preflight(det_data, det_models, fold, nnunet_splits, log_dir):
    """Refuse unless the preprocessed plan data exist, splits_final.pkl equals Dataset903's splits (nnDetection makes
    its own KFold split when the file is missing), and neither the training dir nor the log exists yet."""
    prep = Path(det_data) / TASK_NAME / "preprocessed"
    if not (prep / PLAN_ID / "imagesTr").is_dir():
        raise FileNotFoundError(f"{prep / PLAN_ID / 'imagesTr'} missing: run nndet_prep and nndet_unpack first")
    if read_splits_pkl(prep / "splits_final.pkl") != splits_payload(json.loads(Path(nnunet_splits).read_text())):
        raise ValueError(f"{prep / 'splits_final.pkl'} differs from Dataset903's splits")
    td = train_dir(det_models, fold)
    if td.exists():
        raise FileExistsError(f"{td} exists; nnDetection's overwrite mode would reuse it")
    log = Path(log_dir) / f"fold{fold}.log"
    if log.exists():
        raise FileExistsError(f"{log} exists")
    return log


def command(fold, gpu, log, env_sh):
    inner = (f"source {shlex.quote(str(env_sh))} && CUDA_VISIBLE_DEVICES={int(gpu)} nice -n 19 "
             f"nndet_train {TASK_ID} -o exp.fold={int(fold)} --sweep > {shlex.quote(str(log))} 2>&1")
    return ["setsid", "bash", "-c", inner]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Launch nnDetection training of one Task903 fold")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpus", type=int, nargs="+", required=True, help="candidate GPUs; the first idle one is used")
    a = ap.parse_args(argv)
    log = preflight(DET_DATA, DET_MODELS, a.fold, NNUNET_SPLITS, LOG_DIR)
    idle = idle_gpus(query_nvidia_smi(), a.gpus, query_busy_pids())
    if not idle:
        raise RuntimeError(f"none of GPUs {a.gpus} is idle")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(command(a.fold, idle[0], log, REPO / "scripts/nndet_env.sh"), cwd=str(REPO),
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"launched fold {a.fold} on GPU {idle[0]} (pid {proc.pid}); log {log}; "
          f"training dir {train_dir(DET_MODELS, a.fold)}")


if __name__ == "__main__":
    main()
