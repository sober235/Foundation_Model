#!/usr/bin/env python
# scripts/run_relation_baselines.py
"""Outer five-fold / inner-selection runs of the five relation arms on one feature table (spec 2026-09-27 §5–6.1).

  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/run_relation_baselines.py \
      --table /data2/congcong/data/FM_data/derived/relation/v1 --out runs/relation/c1_stage1 --labels C1 --device cuda

Stage 2 (P13): --labels R --sealed-dir ... --manifest ... --init-from runs/relation/c1_stage1
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.level_r.admin import sha256_file  # noqa: E402
from anatobind.relation.baselines import GEO_KINDS, PRIOR_VARIANTS  # noqa: E402
from anatobind.relation.cv import run_b0, run_bgeo, run_bprior, select_config, write_preds  # noqa: E402
from anatobind.relation.labels import c1_labels, r_train_labels  # noqa: E402
from anatobind.relation.table import load_table  # noqa: E402

ALL_ARMS = ("b0", "bprior", "bgeo", "b1", "b2")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", choices=("C1", "R"), required=True)
    ap.add_argument("--sealed-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16, help="cap on BLAS / OpenMP / torch threads (CLAUDE.md: <= 48)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-k", type=int, default=5)
    ap.add_argument("--init-from", type=Path, help="a previous run directory whose models/<arm>_fold<k>.pt initialise B1/B2 (stage 2)")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists; runs are never overwritten")
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=a.threads)                  # numpy BLAS and scikit-learn OpenMP pools for the rest of the run
    arms = a.arms.split(",")
    table, patches = load_table(a.table)
    if a.labels == "C1":
        labels = c1_labels(table)
    else:
        if not (a.sealed_dir and a.manifest):
            ap.error("--labels R needs --sealed-dir and --manifest")
        labels = lambda k: r_train_labels(k, a.sealed_dir, a.manifest)[0]      # noqa: E731  training folds only; never the test fold
    (a.out / "preds").mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run = {"table": str(a.table), "table_manifest_sha256": sha256_file(a.table / "manifest.json"), "labels": a.labels,
           "seed": a.seed, "epochs": a.epochs, "patience": a.patience, "inner_k": a.inner_k, "device": a.device, "threads": a.threads,
           "init_from": str(a.init_from) if a.init_from else None, "git_commit": commit, "records": {}}

    def save(name, preds, record):
        write_preds(a.out / "preds" / f"{name}.csv", preds)
        run["records"][name] = record
        print(f"{name}: written", flush=True)

    if "b0" in arms:
        save("b0", *run_b0(table))
    if "bprior" in arms:
        for v in PRIOR_VARIANTS:
            save(f"bprior_{v}", *run_bprior(table, labels, v))
    if "bgeo" in arms:
        scores = []
        for kind in GEO_KINDS:
            preds, rec = run_bgeo(table, labels, kind, a.seed, a.inner_k)
            save(f"bgeo_{kind}", preds, rec)
            scores.append((f"bgeo_{kind}", rec["mean_inner_score"]))
        best, sel = select_config(scores)
        run["bgeo_best"], run["bgeo_selection"] = best, sel
    torch_arms = [x for x in ("b1", "b2") if x in arms]
    if torch_arms:
        import torch
        from anatobind.relation.train import run_torch_arm
        torch.set_num_threads(a.threads)
        (a.out / "models").mkdir(exist_ok=True)
        for arm in torch_arms:
            init = None
            if a.init_from:                                  # stage 2: each fold fine-tunes its stage-1 model and config
                prev = json.loads((a.init_from / "run.json").read_text(encoding="utf-8"))["records"][arm]["folds"]
                init = {k: (prev[str(k)]["chosen"], torch.load(a.init_from / "models" / f"{arm}_fold{k}.pt", map_location="cpu", weights_only=True))
                        for k in sorted(set(table.folds().tolist()))}
            preds, rec, states = run_torch_arm(arm, table, patches, labels, seed=a.seed, device=a.device, epochs=a.epochs,
                                               patience=a.patience, inner_k=a.inner_k, init=init,
                                               log=lambda m: print(m, flush=True))
            for k, st in states.items():
                torch.save(st, a.out / "models" / f"{arm}_fold{k}.pt")
            save(arm, preds, rec)
    (a.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(f"run written to {a.out}")


if __name__ == "__main__":
    main()
