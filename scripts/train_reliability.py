#!/usr/bin/env python
# scripts/train_reliability.py
"""按 spec 2.4 的折套法训练两个可靠性头。

第 k 个头在 folds != k 的折外检出上训练，在 fold k 的折外检出上评估；报告的分数全部来自各自 fold k，
不做二次选择。检出阈值用 Task 10 定下的 SCORE_MIN，不再调。

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/train_reliability.py --score-min 0.30
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, VIEWS, load_lesions  # noqa: E402
from anatobind.eval.detect3d import failure_labels  # noqa: E402
from anatobind.model.reliability import (  # noqa: E402
    LesionReliability, ScanReliability, lesion_scalars, scan_scalars,
)
from anatobind.train.train_detector import load_fold  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)


def lesions_by_file(export_root):
    out = {}
    for r in load_lesions(export_root):
        out.setdefault(r["file"], []).append(r)
    return out


def gather(export_root, det_root, fold, score_min, by_file):
    """fold 的折外检出 -> (逐病灶样本, 扫描级样本)。"""
    _, held, _ = load_fold(export_root, fold)
    lesion_rows, scan_rows = [], []
    for f in held:
        patient = json.loads((Path(export_root) / f / "meta.json").read_text())["patient_id"]
        for view in VIEWS:
            path = Path(det_root) / f"fold{fold}" / f"{f}__{view}.pkl"
            if not path.exists():
                continue
            blob = pickle.load(open(path, "rb"))
            dets = [d for d in blob["dets"] if d["score"] >= score_min]
            per, scan = failure_labels(by_file.get(f, []), dets)
            for r in per:
                lesion_rows.append({"file": f, "view": view, "fold": fold, "patient": patient,
                                    "embed": r["embed"], "scalars": lesion_scalars(r),
                                    "label": float(r["correct"]), "peak_score": r["score"]})
            scan_rows.append({"file": f, "view": view, "fold": fold, "patient": patient,
                              "global_feat": blob["global_feat"], "scalars": scan_scalars(per),
                              "label": float(1 - scan),
                              "peak_score": max([r["score"] for r in per], default=0.0),
                              "min_lesion": min([r["score"] for r in per], default=0.0)})
    return lesion_rows, scan_rows


def fit(head, feats, scalars, labels, steps, batch, lr, seed):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(labels)
    head.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        loss = F.binary_cross_entropy_with_logits(head(feats[idx], scalars[idx]), labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return head


def stack(rows, key):
    return torch.from_numpy(np.stack([r[key] for r in rows]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-min", type=float, required=True, help="Task 10 定下的 SCORE_MIN")
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--det-root", type=Path, default=EXPORT_ROOT / "detections")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    by_file = lesions_by_file(a.export_root)
    per_fold = {k: gather(a.export_root, a.det_root, k, a.score_min, by_file) for k in FOLDS}

    out_rows = []
    for k in FOLDS:
        tr_les = [r for j in FOLDS if j != k for r in per_fold[j][0]]
        tr_scan = [r for j in FOLDS if j != k for r in per_fold[j][1]]
        te_les, te_scan = per_fold[k]
        run = Path(f"runs/reliability_fold{k}")
        run.mkdir(parents=True, exist_ok=True)

        if tr_les and te_les:
            head = LesionReliability(embed_dim=len(tr_les[0]["embed"]), n_scalar=3)
            fit(head, stack(tr_les, "embed"), stack(tr_les, "scalars"),
                torch.tensor([r["label"] for r in tr_les]), a.steps, 256, a.lr, a.seed)
            with torch.no_grad():
                score = torch.sigmoid(head(stack(te_les, "embed"), stack(te_les, "scalars"))).numpy()
            torch.save(head.state_dict(), run / "lesion.pt")
            for r, sc in zip(te_les, score):
                out_rows.append({"kind": "lesion", "file": r["file"], "view": r["view"], "fold": k,
                                 "patient": r["patient"], "head_score": float(sc),
                                 "peak_score": r["peak_score"], "correct": r["label"], "min_lesion": ""})

        if not (tr_scan and te_scan):
            print(f"fold {k}: no scan rows, skipping the scan head", flush=True)
            continue

        head = ScanReliability(global_dim=len(tr_scan[0]["global_feat"]), n_scalar=5)
        fit(head, stack(tr_scan, "global_feat"), stack(tr_scan, "scalars"),
            torch.tensor([r["label"] for r in tr_scan]), a.steps, 64, a.lr, a.seed)
        with torch.no_grad():
            score = torch.sigmoid(head(stack(te_scan, "global_feat"), stack(te_scan, "scalars"))).numpy()
        torch.save(head.state_dict(), run / "scan.pt")
        for r, sc in zip(te_scan, score):
            out_rows.append({"kind": "scan", "file": r["file"], "view": r["view"], "fold": k,
                             "patient": r["patient"], "head_score": float(sc),
                             "peak_score": r["peak_score"], "correct": r["label"],
                             "min_lesion": r["min_lesion"]})

    fields = ["kind", "file", "view", "fold", "patient", "head_score", "peak_score", "correct", "min_lesion"]
    out = Path("runs/reliability_scores.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    n_les = sum(1 for r in out_rows if r["kind"] == "lesion")
    print(f"{out}: {n_les} lesion rows, {len(out_rows) - n_les} scan rows")


if __name__ == "__main__":
    main()
