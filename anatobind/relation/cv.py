"""Outer five folds by patient (the Level R fold table), inner patient folds for selection, the P11 tie rule, the
out-of-fold prediction files, and the drivers of the arms that need no torch (spec 2026-09-27 §5.8, §6.1)."""
import csv
import json
from pathlib import Path

import numpy as np

from anatobind.eval.relation_metrics import is_correct
from anatobind.relation.baselines import GEO_GRIDS, GeoLearner, PriorModel, b0_probs
from anatobind.relation.labels import acceptable_matrix
from anatobind.relation.table import N_OUT, SLOTS, SLOT_PREFIX, features_flat

TIE_TOL = 1e-3
PROB_COLUMNS = [f"p_{SLOT_PREFIX[s]}" for s in SLOTS] + ["p_none"]


def outer_folds(table):
    folds = table.folds()
    return [(int(k), np.nonzero(folds != k)[0], np.nonzero(folds == k)[0]) for k in sorted(set(folds.tolist()))]


def inner_folds(patients, k=5, seed=0):
    """Patients shuffled with ``seed`` and dealt round-robin into k groups; indices are positions in ``patients``."""
    p = np.asarray(patients)
    ids = np.unique(p)
    ids = ids[np.random.default_rng(seed).permutation(len(ids))]
    out = []
    for g in range(k):
        held = set(ids[g::k].tolist())
        val = np.array([i for i in range(len(p)) if p[i] in held], int)
        train = np.array([i for i in range(len(p)) if p[i] not in held], int)
        if len(val):
            out.append((train, val))
    return out


def select_config(scores, tol=TIE_TOL):
    """scores: [(key, score)] in tie order. The first key whose score is within ``tol`` of the best wins (P11)."""
    best = max(s for _, s in scores)
    chosen = next(k for k, s in scores if s >= best - tol)
    tie = sum(1 for _, s in scores if s >= best - tol) > 1
    return chosen, {"scores": [(str(k), float(s)) for k, s in scores], "chosen": str(chosen), "tie": bool(tie)}


def labels_for_fold(labels, k):
    return labels(k) if callable(labels) else labels


def set_accuracy(probs, acceptable):
    c = is_correct(probs, acceptable)
    return float(c.mean()) if len(c) else float("nan")


def trainable_rows(has, acceptable, candidates):
    """A lesion trains an arm only if it is labelled and one of its acceptable slots is a candidate: every arm's output
    is restricted to the candidates (P7), so a truth outside them is unanswerable, and for B1 its loss would be infinite.
    Such lesions still count, as wrong for every arm, in evaluation."""
    return np.asarray(has, bool) & (np.asarray(acceptable, bool) & np.asarray(candidates, bool)).any(1)


def empty_preds(table):
    return {"lesion_id": table.lesion_id.copy(), "fold": table.folds().copy(), "config": np.array([""] * len(table), object),
            "probs": np.zeros((len(table), N_OUT))}


def write_preds(path, preds):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lesion_id", "fold", "config"] + PROB_COLUMNS)
        for i in range(len(preds["lesion_id"])):
            w.writerow([int(preds["lesion_id"][i]), int(preds["fold"][i]), preds["config"][i]] + [f"{v:.8f}" for v in preds["probs"][i]])


def read_preds(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {"lesion_id": np.array([int(r["lesion_id"]) for r in rows]), "fold": np.array([int(r["fold"]) for r in rows]),
            "config": np.array([r["config"] for r in rows], object),
            "probs": np.array([[float(r[c]) for c in PROB_COLUMNS] for r in rows])}


def run_b0(table):
    preds = empty_preds(table)
    preds["probs"] = b0_probs(table)
    preds["config"][:] = "c1"
    return preds, {"arm": "b0"}


def run_bprior(table, labels, variant):
    preds = empty_preds(table)
    preds["config"][:] = variant
    record = {"arm": f"bprior_{variant}", "n_untrainable": {}}
    for k, tr, te in outer_folds(table):
        has, acc = acceptable_matrix(labels_for_fold(labels, k), table.lesion_id)
        ok = trainable_rows(has, acc, table.candidates())
        record["n_untrainable"][k] = int((has[tr] & ~ok[tr]).sum())
        tr = tr[ok[tr]]
        preds["probs"][te] = PriorModel(variant).fit(table.subset(tr), acc[tr]).predict(table.subset(te))
    return preds, record


def run_bgeo(table, labels, kind, seed=0, inner_k=5):
    X = features_flat(table)
    cand = table.candidates()
    patients = table.patients()
    preds = empty_preds(table)
    record = {"arm": f"bgeo_{kind}", "folds": {}}
    for k, tr, te in outer_folds(table):
        has, acc = acceptable_matrix(labels_for_fold(labels, k), table.lesion_id)
        ok = trainable_rows(has, acc, cand)
        n_untrainable = int((has[tr] & ~ok[tr]).sum())
        tr = tr[ok[tr]]
        scores = []
        for params in GEO_GRIDS[kind]:
            accs = []
            for itr, ival in inner_folds(patients[tr], inner_k, seed):
                m = GeoLearner(kind, params, seed).fit(X[tr[itr]], acc[tr[itr]])
                accs.append(set_accuracy(m.predict_probs(X[tr[ival]], cand[tr[ival]]), acc[tr[ival]]))
            scores.append((json.dumps(params, sort_keys=True), float(np.mean(accs))))
        chosen, rec = select_config(scores)
        params = json.loads(chosen)
        preds["probs"][te] = GeoLearner(kind, params, seed).fit(X[tr], acc[tr]).predict_probs(X[te], cand[te])
        preds["config"][te] = chosen
        record["folds"][k] = {**rec, "n_untrainable": n_untrainable}
    record["mean_inner_score"] = float(np.mean([max(s for _, s in record["folds"][k]["scores"]) for k in record["folds"]]))
    return preds, record
