"""Deterministic stratified k-fold assignment at scan (= subject) level."""
import random


def stratum_of(n_in_seg):
    return 0 if n_in_seg == 0 else (1 if n_in_seg <= 2 else 2)


def five_fold_by_scan(scans, strata, seed=0, n_folds=5):
    rng = random.Random(seed)
    folds = {}
    next_fold = 0
    for st in sorted(set(strata.values())):
        members = sorted(s for s in scans if strata[s] == st)
        rng.shuffle(members)
        for s in members:  # deal round-robin, continuing the counter across strata keeps totals balanced
            folds[s] = next_fold % n_folds
            next_fold += 1
    return folds
