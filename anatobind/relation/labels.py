"""Label sources for the relation baselines (spec 2026-09-27 §6.2): C1 from the table, R from the sealed Level R folds.
Both hand the evaluation the same structure, {lesion_id: frozenset of acceptable slot names}. Readers answer "other";
in the table that is the seventh slot, other_deep_grey."""
import json
from pathlib import Path

import numpy as np

from anatobind.eval.level_r_labels import SealedAccessError, load_test_labels, load_train_labels
from anatobind.relation.table import SLOTS

HOST_OF_READER = {"other": "other_deep_grey"}


def c1_labels(table):
    return {int(r["lesion_id"]): frozenset({r["c1_class"]}) for r in table.rows}


def from_level_r_rows(rows):
    labels, excluded = {}, []
    for r in rows:
        if r["not_a_lesion"]:
            excluded.append(int(r["lesion_id"]))
            continue
        hosts = frozenset(HOST_OF_READER.get(h, h) for h in r["acceptable_hosts"])
        unknown = hosts - set(SLOTS)
        if not hosts or unknown:
            raise ValueError(f"lesion {r['lesion_id']}: acceptable hosts {sorted(hosts)} are not table slots")
        labels[int(r["lesion_id"])] = hosts
    return labels, sorted(excluded)


def r_train_labels(k, sealed_dir, manifest_path):
    return from_level_r_rows(load_train_labels(k, sealed_dir, manifest_path))


def r_test_labels(sealed_dir, manifest_path, unblind=False):
    """Every sealed fold's final labels: the one-shot final evaluation (spec §6.8). unblind must be the literal True."""
    if unblind is not True:
        raise SealedAccessError("test labels open only with unblind=True, from the final evaluation script")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    folds = sorted(int(name[4:]) for name in manifest)
    rows = [r for k in folds for r in load_test_labels(k, unblind=True, sealed_dir=sealed_dir, manifest_path=manifest_path)]
    return from_level_r_rows(rows)


def acceptable_matrix(labels, lesion_ids):
    ids = np.asarray(lesion_ids)
    has = np.array([int(l) in labels for l in ids], bool)
    acc = np.zeros((len(ids), len(SLOTS)), bool)
    for i, l in enumerate(ids):
        for h in labels.get(int(l), ()):
            acc[i, SLOTS.index(h)] = True
    return has, acc
