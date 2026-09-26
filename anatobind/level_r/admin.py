"""Admin operations behind scripts/level_r_admin.py: patient folds (v2.6 §4.4 / §12.6), per-reader orders, CSV export,
one-shot sealing with a sha256 manifest (v2.6 §12.7) and online sqlite backups. Nothing here ever overwrites a file."""
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from anatobind.data_engine.fastmri_knee import assert_folds_by_patient, make_folds
from anatobind.level_r.store import now_iso

LABEL_COLUMNS = ("row_id", "reader_id", "lesion_id", "primary_host", "acceptable_hosts", "topography", "adjacency", "ambiguity",
                 "not_a_lesion", "local_quality", "confidence", "comment", "time_seconds", "window", "ts")
ADJ_COLUMNS = ("row_id", "adjudicator_id", "lesion_id", "primary_host", "acceptable_hosts", "topography", "adjacency", "ambiguity",
               "not_a_lesion", "reason", "ts")
FINAL_COLUMNS = ("lesion_id", "status", "primary_host", "acceptable_hosts", "not_a_lesion")


def make_patient_folds(patient_ids, k=5, seed=0):
    """Same rule as fastmri_knee.make_folds, keyed by patient: sorted unique ids, seeded permutation, round robin."""
    return make_folds({p: p for p in set(patient_ids)}, k, seed)


def lesion_folds(registry, patient_fold):
    folds = {str(r["lesion_id"]): patient_fold[r["patient_id"]] for r in registry}
    assert_folds_by_patient(folds, {str(r["lesion_id"]): r["patient_id"] for r in registry})
    return {int(k): v for k, v in folds.items()}


def make_order(lesion_ids, pilot_ids, seed):
    rng = np.random.default_rng(seed)
    pilot = [l for l in lesion_ids if l in pilot_ids]
    rest = [l for l in lesion_ids if l not in pilot_ids]
    return [pilot[int(i)] for i in rng.permutation(len(pilot))] + [rest[int(i)] for i in rng.permutation(len(rest))]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, columns, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def export_csvs(store, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for r in store.readers():
        if r["role"] == "reader":
            p = out_dir / f"labels_{r['reader_id']}.csv"
            _write_csv(p, LABEL_COLUMNS, store.label_rows(r["reader_id"]))
            written.append(p)
    _write_csv(out_dir / "adjudications.csv", ADJ_COLUMNS, store.adjudication_rows())
    try:
        final = store.final_labels()
    except ValueError:                       # fewer than two readers yet
        final = []
    _write_csv(out_dir / "final_labels.csv", FINAL_COLUMNS, final)
    return written + [out_dir / "adjudications.csv", out_dir / "final_labels.csv"]


def seal(final_labels_csv, lesion_fold, out_dir, manifest_path, now=None):
    """Split final_labels.csv by outer fold into out_dir/labels_fold{k}.csv and write the sha256 manifest. One shot."""
    with open(final_labels_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    pending = [r for r in rows if r["status"] == "pending"]
    if pending:
        raise ValueError(f"{len(pending)} lesions are pending adjudication; seal after the adjudicator has finished")
    out_dir, manifest_path = Path(out_dir), Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; sealing is one-shot")
    k = max(lesion_fold.values()) + 1
    per = {f: [] for f in range(k)}
    for r in rows:
        per[lesion_fold[int(r["lesion_id"])]].append(r)
    targets = [out_dir / f"labels_fold{f}.csv" for f in range(k)]
    for p in targets:
        if p.exists():
            raise FileExistsError(f"{p} exists; sealing is one-shot, never overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now or now_iso()
    manifest = {}
    for f, p in enumerate(targets):
        _write_csv(p, FINAL_COLUMNS, per[f])
        manifest[f"fold{f}"] = {"path": str(p), "sha256": sha256_file(p), "rows": len(per[f]), "sealed_at": stamp}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def backup_db(db_path, out_dir, now=None):
    """Consistent copy of the live sqlite file (sqlite backup API, safe while the server runs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or now_iso()).split("+")[0].replace(":", "-")        # 2026-10-01T08-00-00
    dst = out_dir / f"level_r_{stamp}.sqlite"
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    src, dest = sqlite3.connect(str(db_path)), sqlite3.connect(str(dst))
    with dest:
        src.backup(dest)
    src.close()
    dest.close()
    return dst
