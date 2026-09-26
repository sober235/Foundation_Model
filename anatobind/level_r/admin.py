"""Admin operations behind scripts/level_r_admin.py: patient folds (v2.6 §4.4 / §12.6), per-reader orders, CSV export,
one-shot sealing with a sha256 manifest (v2.6 §12.7) and online sqlite backups. File policy: every export goes into a new
out_dir/<timestamp>/ directory, sealing and backups are one-shot, and each of them raises FileExistsError rather than
write over an existing directory or file."""
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from urllib.parse import quote

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
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def _stamp(now):
    return (now or now_iso()).split("+")[0].replace(":", "-")         # 2026-10-01T08-00-00


def export_csvs(store, out_dir, now=None):
    """labels_<reader>.csv (full history), adjudications.csv and final_labels.csv in a new out_dir/<timestamp>/.
    Raises ValueError unless the store has exactly two readers, FileExistsError if that timestamp directory exists."""
    final = store.final_labels()
    out_dir = Path(out_dir) / _stamp(now)
    out_dir.mkdir(parents=True)
    written = []
    for r in store.readers():
        if r["role"] == "reader":
            p = out_dir / f"labels_{r['reader_id']}.csv"
            _write_csv(p, LABEL_COLUMNS, store.label_rows(r["reader_id"]))
            written.append(p)
    _write_csv(out_dir / "adjudications.csv", ADJ_COLUMNS, store.adjudication_rows())
    _write_csv(out_dir / "final_labels.csv", FINAL_COLUMNS, final)
    return written + [out_dir / "adjudications.csv", out_dir / "final_labels.csv"]


def seal(final_labels_csv, lesion_fold, out_dir, manifest_path, k, now=None):
    """Split final_labels.csv by outer fold into out_dir/labels_fold{0..k-1}.csv and write the sha256 manifest. One shot.
    k is the fold count of the fold table; the CSV must hold every lesion of lesion_fold exactly once, none pending."""
    with open(final_labels_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pending = [r for r in rows if r["status"] == "pending"]
    if pending:
        raise ValueError(f"{len(pending)} lesions are pending adjudication; seal after the adjudicator has finished")
    ids = Counter(int(r["lesion_id"]) for r in rows)
    missing, extra = set(lesion_fold) - set(ids), set(ids) - set(lesion_fold)
    duplicated = [i for i, n in ids.items() if n > 1]
    if missing or extra or duplicated:
        raise ValueError(f"{final_labels_csv} must hold each of the {len(lesion_fold)} lesions of the fold table exactly once: "
                         f"{len(missing)} missing, {len(extra)} extra, {len(duplicated)} duplicated")
    out_dir, manifest_path = Path(out_dir), Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; sealing is one-shot")
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
    """Consistent copy of the live sqlite file (sqlite backup API, safe while the server runs). The source is opened
    read-only, so a mistyped path raises sqlite3.OperationalError instead of backing up a new empty database."""
    src = sqlite3.connect(f"file:{quote(str(db_path))}?mode=ro", uri=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"level_r_{_stamp(now)}.sqlite"
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    dest = sqlite3.connect(str(dst))
    with dest:
        src.backup(dest)
    src.close()
    dest.close()
    return dst
