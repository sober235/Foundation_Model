"""Sealed Level R labels (v2.6 §12.7). Development code reads training folds; the outer test fold opens only with the
literal unblind=True and every such read is logged. The sha256 manifest lives in the repository, the CSVs on the
server, so neither can drift without the other noticing."""
import csv
import inspect
import json
from pathlib import Path

from anatobind.level_r.admin import sha256_file
from anatobind.level_r.store import now_iso

SEALED_DIR = Path("/data2/congcong/data/FM_data/derived/level_r/sealed")
MANIFEST = Path("data/level_r/sealed_manifest.json")


class SealIntegrityError(RuntimeError):
    pass


class SealedAccessError(RuntimeError):
    pass


def _parse(r):
    return {"lesion_id": int(r["lesion_id"]), "status": r["status"], "primary_host": r["primary_host"] or None,
            "acceptable_hosts": json.loads(r["acceptable_hosts"] or "[]"), "not_a_lesion": r["not_a_lesion"] in ("True", "true", "1")}


def _manifest(manifest_path):
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _check_fold(k, manifest):
    """k must be a plain int naming a sealed fold: 5, -1, "0" or True would otherwise slip through the f != k filter
    of load_train_labels and hand out every fold, the test fold included."""
    if type(k) is not int or f"fold{k}" not in manifest:
        raise SealedAccessError(f"fold id {k!r} does not name a sealed fold (manifest has {sorted(manifest)})")


def load_fold(k, sealed_dir=SEALED_DIR, manifest_path=MANIFEST):
    manifest = _manifest(manifest_path)
    _check_fold(k, manifest)
    entry = manifest[f"fold{k}"]
    p = Path(sealed_dir) / f"labels_fold{k}.csv"
    if sha256_file(p) != entry["sha256"]:
        raise SealIntegrityError(f"{p}: sha256 differs from {manifest_path}")
    with open(p, newline="", encoding="utf-8") as fh:
        rows = [_parse(r) for r in csv.DictReader(fh)]
    if len(rows) != entry["rows"]:
        raise SealIntegrityError(f"{p}: {len(rows)} rows, manifest says {entry['rows']}")
    return rows


def load_train_labels(k, sealed_dir=SEALED_DIR, manifest_path=MANIFEST):
    manifest = _manifest(manifest_path)
    _check_fold(k, manifest)
    folds = sorted(int(name[4:]) for name in manifest)
    return [r for f in folds if f != k for r in load_fold(f, sealed_dir, manifest_path)]


def load_test_labels(k, unblind=False, sealed_dir=SEALED_DIR, manifest_path=MANIFEST, log_path=None):
    if unblind is not True:
        raise SealedAccessError(f"fold {k} is a sealed outer test fold (v2.6 §12.7); pass unblind=True only for the final evaluation")
    _check_fold(k, _manifest(manifest_path))
    caller = inspect.stack()[1].filename
    log = Path(log_path) if log_path else Path(sealed_dir) / "access_log.txt"
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"{now_iso()}\tfold{k}\t{caller}\n")
    return load_fold(k, sealed_dir, manifest_path)
