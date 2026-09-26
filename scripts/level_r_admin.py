#!/usr/bin/env python
"""Level R administration (spec §3, §7, §10). Subcommands, all one-shot and none overwriting:

  folds       registry -> data/level_r/folds.json (165 patients, seed 0, five folds)
  init        create the sqlite database and load lesions.json into it
  add-reader  create a reader or adjudicator; the token is printed exactly once
  order       give one reader their randomised order (pilot first)
  export      labels_<reader>.csv / adjudications.csv / final_labels.csv
  seal        final_labels.csv -> sealed/labels_fold{k}.csv + data/level_r/sealed_manifest.json
  backup      timestamped copy of the live database

  D=/data2/congcong/data/FM_data/derived/level_r
  python scripts/level_r_admin.py folds --out data/level_r/folds.json
  python scripts/level_r_admin.py init --db $D/level_r.sqlite --lesions $D/lesions.json
  python scripts/level_r_admin.py add-reader --db $D/level_r.sqlite --reader-id r1 --role reader --display "读者 1"
  python scripts/level_r_admin.py order --db $D/level_r.sqlite --reader-id r1 --seed 1 --pilot data/level_r/pilot_150.json
  python scripts/level_r_admin.py export --db $D/level_r.sqlite --out $D/export
  python scripts/level_r_admin.py seal --final $D/export/final_labels.csv --folds data/level_r/folds.json --out $D/sealed
  python scripts/level_r_admin.py backup --db $D/level_r.sqlite --out $D/backup
"""
import argparse
import json
import secrets
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.level_r.admin import backup_db, export_csvs, lesion_folds, make_order, make_patient_folds, seal  # noqa: E402
from anatobind.level_r.registry import REGISTRY, load_registry  # noqa: E402
from anatobind.level_r.store import Store  # noqa: E402

MANIFEST = Path("data/level_r/sealed_manifest.json")


def cmd_folds(a):
    if a.out.exists():
        sys.exit(f"{a.out} exists; folds are generated once")
    registry = load_registry(a.registry)
    pf = make_patient_folds([r["patient_id"] for r in registry], k=a.k, seed=a.seed)
    lf = lesion_folds(registry, pf)
    out = {"k": a.k, "seed": a.seed, "n_patients": len(pf), "patient_fold": pf,
           "lesions_per_fold": {str(k): v for k, v in sorted(Counter(lf.values()).items())}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"{len(pf)} patients, lesions per fold {out['lesions_per_fold']} -> {a.out}")


def cmd_init(a):
    store = Store(a.db)
    records = json.loads(a.lesions.read_text())
    store.load_lesions(records)
    print(f"{a.db}: {len(store.lesion_ids())} lesions loaded from {a.lesions}")


def cmd_add_reader(a):
    token = secrets.token_hex(8)
    Store(a.db).add_reader(a.reader_id, a.role, token, a.display)
    print(f"reader {a.reader_id} ({a.role}, {a.display}) created. Link (shown once, not stored):\n  /?token={token}")


def cmd_order(a):
    store = Store(a.db)
    pilot = set(json.loads(a.pilot.read_text())["lesion_ids"]) if a.pilot else set()
    order = make_order(store.lesion_ids(), pilot, a.seed)
    store.set_order(a.reader_id, order, pilot)
    print(f"{a.reader_id}: {len(order)} lesions, {len(pilot)} pilot first, seed {a.seed}")


def cmd_export(a):
    for p in export_csvs(Store(a.db), a.out):
        print(p)


def cmd_seal(a):
    folds = json.loads(a.folds.read_text())
    registry = load_registry(a.registry)
    lf = lesion_folds(registry, folds["patient_fold"])
    man = seal(a.final, lf, a.out, a.manifest)
    for k, m in man.items():
        print(f"{k}: {m['rows']} rows sha256 {m['sha256'][:12]}… -> {m['path']}")
    print(f"manifest -> {a.manifest}")


def cmd_backup(a):
    print(backup_db(a.db, a.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("folds"); p.add_argument("--registry", type=Path, default=REGISTRY); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=5); p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_folds)
    p = sub.add_parser("init"); p.add_argument("--db", type=Path, required=True); p.add_argument("--lesions", type=Path, required=True); p.set_defaults(fn=cmd_init)
    p = sub.add_parser("add-reader"); p.add_argument("--db", type=Path, required=True); p.add_argument("--reader-id", required=True)
    p.add_argument("--role", choices=("reader", "adjudicator"), required=True); p.add_argument("--display", required=True); p.set_defaults(fn=cmd_add_reader)
    p = sub.add_parser("order"); p.add_argument("--db", type=Path, required=True); p.add_argument("--reader-id", required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--pilot", type=Path); p.set_defaults(fn=cmd_order)
    p = sub.add_parser("export"); p.add_argument("--db", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.set_defaults(fn=cmd_export)
    p = sub.add_parser("seal"); p.add_argument("--final", type=Path, required=True); p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--registry", type=Path, default=REGISTRY); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST); p.set_defaults(fn=cmd_seal)
    p = sub.add_parser("backup"); p.add_argument("--db", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.set_defaults(fn=cmd_backup)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
