#!/usr/bin/env python
# scripts/build_fastmri_knee.py
"""把 fastMRI+ 膝关节导出成 leg 2 的七视图训练树。

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/build_fastmri_knee.py --workers 4
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import (  # noqa: E402
    ANNOTATIONS, EXPORT_ROOT, clean_boxes, export_volume, make_folds, merge_to_3d,
    patient_of, read_annotations, volume_paths,
)

LESION_FIELDS = ["lesion_id", "file", "family", "z0", "z1", "x0", "y0", "x1", "y1", "n_boxes"]
MANIFEST_FIELDS = ["file", "out_dir", "slices", "n_lesions", "status"]


def job(args):
    name, path, lesions, out_root, seed = args
    os.nice(19)
    return export_volume(path, lesions, Path(out_root) / name, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="export only the first N volumes (smoke test)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = read_annotations(ANNOTATIONS)
    kept, dropped = clean_boxes(rows)
    lesions = merge_to_3d(kept)
    with open(a.out / "lesions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LESION_FIELDS)
        w.writeheader()
        for i, L in enumerate(sorted(lesions, key=lambda L: (L["file"], L["family"], L["z0"], L["x0"]))):
            w.writerow({"lesion_id": i, **{k: L[k] for k in LESION_FIELDS[1:]}})
    print(f"boxes kept {len(kept)}, dropped {dict(dropped)}; 3D lesions {len(lesions)}; "
          f"by family {dict(Counter(L['family'] for L in lesions))}")

    paths = volume_paths()
    names = sorted(paths)[:a.limit] if a.limit else sorted(paths)
    folds = make_folds(patient_of({n: paths[n] for n in names}))
    (a.out / "folds.json").write_text(json.dumps({"folds": folds}, indent=1))

    by_file = {}
    for L in lesions:
        by_file.setdefault(L["file"], []).append(L)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(job, [(n, paths[n], by_file.get(n, []), a.out, 1000 + i)
                                    for i, n in enumerate(names)]))
    with open(a.out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    bad = [r["file"] for r in results if r["status"] != "ok"]
    print(f"{len(results)} volumes, {len(bad)} failed: {bad[:10]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
