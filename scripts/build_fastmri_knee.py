#!/usr/bin/env python
# scripts/build_fastmri_knee.py
"""把 fastMRI+ 膝关节导出成 leg 2 的七视图训练树。

  PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/build_fastmri_knee.py --workers 4
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import rows_to_rss_frame  # noqa: E402
from anatobind.data_engine.fastmri_knee import (  # noqa: E402
    ANNOTATIONS, EXPORT_ROOT, assert_folds_by_patient, clean_boxes, export_volume, make_folds, manifest_row,
    merge_to_3d, read_annotations, volume_geometry, volume_paths, write_lesions, write_manifest,
)


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

    paths = volume_paths()
    names = sorted(paths)[:a.limit] if a.limit else sorted(paths)
    geo = {n: volume_geometry(paths[n]) for n in names}
    kept, dropped = clean_boxes(read_annotations(ANNOTATIONS))
    kept = [r for r in kept if r["file"] in geo]                       # boxes of volumes that are not on disk cannot be exported
    lesions = merge_to_3d(rows_to_rss_frame(kept, {n: g["n_rows"] for n, g in geo.items()}))
    write_lesions(a.out / "lesions.csv", lesions)
    print(f"boxes kept {len(kept)}, dropped {dict(dropped)}; 3D lesions {len(lesions)}; "
          f"by family {dict(Counter(L['family'] for L in lesions))}")

    patients = {n: geo[n]["patient_id"] for n in names}
    folds = make_folds(patients)
    assert_folds_by_patient(folds, patients)
    (a.out / "folds.json").write_text(json.dumps({"folds": folds}, indent=1))

    by_file = {}
    for L in lesions:
        by_file.setdefault(L["file"], []).append(L)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(job, [(n, paths[n], by_file.get(n, []), a.out, 1000 + i) for i, n in enumerate(names)]))
    rows = [manifest_row(r["file"], paths[r["file"]], r["out_dir"], r["n_lesions"], r["status"]) for r in results]
    write_manifest(a.out / "manifest.csv", rows)
    bad = [r["file"] for r in results if r["status"] != "ok"]
    print(f"{len(results)} volumes, {len(bad)} failed: {bad[:10]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
