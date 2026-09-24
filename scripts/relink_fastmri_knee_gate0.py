#!/usr/bin/env python
# scripts/relink_fastmri_knee_gate0.py
"""Gate 0 for the leg 2 knee export (v2.6 §4.3): a new export root whose volume directories are relative links to
the 2026-09-14 images and whose lesions.csv is in the RSS frame. The legacy root is read, never written.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/relink_fastmri_knee_gate0.py
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri import TRANSFORM_VERSION, rows_to_rss_frame  # noqa: E402
from anatobind.data_engine.fastmri_knee import (  # noqa: E402
    ANNOTATIONS, EXPORT_ROOT, LEGACY_EXPORT_ROOT, assert_folds_by_patient, clean_boxes, make_folds, manifest_row,
    merge_to_3d, read_annotations, volume_geometry, volume_paths, write_lesions, write_manifest,
)


def check_against_legacy(legacy_csv, lesions, n_rows_of):
    """Every legacy lesion of a volume on disk must reappear with the same file/family/z/x/n_boxes and rows
    mapped by row0' = nr - row1, row1' = nr - row0. Returns the number of matched lesions."""
    with open(legacy_csv, newline="") as fh:
        old = [r for r in csv.DictReader(fh) if r["file"] in n_rows_of]

    def key(d):
        return (d["file"], d["family"], int(d["z0"]), int(d["z1"]), int(d["x0"]), int(d["x1"]), int(d["n_boxes"]))

    a = sorted((key(r), n_rows_of[r["file"]] - int(r["y1"]), n_rows_of[r["file"]] - int(r["y0"])) for r in old)
    b = sorted((key(L), L["y0"], L["y1"]) for L in lesions)
    if a != b:
        diff = next(((x, y) for x, y in zip(a, b) if x != y), None)
        raise ValueError(f"legacy/new lesion mismatch: {len(a)} legacy vs {len(b)} new; first difference {diff}")
    return len(b)


def relink(legacy_root, new_root, annotations, paths):
    legacy_root, new_root = Path(legacy_root), Path(new_root)
    with open(legacy_root / "manifest.csv", newline="") as fh:
        legacy = {r["file"]: r for r in csv.DictReader(fh)}
    names = sorted(legacy)
    geo = {n: volume_geometry(paths[n]) for n in names}
    for n in names:
        meta = json.loads((legacy_root / n / "meta.json").read_text())
        if geo[n]["n_rows"] != meta["size"] or geo[n]["slices"] != meta["slices"]:
            raise ValueError(f"{n}: exported grid {meta['size']}x{meta['slices']} slices != reconstruction_rss "
                             f"{geo[n]['n_rows']} rows x {geo[n]['slices']} slices; the CSV rows were labelled on the RSS grid")
    n_rows_of = {n: geo[n]["n_rows"] for n in names}

    kept, dropped = clean_boxes(read_annotations(annotations))
    missing = sorted({r["file"] for r in kept} - set(n_rows_of))
    kept = [r for r in kept if r["file"] in n_rows_of]
    lesions = merge_to_3d(rows_to_rss_frame(kept, n_rows_of))
    matched = check_against_legacy(legacy_root / "lesions.csv", lesions, n_rows_of)

    patients = {n: geo[n]["patient_id"] for n in names}
    folds = make_folds(patients)
    assert_folds_by_patient(folds, patients)
    legacy_folds = json.loads((legacy_root / "folds.json").read_text())["folds"]
    if folds != legacy_folds:
        raise ValueError("the regenerated folds differ from the legacy folds.json; the H1 rerun must keep the same "
                         "patients per fold, stop and look")

    new_root.mkdir(parents=True, exist_ok=True)
    for n in names:
        link = new_root / n
        if not link.exists():
            os.symlink(os.path.relpath(legacy_root / n, new_root), link)
    write_lesions(new_root / "lesions.csv", lesions)
    (new_root / "folds.json").write_text(json.dumps({"folds": folds}, indent=1))
    by_file = Counter(L["file"] for L in lesions)
    write_manifest(new_root / "manifest.csv",
                   [manifest_row(n, paths[n], new_root / n, by_file.get(n, 0), legacy[n]["status"]) for n in names])
    (new_root / "README.txt").write_text(
        f"Gate 0 export (transform_version {TRANSFORM_VERSION}): volume directories link to {legacy_root}; only "
        f"lesions.csv (RSS frame, rows from the top), folds.json and manifest.csv are new. Built by "
        f"scripts/relink_fastmri_knee_gate0.py from {annotations}.\n")
    return {"n_volumes": len(names), "n_lesions": len(lesions), "legacy_match": matched, "dropped": dict(dropped),
            "files_without_volume": len(missing), "by_family": dict(Counter(L["family"] for L in lesions))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-root", type=Path, default=LEGACY_EXPORT_ROOT)
    ap.add_argument("--out", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    a = ap.parse_args()
    print(json.dumps(relink(a.legacy_root, a.out, a.annotations, volume_paths()), indent=1))


if __name__ == "__main__":
    main()
