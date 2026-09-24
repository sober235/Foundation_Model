import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from anatobind.data_engine.fastmri_knee import LegacyBoxConvention, VIEWS, load_folds, load_lesions

HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding>'
    "<encodedSpace><matrixSize><x>64</x><y>40</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>28</x><y>17.5</y><z>4.5</z></fieldOfView_mm></encodedSpace>"
    "<reconSpace><matrixSize><x>32</x><y>32</y><z>1</z></matrixSize>"
    "<fieldOfView_mm><x>14</x><y>14</y><z>3</z></fieldOfView_mm></reconSpace>"
    "</encoding></ismrmrdHeader>"
)
SIZE = 32


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/relink_fastmri_knee_gate0.py"
    spec = importlib.util.spec_from_file_location("relink_gate0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _h5(path, patient, slices=4):
    with h5py.File(path, "w") as h:
        h.create_dataset("kspace", data=np.zeros((slices, 2, 64, 40), np.complex64))
        h.create_dataset("reconstruction_rss", data=np.zeros((slices, SIZE, SIZE), np.float32))
        h.create_dataset("ismrmrd_header", data=np.bytes_(HEADER.encode()))
        h.attrs["patient_id"] = patient
        h.attrs["acquisition"] = "CORPDFS_FBK"
    return path


def _legacy_export(root, files, patients, slices=4):
    """The 2026-09-14 layout: images + meta.json per volume, as-is lesions.csv, folds.json, five-column manifest."""
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for f, p in zip(files, patients):
        d = root / f
        d.mkdir()
        for v in VIEWS:
            np.save(d / f"{v}.npy", rng.normal(size=(slices, SIZE, SIZE)).astype(np.float16))
        (d / "meta.json").write_text(json.dumps({"patient_id": p, "acquisition": "CORPDFS_FBK", "slices": slices,
                                                 "size": SIZE, "n_lesions": 1, "seed": 1}))
    (root / "lesions.csv").write_text("lesion_id,file,family,z0,z1,x0,y0,x1,y1,n_boxes\n"
                                      f"0,{files[0]},effusion,1,2,4,6,14,18,2\n")        # CSV frame: y from the bottom
    (root / "folds.json").write_text(json.dumps({"folds": {files[0]: 0, files[1]: 1}}))
    (root / "manifest.csv").write_text("file,out_dir,slices,n_lesions,status\n" +
                                       "".join(f"{f},{root / f},{slices},{int(f == files[0])},ok\n" for f in files))


def _annotations(path, file):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "slice", "study_level", "x", "y", "width", "height", "label"])
        w.writeheader()
        w.writerow({"file": file, "slice": 1, "study_level": "No", "x": 4, "y": 6, "width": 10, "height": 12, "label": "Joint Effusion "})
        w.writerow({"file": file, "slice": 2, "study_level": "No", "x": 4, "y": 6, "width": 10, "height": 12, "label": "Joint Effusion "})


def _tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_relink_flips_rows_links_volumes_and_leaves_the_legacy_root_untouched(tmp_path):
    files, patients = ["file1", "file2"], ["pa", "pb"]
    paths = {f: _h5(tmp_path / f"{f}.h5", p) for f, p in zip(files, patients)}
    legacy = tmp_path / "leg2"
    _legacy_export(legacy, files, patients)
    ann = tmp_path / "knee.csv"
    _annotations(ann, "file1")
    before = _tree_hash(legacy)

    new = tmp_path / "leg2_gate0"
    summary = _load().relink(legacy, new, ann, paths)

    assert _tree_hash(legacy) == before                                   # nothing in the legacy root changed
    assert (new / "file1").is_symlink() and (new / "file1" / "clean.npy").exists()
    lesions = load_lesions(new)                                           # loads: manifest says transform_version 2
    assert len(lesions) == 1
    L = lesions[0]
    assert (L["x0"], L["x1"]) == (4, 14)
    assert (L["y0"], L["y1"]) == (SIZE - 18, SIZE - 6)                    # rows flipped: [nr - y - h, nr - y)
    assert load_folds(new) == {"file1": 0, "file2": 1}
    assert summary["n_volumes"] == 2 and summary["n_lesions"] == 1 and summary["legacy_match"] == 1
    with pytest.raises(LegacyBoxConvention):
        load_lesions(legacy)


def test_relink_refuses_when_the_folds_would_change(tmp_path):
    files, patients = ["file1", "file2"], ["pa", "pb"]
    paths = {f: _h5(tmp_path / f"{f}.h5", p) for f, p in zip(files, patients)}
    legacy = tmp_path / "leg2"
    _legacy_export(legacy, files, patients)
    (legacy / "folds.json").write_text(json.dumps({"folds": {"file1": 1, "file2": 0}}))   # not what make_folds gives
    ann = tmp_path / "knee.csv"
    _annotations(ann, "file1")
    with pytest.raises(ValueError, match="folds"):
        _load().relink(legacy, tmp_path / "new", ann, paths)
