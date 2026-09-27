# tests/test_relation_build.py
import json

import numpy as np
import pytest

from anatobind.eval.geometry import CLASS_NAMES
from anatobind.relation.build import BuildError, build_table, check_rows, lesion_row, volume_context
from anatobind.relation.table import ALL_COLUMNS, load_table
from synth_relation import synthetic_registry, synthetic_volume

FILES = ["file_brain_AXFLAIR_200_1", "file_brain_AXFLAIR_200_2", "file_brain_AXFLAIR_200_3"]


def _loader(members_of):
    def load_volume(f):
        seg, sp, rss = synthetic_volume(0)
        return seg, sp, rss, members_of[f]
    return load_volume


def _build(tmp_path, per_file=3, mutate=None):
    reg, members = synthetic_registry(FILES, per_file)
    if mutate:
        mutate(reg)
    folds = {f"patient_{i}": i for i in range(len(FILES))}
    return build_table(reg, folds, _loader(members), tmp_path / "v1", {"registry_sha256": "x", "folds_sha256": "y"}, log=lambda *_: None)


def test_lesion_row_has_every_column_and_matches_the_registry_geometry():
    reg, members = synthetic_registry(FILES[:1], 2)
    seg, sp, _ = synthetic_volume(0)
    ctx = volume_context(seg, sp)
    row = lesion_row(reg[0], members[FILES[0]][0], ctx, 0)
    assert set(row) == set(ALL_COLUMNS)
    assert row["d_interface_mm"] == pytest.approx(reg[0]["d_interface_mm"]) and row["d1_mm"] == pytest.approx(reg[0]["d1_mm"])
    assert row["c1_class"] in CLASS_NAMES and row["side"] in ("image_left", "image_right", "midline")
    assert row["coarse_location"] == f"{row['side']}|{row['row_third']}|{row['slice_third']}"
    assert row["volume_mm3"] == pytest.approx(16 * 0.5 * 0.5 * 5.0) and row["extent_z_mm"] == 5.0
    assert row["lesion_type"] == "lacunar_infarct"                                  # lesion 0: lid % 5 == 0


def test_build_writes_table_patches_and_manifest_with_the_checks(tmp_path):
    rows, patches, manifest = _build(tmp_path)
    t, p = load_table(tmp_path / "v1")
    assert len(t) == 9 and p["image"].shape == (9, 3, 48, 48) and t.lesion_id.tolist() == list(range(9))
    m = json.loads((tmp_path / "v1" / "manifest.json").read_text())
    assert m["n_lesions"] == 9 and m["n_patients"] == 3 and m["c1_agreement"]["fraction"] == 1.0
    assert m["per_fold"]["0"]["lesions"] == 3 and m["registry_sha256"] == "x" and m["parameters"]["candidate_mm"] == 15.0
    assert sum(m["c1_distribution"].values()) == 9


def test_build_refuses_a_patient_missing_from_the_fold_table(tmp_path):
    reg, members = synthetic_registry(FILES, 2)
    with pytest.raises(BuildError, match="fold"):
        build_table(reg, {"patient_0": 0}, _loader(members), tmp_path / "v1", {}, log=lambda *_: None)
    assert not (tmp_path / "v1").exists()


def test_build_refuses_when_c1_disagrees_with_the_registry_too_often(tmp_path):
    def corrupt(reg):
        for r in reg:
            r["host_lookup_parenchyma"] = "16"                                     # brainstem for everyone
    with pytest.raises(BuildError, match="C1"):
        _build(tmp_path, mutate=corrupt)


def test_build_refuses_when_a_registry_distance_differs(tmp_path):
    def corrupt(reg):
        reg[0]["d_interface_mm"] += 0.5
    with pytest.raises(BuildError, match="d_interface"):
        _build(tmp_path, mutate=corrupt)


def test_check_rows_lists_mismatches_below_the_threshold(tmp_path):
    rows, patches, manifest = _build(tmp_path)
    rows[0]["registry_lookup_class"] = "brainstem"
    report = check_rows(rows, [], None, patches, min_agreement=0.5)
    assert report["c1_agreement"]["mismatches"] == [0] and report["c1_agreement"]["fraction"] == pytest.approx(8 / 9)


def _break(case, rows, patches, reg):
    if case == "duplicate":
        rows[1]["lesion_id"] = rows[0]["lesion_id"]
    elif case == "registry_set":
        reg.append({**reg[0], "lesion_id": 999})
    elif case == "two_folds":
        rows[1]["patient_id"], rows[1]["fold"] = rows[0]["patient_id"], rows[0]["fold"] + 1
    elif case == "no_candidate":
        for k in rows[0]:
            if k.endswith("_candidate"):
                rows[0][k] = False
    elif case == "constant_patch":
        patches["image"][0] = 0
    return rows, patches, reg


@pytest.mark.parametrize("case,match", [("duplicate", "duplicate"), ("registry_set", "registry"), ("two_folds", "fold"),
                                        ("no_candidate", "candidate"), ("constant_patch", "constant")])
def test_check_rows_refuses_each_broken_invariant(tmp_path, case, match):
    rows, patches, _ = _build(tmp_path)
    reg, _ = synthetic_registry(FILES, 3)
    rows, patches, reg = _break(case, rows, dict(patches, image=patches["image"].copy()), reg)
    with pytest.raises(BuildError, match=match):
        check_rows(rows, reg if case == "registry_set" else [], None, patches, min_agreement=0.0)
