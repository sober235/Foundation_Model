import csv
import json

import nibabel as nib
import numpy as np
import pytest

from anatobind.train.dataset import (
    UNKNOWN_HOST,
    SkmteaArmBDataset,
    box_to_model_frame,
    list_ready_scans,
    load_fold,
    normalise_volume,
    to_model_frame,
)

MANIFEST_FIELDS = [
    "scan_id", "out_dir", "n_boxes_kept", "n_ambiguous", "n_no_overlap",
    "n_unresolved", "spacing", "nrmse", "files", "status", "seconds",
]
BOX_FIELDS = [
    "ann_id", "split", "layer", "supercategory", "category_id", "tissue_id",
    "host_label", "host_side", "host_ratio",
    "x0", "y0", "z0", "x1", "y1", "z1",
    "x0_full", "y0_full", "z0_full", "x1_full", "y1_full", "z1_full",
]
SPACING = (0.625, 0.625, 0.8)


def _write_scan(root, scan_id, boxes, shape=(32, 40, 24), with_boxes_csv=True):
    """Write one synthetic export folder in the real (X, Y, Z) layout."""
    d = root / scan_id
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(scan_id)) % 2**32)
    affine = np.diag([*SPACING, 1.0])
    img = rng.gamma(2.0, 1e6, size=shape).astype(np.float32)
    nib.save(nib.Nifti1Image(img, affine), d / "image_clean_e1.nii.gz")

    seg = np.zeros(shape, dtype=np.uint8)
    for row in boxes:  # paint the host structure over its own box
        if str(row["host_label"]).strip():  # an unresolved host paints nothing
            seg[row["x0"]:row["x1"], row["y0"]:row["y1"], row["z0"]:row["z1"]] = row["host_label"]
    nib.save(nib.Nifti1Image(seg, affine), d / "seg.nii.gz")

    if with_boxes_csv:
        with open(d / "boxes.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=BOX_FIELDS)
            w.writeheader()
            for row in boxes:
                full = {f"{k}_full": row[k] * (2 if k[0] in "xy" else 1) for k in
                        ("x0", "y0", "z0", "x1", "y1", "z1")}
                w.writerow({**row, **full})
    return d


def _write_export(tmp_path, scans, ready=None, folds=None):
    """scans: {scan_id: [box dicts]}.  ready: subset that gets a boxes.csv."""
    ready = set(scans) if ready is None else set(ready)
    for scan_id, boxes in scans.items():
        _write_scan(tmp_path, scan_id, boxes, with_boxes_csv=scan_id in ready)
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for scan_id in scans:
            w.writerow({
                "scan_id": scan_id, "out_dir": str(tmp_path / scan_id),
                "n_boxes_kept": len(scans[scan_id]), "n_ambiguous": 0,
                "n_no_overlap": 0, "n_unresolved": 0,
                "spacing": ";".join(f"{s:.5f}" for s in SPACING),
                "nrmse": "", "files": "", "status": "ok", "seconds": "1.0",
            })
    if folds is None:
        folds = {s: i % 5 for i, s in enumerate(scans)}
    (tmp_path / "splits.json").write_text(json.dumps({"seed": 0, "folds": folds}))


def _box(ann_id, host_label, x0, y0, z0, x1, y1, z1, super_="Meniscal Tear", side="medial"):
    return {
        "ann_id": ann_id, "split": "train", "layer": "in_seg", "supercategory": super_,
        "category_id": 5, "tissue_id": 1, "host_label": host_label,
        "host_side": side, "host_ratio": "1.000",
        "x0": x0, "y0": y0, "z0": z0, "x1": x1, "y1": y1, "z1": z1,
    }


# --- frame conversion -------------------------------------------------------

def test_to_model_frame_maps_xyz_to_zyx():
    a = np.zeros((4, 6, 8), dtype=np.float32)
    a[1, 2, 3] = 1.0
    b = to_model_frame(a)
    assert b.shape == (8, 6, 4)
    assert b[3, 2, 1] == 1.0


def test_box_permutation_is_the_inverse_of_the_array_permutation():
    arr = np.zeros((10, 12, 14), dtype=np.float32)
    x0, y0, z0, x1, y1, z1 = 2, 3, 4, 6, 9, 11
    arr[x0:x1, y0:y1, z0:z1] = 1.0
    m = to_model_frame(arr)
    bz0, by0, bx0, bz1, by1, bx1 = box_to_model_frame((x0, y0, z0, x1, y1, z1))
    assert m[bz0:bz1, by0:by1, bx0:bx1].min() == 1.0
    assert m.sum() == (bz1 - bz0) * (by1 - by0) * (bx1 - bx0)


# --- normalisation ----------------------------------------------------------

def test_normalise_is_scale_invariant_and_zero_mean():
    rng = np.random.default_rng(0)
    v = rng.gamma(2.0, 1e6, size=(8, 8, 8)).astype(np.float32)
    a, b = normalise_volume(v), normalise_volume(v * 7.0)
    assert abs(float(a.mean())) < 1e-3
    assert abs(float(a.std()) - 1.0) < 0.05
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_normalise_clips_the_extreme_tail():
    v = np.ones((10, 10, 10), dtype=np.float32)
    v[0, 0, 0] = 1e9
    out = normalise_volume(v)
    assert float(out.max()) < 10.0


# --- manifest and split -----------------------------------------------------

def test_list_ready_scans_skips_rows_without_boxes_csv(tmp_path):
    _write_export(tmp_path, {"MTR_001": [_box(1, 5, 2, 2, 2, 6, 8, 6)],
                             "MTR_002": [_box(2, 2, 2, 2, 2, 6, 8, 6)]},
                  ready={"MTR_001"})
    assert list_ready_scans(tmp_path) == ["MTR_001"]


def test_load_fold_uses_the_folds_mapping(tmp_path):
    scans = {f"MTR_{i:03d}": [_box(i, 5, 2, 2, 2, 6, 8, 6)] for i in range(1, 6)}
    folds = {"MTR_001": 0, "MTR_002": 0, "MTR_003": 1, "MTR_004": 2, "MTR_005": 3}
    _write_export(tmp_path, scans, folds=folds)
    train, val = load_fold(tmp_path, 0)
    assert val == ["MTR_001", "MTR_002"]
    assert train == ["MTR_003", "MTR_004", "MTR_005"]


def test_load_fold_drops_scans_that_are_not_ready(tmp_path):
    scans = {f"MTR_{i:03d}": [_box(i, 5, 2, 2, 2, 6, 8, 6)] for i in range(1, 4)}
    _write_export(tmp_path, scans, ready={"MTR_001", "MTR_003"},
                  folds={"MTR_001": 0, "MTR_002": 1, "MTR_003": 1})
    train, val = load_fold(tmp_path, 0)
    assert val == ["MTR_001"]
    assert train == ["MTR_003"]


# --- dataset ----------------------------------------------------------------

@pytest.fixture
def export(tmp_path):
    boxes = [_box(1, 5, 4, 6, 3, 12, 18, 9),
             _box(2, 2, 18, 22, 12, 26, 32, 18, super_="Cartilage Lesion")]
    _write_export(tmp_path, {"MTR_001": boxes, "MTR_002": boxes})
    return tmp_path


def test_sample_shapes_and_dtypes(export):
    ds = SkmteaArmBDataset(["MTR_001"], export, patch=(8, 16, 16), train=True, seed=0)
    s = ds[0]
    assert s["image"].shape == (1, 8, 16, 16)
    assert s["seg"].shape == (8, 16, 16)
    assert s["image"].dtype == np.float32
    assert s["present"].shape == (6,)


def test_spacing_is_returned_in_model_frame_order(export):
    ds = SkmteaArmBDataset(["MTR_001"], export, patch=(8, 16, 16), train=False, seed=0)
    np.testing.assert_allclose(ds[0]["spacing_mm"], np.array(SPACING[::-1]), rtol=1e-5)


def test_crop_stays_inside_the_volume_and_keeps_the_seed_box(export):
    ds = SkmteaArmBDataset(["MTR_001"], export, patch=(8, 16, 16), train=True, seed=0)
    for _ in range(20):
        s = ds[0]
        d, h, w = s["image"].shape[1:]
        assert len(s["boxes"]) >= 1
        b = s["boxes"]
        assert b[:, :3].min() >= 0
        assert (b[:, 3] <= d).all() and (b[:, 4] <= h).all() and (b[:, 5] <= w).all()
        assert (b[:, 3:] > b[:, :3]).all()


def test_kept_boxes_carry_host_label_and_class(export):
    ds = SkmteaArmBDataset(["MTR_001"], export, patch=(24, 40, 32), train=False, seed=0)
    s = ds[0]
    assert set(s["host_label"].tolist()) <= {1, 2, 3, 4, 5, 6}
    assert set(s["box_classes"].tolist()) <= {0, 1}
    assert s["host_label"].shape == s["box_classes"].shape


def test_boxes_land_on_their_host_structure_in_the_model_frame(export):
    """The permutation must keep every box on top of its own segmentation label."""
    ds = SkmteaArmBDataset(["MTR_001"], export, patch=(24, 40, 32), train=False, seed=0)
    s = ds[0]
    for box, host in zip(s["boxes"].astype(int), s["host_label"]):
        z0, y0, x0, z1, y1, x1 = box
        patch = s["seg"][z0:z1, y0:y1, x0:x1]
        assert (patch == host).any(), f"box does not overlap host label {host}"


def test_present_flags_match_the_cropped_segmentation(export):
    ds = SkmteaArmBDataset(["MTR_001"], export, patch=(24, 40, 32), train=False, seed=0)
    s = ds[0]
    for k in range(6):
        assert bool(s["present"][k]) == bool((s["seg"] == k + 1).any())


def test_validation_sampling_is_deterministic(export):
    ds = SkmteaArmBDataset(["MTR_001"], export, patch=(8, 16, 16), train=False, seed=0)
    np.testing.assert_array_equal(ds[0]["image"], ds[0]["image"])


def test_an_unresolved_host_is_kept_as_unknown_not_as_a_label(tmp_path):
    """host_side 'unresolved' leaves host_label empty: the truth is UNKNOWN.

    Spec 4.3 / decision 2A keep present, confirmed-absent and unknown apart.  The
    box still supervises U_B, but no host may be invented for it.
    """
    boxes = [_box(1, "", 4, 6, 3, 12, 18, 9, side="unresolved"),
             _box(2, 2, 18, 22, 12, 26, 32, 18, super_="Cartilage Lesion",
                  side="single_no_overlap")]
    _write_export(tmp_path, {"MTR_001": boxes})
    ds = SkmteaArmBDataset(["MTR_001"], tmp_path, patch=(24, 40, 32), train=False, seed=0)
    s = ds[0]
    assert len(s["boxes"]) == 2, "an unresolved host must not drop the detection target"
    assert s["host_label"].tolist() == [UNKNOWN_HOST, 2]


def test_collate_stacks_dense_fields_and_keeps_boxes_ragged(export):
    from anatobind.train.dataset import collate

    ds = SkmteaArmBDataset(["MTR_001", "MTR_002"], export, patch=(8, 16, 16),
                           train=False, seed=0)
    batch = collate([ds[0], ds[1]])
    assert batch["image"].shape == (2, 1, 8, 16, 16)
    assert batch["seg"].shape == (2, 8, 16, 16)
    assert batch["present"].shape == (2, 6)
    assert batch["spacing_mm"].shape == (2, 3)
    assert len(batch["boxes"]) == len(batch["host_label"]) == 2
    assert batch["scan_id"] == ["MTR_001", "MTR_002"]
