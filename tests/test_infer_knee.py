import csv
import json

import nibabel as nib
import numpy as np

from anatobind.infer.knee import CSV_FIELDS, lesion_table, render_overlay, run, write_table

SP = (0.625, 0.625, 0.8)


def _anatomy():
    a = np.zeros((40, 40, 20), np.uint8)
    a[5:20, 5:30, 2:12] = 2     # femoral cartilage
    a[22:36, 5:30, 4:10] = 5    # medial meniscus
    return a


def test_lesion_table_names_the_host_and_keeps_the_box():
    dets = [{"family": "Cartilage Lesion", "label": 1, "box": (8, 8, 3, 12, 14, 8), "score": 0.9, "n_voxels": 120},
            {"family": "Effusion", "label": 4, "box": (0, 0, 0, 3, 3, 3), "score": 0.4, "n_voxels": 27}]
    rows = lesion_table(dets, _anatomy(), SP)
    assert [r["lesion_id"] for r in rows] == [1, 2] and list(rows[0]) == CSV_FIELDS
    assert rows[0]["host_label"] == 2 and rows[0]["host_name"] == "femoral_cartilage" and rows[0]["side"] == "single"
    assert json.loads(rows[0]["host_fractions"])["2"] == 1.0 and rows[0]["x0"] == 8 and rows[0]["z1"] == 8
    assert rows[1]["host_name"] == "none" and rows[1]["host_label"] == "" and rows[1]["host_fractions"] == "{}"


def test_write_table_and_overlay_produce_files(tmp_path):
    rows = lesion_table([{"family": "Meniscal Tear", "label": 2, "box": (25, 10, 5, 30, 20, 9), "score": 0.7, "n_voxels": 200}],
                        _anatomy(), SP)
    write_table(rows, tmp_path / "lesions.csv")
    with open(tmp_path / "lesions.csv", newline="") as fh:
        back = list(csv.DictReader(fh))
    assert back[0]["host_name"] == "meniscus_medial" and back[0]["side"] == "medial"
    render_overlay(np.random.default_rng(0).random((40, 40, 20), np.float32), _anatomy(), (25, 10, 5, 30, 20, 9), tmp_path / "o.png")
    assert (tmp_path / "o.png").stat().st_size > 1000


def _fake_predict(anatomy, lesion):
    def predict(dataset_id, in_dir, out_dir, folds, gpu, save_probabilities):
        out_dir.mkdir(parents=True, exist_ok=True)
        affine = nib.load(str(in_dir / "case_0000.nii.gz")).affine
        lab = anatomy if dataset_id == 901 else lesion
        nib.save(nib.Nifti1Image(lab, affine), str(out_dir / "case.nii.gz"))
        if save_probabilities:
            p = np.zeros((5,) + lab.shape, np.float32)
            for c in range(5):
                p[c] = lab == c
            np.savez_compressed(out_dir / "case.npz", probabilities=np.ascontiguousarray(p.transpose(0, 3, 2, 1)))
    return predict


def test_run_writes_the_four_outputs_from_an_h5_frame_volume(tmp_path):
    anatomy = _anatomy()
    lesion = np.zeros_like(anatomy)
    lesion[8:12, 8:14, 3:8] = 1
    image = np.random.default_rng(0).random(anatomy.shape, np.float32)
    nib.save(nib.Nifti1Image(image, np.diag([0.625, 0.625, 0.8, 1.0])), str(tmp_path / "vol.nii.gz"))
    rows = run(tmp_path / "vol.nii.gz", tmp_path / "out", "h5", [0], 0, predict=_fake_predict(anatomy, lesion))
    for name in ("anatomy.nii.gz", "lesions.nii.gz", "lesions.csv", "overlay.png"):
        assert (tmp_path / "out" / name).exists()
    assert len(rows) == 1 and rows[0]["family"] == "Cartilage Lesion" and rows[0]["host_name"] == "femoral_cartilage"
    assert np.array_equal(np.asanyarray(nib.load(str(tmp_path / "out" / "anatomy.nii.gz")).dataobj), anatomy)


def test_run_reorients_a_world_frame_volume_before_predicting(tmp_path):
    anatomy = _anatomy()
    seen = {}

    def predict(dataset_id, in_dir, out_dir, folds, gpu, save_probabilities):
        img = nib.load(str(in_dir / "case_0000.nii.gz"))
        seen["axcodes"], seen["shape"] = nib.aff2axcodes(img.affine), img.shape
        _fake_predict(anatomy, np.zeros_like(anatomy))(dataset_id, in_dir, out_dir, folds, gpu, save_probabilities)

    # an RAS volume with 0.8 mm along its first axis: after reordering to (I, P, R) it is 40 x 40 x 20
    nib.save(nib.Nifti1Image(np.zeros((20, 40, 40), np.float32), np.diag([0.8, 0.625, 0.625, 1.0])), str(tmp_path / "ras.nii.gz"))
    rows = run(tmp_path / "ras.nii.gz", tmp_path / "out", "world", [0], 0, predict=predict)
    assert seen["axcodes"] == ("I", "P", "R") and seen["shape"] == (40, 40, 20) and rows == []
