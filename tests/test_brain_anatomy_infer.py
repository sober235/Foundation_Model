# tests/test_brain_anatomy_infer.py
import json

import nibabel as nib
import numpy as np
import pytest

import anatobind.infer.brain_anatomy as I
from anatobind.data_engine.fastmri import rss_affine


def _stack(tmp_path):
    data = np.zeros((40, 40, 8), np.float32)
    data[5:35, 5:35, :] = 200.0                        # "head"
    data[10:30, 10:30, :] = 400.0                      # "brain"
    aff = rss_affine(2.0, 2.0, 5.0, data.shape)          # 2 mm pixels: the toy brain covers more than 5 cm2 per slice
    p = tmp_path / "stack.nii.gz"
    nib.save(nib.Nifti1Image(data, aff), str(p))
    return p


def _fake_predict(calls):
    def predict(dataset_id, config, in_dir, out_dir, folds, gpu):
        img = nib.load(str(in_dir / "case_0000.nii.gz"))
        data = np.asarray(img.dataobj)
        out_dir.mkdir(parents=True)
        calls.append((dataset_id, config, folds, gpu, float(data.max()), float(data[7, 7, 0])))
        if dataset_id == 908:
            lab = (data > 300).astype(np.uint8)                      # the brain; a speck outside
            lab[0, 0, 0] = 1
        else:
            lab = np.zeros(data.shape, np.uint8)
            lab[10:20, 10:30, :] = 1                                 # white matter left
            lab[20:30, 10:30, :] = 2                                 # white matter right
            lab[14:16, 14:16, 3:5] = 5                               # thalamus left
        nib.save(nib.Nifti1Image(lab, img.affine), str(out_dir / "case.nii.gz"))
    return predict


def test_the_chain_strips_the_skull_binds_a_box_and_refuses_an_existing_out(tmp_path):
    p = _stack(tmp_path)
    calls = []
    rec = I.run(tmp_path / "out", 2, nifti=p, box=(14, 14, 3, 16, 16, 5), predict=_fake_predict(calls))
    assert [c[:4] for c in calls] == [(908, "2d", [0], 2), (907, "3d_fullres", [0], 2)]
    assert calls[1][5] == 0.0 and calls[0][5] == 200.0                 # the student saw the head set to zero
    anatomy = np.asarray(nib.load(rec["anatomy"]).dataobj)
    assert anatomy.dtype == np.int16 and set(np.unique(anatomy)) == {0, 2, 41, 10}
    mask = np.asarray(nib.load(rec["brain_mask"]).dataobj)
    assert mask[0, 0, 0] == 0 and mask[15, 15, 2] == 1                 # speck removed, brain kept
    assert rec["class_volumes_ml"]["thalamus_left"] == pytest.approx(8 * 2.0 * 2.0 * 5 / 1000, abs=1e-3)
    assert rec["binding"]["host"] == "thalamus" and rec["binding"]["host_rule"] == "overlap" and rec["box"] == [14, 14, 3, 16, 16, 5]
    assert rec["reliable_slices"] == [2, 6] and "NOT_EVIDENCE" in rec["anatomy_source"]
    assert json.loads((tmp_path / "out" / "record.json").read_text())["brain_ml"] == rec["brain_ml"]
    with pytest.raises(FileExistsError):
        I.run(tmp_path / "out", 2, nifti=p, predict=_fake_predict(calls))
    with pytest.raises(ValueError, match="exactly one"):
        I.run(tmp_path / "out2", 2, predict=_fake_predict(calls))
