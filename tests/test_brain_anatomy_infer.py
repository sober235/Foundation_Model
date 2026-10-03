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
    with pytest.raises(FileExistsError, match="use a new output directory"):
        I.run(tmp_path / "out", 2, nifti=p, predict=_fake_predict(calls))
    with pytest.raises(ValueError, match="exactly one"):
        I.run(tmp_path / "out2", 2, predict=_fake_predict(calls))


def test_nothing_is_created_for_a_missing_input_and_a_failed_stage_names_itself(tmp_path):
    p = _stack(tmp_path)
    with pytest.raises(FileNotFoundError, match="is missing"):
        I.run(tmp_path / "a", 0, nifti=tmp_path / "nope.nii.gz", predict=_fake_predict([]))
    with pytest.raises(FileNotFoundError):
        I.run(tmp_path / "a", 0, h5=tmp_path / "nope.h5", predict=_fake_predict([]))
    assert not (tmp_path / "a").exists()                            # nothing staged: the corrected rerun may use the same name

    def broken(dataset_id, config, in_dir, out_dir, folds, gpu):
        if dataset_id == 907:
            raise OSError("no GPU")
        _fake_predict([])(dataset_id, config, in_dir, out_dir, folds, gpu)

    with pytest.raises(RuntimeError, match="the student prediction failed; the partial output stays in .*rerun into a new output directory"):
        I.run(tmp_path / "b", 0, nifti=p, predict=broken)
    assert (tmp_path / "b" / "brain_mask.nii.gz").exists() and not (tmp_path / "b" / "anatomy.nii.gz").exists()
    with pytest.raises(RuntimeError, match="checking the box failed"):
        I.run(tmp_path / "c", 0, nifti=p, box=(5, 5, 0, 5, 9, 2), predict=_fake_predict([]))        # empty along x
    assert I.check_box((0, 0, 0, 40, 40, 8), (40, 40, 8)) == (0, 0, 0, 40, 40, 8)
    with pytest.raises(ValueError, match="outside the grid"):
        I.check_box((0, 0, 0, 41, 40, 8), (40, 40, 8))


def test_the_nnunet_command_line_and_the_float32_staging(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(I.subprocess, "run", lambda cmd, check, env: seen.update(cmd=cmd, check=check, gpu=env["CUDA_VISIBLE_DEVICES"]))
    I.run_nnunet(908, "2d", tmp_path / "in", tmp_path / "o", [0], 3)
    assert seen["cmd"] == ["nice", "-n", "19", "nnUNetv2_predict", "-i", str(tmp_path / "in"), "-o", str(tmp_path / "o"), "-d", "908", "-c", "2d",
                           "-tr", "nnUNetTrainer_250epochs", "-f", "0", "-npp", "2", "-nps", "2", "--disable_progress_bar"]
    assert seen["check"] is True and seen["gpu"] == "3"
    ints = tmp_path / "int16.nii.gz"
    nib.save(nib.Nifti1Image(np.full((4, 4, 4), 7, np.int16), np.eye(4)), str(ints))
    img = I.stage_input(None, ints, tmp_path / "staged")
    assert img.get_data_dtype() == np.float32 and float(np.asarray(img.dataobj).max()) == 7.0
