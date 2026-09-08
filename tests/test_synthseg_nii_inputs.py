"""The runner must also accept NIfTI inputs (PDGM, HCP, BMSR, ISLES) without h5 staging."""
import nibabel as nib
import numpy as np

from anatobind.data_engine.synthseg_pipeline import run_batch, stage_niftis, stem_of


def _nii(path, shape=(6, 8, 3), zoom=(1.0, 1.0, 1.0)):
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.diag(list(zoom) + [1.0])), str(path))
    return path


def test_stem_of_strips_nii_and_nii_gz():
    assert stem_of("a/b/sub-01_T1w.nii.gz") == "sub-01_T1w"
    assert stem_of("a/b/T1w_acpc_dc_restore_brain.nii") == "T1w_acpc_dc_restore_brain"
    assert stem_of("file_brain_AXT2_200_2000019.h5") == "file_brain_AXT2_200_2000019"


def test_stage_niftis_symlinks_nifti_inputs_keeping_extension(tmp_path):
    src = _nii(tmp_path / "hcp_T1w.nii")
    out = stage_niftis([src], tmp_path / "stage")
    assert out[0] == tmp_path / "stage" / "hcp_T1w.nii"
    assert out[0].is_symlink() and out[0].resolve() == src.resolve()


def test_run_batch_handles_plain_nii_outputs(tmp_path):
    src = _nii(tmp_path / "hcp_T1w.nii", zoom=(0.7, 0.7, 0.7))

    def runner(argv):
        out_dir = argv[argv.index("--o") + 1]
        seg = nib.Nifti1Image(np.full((4, 6, 2), 3, dtype=np.int16), np.diag([1.0, 1.0, 1.0, 1.0]))
        nib.save(seg, f"{out_dir}/hcp_T1w_synthseg.nii")  # SynthSeg keeps the input extension

    rows = run_batch([src], tmp_path / "work", synthseg_home="/s", python="/p", threads=1, runner=runner)
    assert rows[0]["status"] == "ok"
    native = nib.load(rows[0]["seg_native"])
    assert native.shape == (6, 8, 3)  # back on the 0.7 mm native grid
