"""Plumbing for the SynthSeg pseudo-label pipeline over fastMRI brain volumes."""
from pathlib import Path

import pytest

from anatobind.data_engine.synthseg_pipeline import (
    annotated_files,
    resolve_h5,
    synthseg_command,
)


def test_annotated_files_are_unique_sorted_stems_from_annotation_csv(tmp_path):
    csv = tmp_path / "brain.csv"
    csv.write_text(
        "file,slice,study_level,x,y,width,height,label\n"
        "file_brain_AXFLAIR_200_6002425,2,No,91,132,7,6,Nonspecific white matter lesion\n"
        "file_brain_AXFLAIR_200_6002425,5,No,210,170,7,7,Nonspecific white matter lesion\n"
        "file_brain_AXT2_200_2000019,3,No,1,1,2,2,Mass\n"
    )
    assert annotated_files(csv) == ["file_brain_AXFLAIR_200_6002425", "file_brain_AXT2_200_2000019"]


def test_resolve_h5_searches_train_then_val(tmp_path):
    (tmp_path / "multicoil_train").mkdir()
    (tmp_path / "multicoil_val").mkdir()
    (tmp_path / "multicoil_val" / "file_brain_AXT2_200_2000019.h5").touch()
    p = resolve_h5("file_brain_AXT2_200_2000019", tmp_path)
    assert p == tmp_path / "multicoil_val" / "file_brain_AXT2_200_2000019.h5"


def test_resolve_h5_raises_when_missing(tmp_path):
    (tmp_path / "multicoil_train").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_h5("file_brain_missing", tmp_path)


def test_synthseg_command_uses_robust_model_and_folders():
    argv = synthseg_command(
        in_dir=Path("/in"),
        out_dir=Path("/out"),
        resample_dir=Path("/res"),
        vol_csv=Path("/out/vol.csv"),
        threads=8,
        synthseg_home=Path("/home/x/src/SynthSeg"),
        python=Path("/home/x/anaconda3/envs/synthseg/bin/python"),
    )
    assert argv[0] == "/home/x/anaconda3/envs/synthseg/bin/python"
    assert argv[1] == "/home/x/src/SynthSeg/scripts/commands/SynthSeg_predict.py"
    assert "--robust" in argv
    assert argv[argv.index("--i") + 1] == "/in"
    assert argv[argv.index("--o") + 1] == "/out"
    assert argv[argv.index("--resample") + 1] == "/res"
    assert argv[argv.index("--vol") + 1] == "/out/vol.csv"
    assert argv[argv.index("--threads") + 1] == "8"
    assert "--cpu" not in argv


def test_synthseg_command_cpu_flag():
    argv = synthseg_command(
        in_dir=Path("/in"), out_dir=Path("/out"), resample_dir=None, vol_csv=None, threads=4, cpu=True,
        synthseg_home=Path("/s"), python=Path("/p"),
    )
    assert "--cpu" in argv
    assert "--resample" not in argv and "--vol" not in argv
