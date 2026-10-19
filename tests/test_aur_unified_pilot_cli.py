"""The public pilot entry point must consume TRAIN only and complete all three stages."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest
import torch
import nibabel as nib
import numpy as np
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/aur_unified_pilot.py"


def load_script():
    spec = importlib.util.spec_from_file_location("unified_pilot_cli_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(patient, *, source="pdgm", split="train", sequence="FLAIR"):
    return dict(patient=patient, source=source, split=split, sequence=sequence,
                case=patient, u_supervised=True, a_supervised=True, r_supervised=True)


def test_selection_excludes_validation_test_and_duplicate_patient_sequences():
    module = load_script()
    rows = [row("validation"), row("historical", split="test"), row("training"),
            row("training", sequence="T1c"), row("another", source="bmsr", sequence="T1c")]
    selected = module.select_training_rows(rows, {"pdgm": ["validation"]}, 3)
    assert {(r["source"], r["patient"]) for r in selected} == {("pdgm", "training"), ("bmsr", "another")}
    assert len(selected) == 2
    with pytest.raises(ValueError, match="patient_split_overlap"):
        module.select_training_rows(rows + [row("training", split="test")], {"pdgm": ["validation"]}, 3)


@pytest.mark.parametrize("split", ["val", "validation", "unknown", "development"])
def test_selection_never_treats_unlisted_nontrain_rows_as_training(split):
    module = load_script()
    selected = module.select_training_rows([row("allowed"), row("not_training", split=split)], {}, 3)
    assert [item["patient"] for item in selected] == ["allowed"]


def test_synthetic_views_have_true_plane_geometry_and_real_supervision():
    module = load_script()
    views = module.SyntheticViews(seed=3)
    for mode, depth in (("2d", 1), ("3d", 8)):
        batch = views(mode, 0)
        assert batch["image"].shape == (1, 1, depth, 32, 32)
        assert batch["input_mode"] == mode and batch["n_instances"] == [1]
        assert batch["host"][0].tolist() == [0] and batch["r_eligible"][0].tolist() == [True]
        assert batch["u_supervised"].all() and batch["a_supervised"].all()
        if mode == "2d":
            assert (batch["local"][:, 0] == 0).all()
        assert torch.equal(batch["image"], views(mode, 0)["image"])


def test_synthetic_cli_completes_single_checkpoint_three_stage_path(tmp_path):
    out = tmp_path / "private-pilot"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHONPATH": "."}
    run = subprocess.run([sys.executable, str(SCRIPT), "--synthetic", "--out", str(out),
                          "--steps-per-stage", "2", "--device", "cpu"],
                         cwd=SCRIPT.parents[1], env=env, text=True, capture_output=True, timeout=90)
    assert run.returncode == 0, run.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["pilot_only"] and summary["evidence"] == "NOT_EVIDENCE"
    assert not summary["formal_training_ready"] and not summary["historical_test_used"]
    assert set(summary["stages"]) == {"I", "II", "III"}
    for stage, result in summary["stages"].items():
        assert result["exposures"] == {"2d": 1, "3d": 1}
        assert len(result["steps"]) == 2
        assert all(torch.isfinite(torch.tensor(step["loss"])) for step in result["steps"])
        assert (out / f"stage_{stage}.pt").is_file()
    assert (out.stat().st_mode & 0o777) == 0o700
    assert (out / "stage_III.pt").stat().st_mode & 0o077 == 0
    assert "patient" not in run.stdout.lower()


def test_cli_never_overwrites_an_existing_output_directory(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep").write_text("original")
    run = subprocess.run([sys.executable, str(SCRIPT), "--synthetic", "--out", str(out),
                          "--steps-per-stage", "2"], cwd=SCRIPT.parents[1],
                         text=True, capture_output=True, timeout=30)
    assert run.returncode != 0
    assert "output_exists" in run.stderr
    assert (out / "keep").read_text() == "original"


def test_real_pilot_refuses_source_changes_even_when_views_were_cached(tmp_path, monkeypatch):
    module = load_script()
    monkeypatch.setattr(module, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(module, "PRIVATE_ROOT", tmp_path)
    rng = np.random.default_rng(1)
    shape = (40, 40, 8)
    arrays = {"image": rng.uniform(1, 100, shape).astype(np.float32),
              "anatomy": np.full(shape, 2, np.int16), "lesion": np.zeros(shape, np.uint8)}
    arrays["lesion"][8:12, 8:12, 2:5] = 1
    sample = {**row("generated"), "u_values": [1], "a_ignore_values": [1]}
    for key, array in arrays.items():
        path = tmp_path / f"generated_{key}.nii.gz"
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
        sample[key] = str(path)
    samples, validation = tmp_path / "samples.json", tmp_path / "split.json"
    samples.write_text(json.dumps([sample]))
    validation.write_text("{}")
    original_views = module.TrainingViews

    class ChangedSourceViews(original_views):
        def __call__(self, mode, draw):
            batch = super().__call__(mode, draw)
            if not getattr(self, "changed", False):
                with Path(self.rows[0]["image"]).open("ab") as stream:
                    stream.write(b"changed after loading")
                self.changed = True
            return batch

    monkeypatch.setattr(module, "TrainingViews", ChangedSourceViews)
    args = SimpleNamespace(out=tmp_path / "run", synthetic=False, samples=samples, val_patients=validation,
                           max_patients=1, seed=0, fixed_crops=True, steps_per_stage=2,
                           max_seconds=60., device="cpu")
    with pytest.raises(ValueError, match="medical_inputs_changed_during_pilot"):
        module.run_pipeline(args)
    assert not (args.out / "summary.json").exists()


def test_real_output_is_private_outside_git_before_any_medical_input_is_opened(tmp_path):
    module = load_script()
    for path in (SCRIPT.parents[1] / "unapproved_private_run", tmp_path / "outside_home"):
        with pytest.raises(ValueError, match="private_output_unsupported"):
            module.validate_private_output(path, synthetic=False)
        assert not path.exists()
    with pytest.raises(ValueError, match="private_output_unsupported"):
        module.validate_private_output(SCRIPT.parents[1] / "unapproved_synthetic_run", synthetic=True)


def test_argument_errors_do_not_echo_private_values(tmp_path):
    marker = "SYNTHETIC_PRIVATE_PARAMETER_MARKER"
    run = subprocess.run([sys.executable, str(SCRIPT), "--synthetic", "--out", str(tmp_path / "run"),
                          "--steps-per-stage", marker], cwd=SCRIPT.parents[1], text=True, capture_output=True, timeout=30)
    assert run.returncode != 0 and "invalid_arguments" in run.stderr
    assert marker not in run.stderr and marker not in run.stdout


def test_quoted_home_output_is_normalized_for_the_actual_writer(tmp_path, monkeypatch):
    module = load_script()
    leaf = f".aur_output_path_test_{uuid.uuid4().hex}"
    expected = Path.home() / leaf
    args = SimpleNamespace(out=Path("~/") / leaf, synthetic=False)
    monkeypatch.chdir(tmp_path)
    try:
        # Fail at missing input configuration immediately after directory creation; no images or model are loaded.
        with pytest.raises(AttributeError):
            module.run_pipeline(args)
        assert expected.is_dir()
        assert args.out == expected
        assert not (tmp_path / "~").exists()
    finally:
        if expected.is_dir():
            expected.rmdir()


def test_output_normalization_errors_never_echo_private_paths(tmp_path):
    marker = "SYNTHETIC_PRIVATE_SYMLINK_MARKER"
    path = tmp_path / marker
    path.symlink_to(path.name)
    run = subprocess.run([sys.executable, str(SCRIPT), "--synthetic", "--out", str(path)],
                         cwd=SCRIPT.parents[1], text=True, capture_output=True, timeout=30)
    assert run.returncode != 0
    assert marker not in run.stdout and marker not in run.stderr
    assert "Traceback" not in run.stderr and "pilot_failed" in run.stderr
