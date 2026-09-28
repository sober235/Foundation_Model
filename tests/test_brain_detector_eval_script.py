import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_brain_detector.py"
    spec = importlib.util.spec_from_file_location("eval_brain_detector", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_registry():
    """Return synthetic registry rows for 3 lesions in 2 files."""
    return [
        {"lesion_id": 0, "file": "case_les_1", "band": "0", "n_slices": 1, "inplane_mm": 2.5, "stratum_geometry": "rect", "z0": 10, "z1": 10, "x0": 5, "y0": 5, "x1": 10, "y1": 10},
        {"lesion_id": 1, "file": "case_les_1", "band": "0-2", "n_slices": 2, "inplane_mm": 3.5, "stratum_geometry": "rect", "z0": 15, "z1": 16, "x0": 20, "y0": 20, "x1": 25, "y1": 25},
        {"lesion_id": 2, "file": "case_les_2", "band": "2-4", "n_slices": 1, "inplane_mm": 4.5, "stratum_geometry": "3d", "z0": 8, "z1": 8, "x0": 30, "y0": 30, "x1": 35, "y1": 35},
    ]


def test_script_end_to_end(tmp_path, monkeypatch):
    """End-to-end test of the brain detector evaluation script."""
    # Set up fake nnUNet directories
    raw = tmp_path / "raw"
    preprocessed = tmp_path / "preprocessed"
    results = tmp_path / "results"
    raw.mkdir()
    preprocessed.mkdir()
    results.mkdir()

    # Set environment variables
    monkeypatch.setenv("nnUNet_raw", str(raw))
    monkeypatch.setenv("nnUNet_preprocessed", str(preprocessed))
    monkeypatch.setenv("nnUNet_results", str(results))

    # Create dataset directories
    from anatobind.nnunet.brain_lesion import DATASET_NAME
    raw_ds = raw / DATASET_NAME
    preprocessed_ds = preprocessed / DATASET_NAME
    raw_ds.mkdir()
    preprocessed_ds.mkdir()

    # Create cases.json: 2 lesion cases + 1 normal case
    cases_info = {
        "case_les_1": {"patient_id": "p1", "kind": "lesion", "n_label_voxels": 100},
        "case_les_2": {"patient_id": "p2", "kind": "lesion", "n_label_voxels": 80},
        "case_norm_1": {"patient_id": "p3", "kind": "normal", "n_label_voxels": 0},
    }
    (raw_ds / "cases.json").write_text(json.dumps(cases_info))

    # Create splits_final.json
    splits = [
        {"train": ["case_les_1", "case_norm_1"], "val": ["case_les_2"]},
        {"train": ["case_les_2", "case_norm_1"], "val": ["case_les_1"]},
        {"train": ["case_les_1", "case_les_2"], "val": ["case_norm_1"]},
    ]
    (preprocessed_ds / "splits_final.json").write_text(json.dumps(splits))

    # Create synthetic validation outputs
    # Shape: (cols, rows, slices) = (64, 64, 20)
    shape = (64, 64, 20)
    for fold, split in enumerate(splits):
        fold_dir = results / DATASET_NAME / "nnUNetTrainer_250epochs__nnUNetPlans__2d" / f"fold_{fold}" / "validation"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for case in split["val"]:
            # Create label map (only for lesion cases; normal case has zeros)
            lab = np.zeros(shape, np.uint8)
            if case == "case_les_2" and fold == 0:
                # Paint lesion boxes from registry
                lab[30:35, 30:35, 8] = 1  # lesion_id 2
            elif case == "case_les_1" and fold == 1:
                # Paint lesion boxes from registry
                lab[5:10, 5:10, 10] = 1   # lesion_id 0
                lab[20:25, 20:25, 15:17] = 1  # lesion_id 1

            # Write label map
            img = nib.Nifti1Image(lab, np.eye(4))
            nib.save(img, str(fold_dir / f"{case}.nii.gz"))

            # Create probabilities: (C, X, Y, Z) shape for the export format
            # decode_boxes will transpose it to (C, X, Y, Z) internally
            probs = np.zeros((2, shape[0], shape[1], shape[2]), np.float32)
            probs[1] = (lab > 0).astype(np.float32)  # class 1 is lesion

            # Save in nnU-Net format (C, Z, Y, X) - transpose back for saving
            probs_nnunet = np.ascontiguousarray(probs.transpose(0, 3, 2, 1)).astype(np.float32)
            np.savez_compressed(fold_dir / f"{case}.npz", probabilities=probs_nnunet)

    # Load the script module
    mod = _load_script()

    # Monkeypatch the constants to match our test data
    with patch.object(mod, "EXPECTED_N_GT", 3), \
         patch.object(mod, "EXPECTED_N_SCANS", 3), \
         patch.object(mod, "load_registry", return_value=_synthetic_registry()):

        # Test that non-existent --out is created
        out = tmp_path / "output"
        mod.main(["--config", "2d", "--out", str(out)])
        assert out.exists()

        # Verify output files were created
        assert (out / "REPORT.md").exists()
        assert (out / "froc.csv").exists()
        assert (out / "output.txt").exists()

        # Verify gate JSON exists in REPORT.md
        report_text = (out / "REPORT.md").read_text()
        assert "Gate" in report_text
        # Extract JSON from report (it's in a code block)
        json_start = report_text.find("```json\n") + 8
        json_end = report_text.find("\n```", json_start)
        gate_json_text = report_text[json_start:json_end]
        gate = json.loads(gate_json_text)
        assert "sensitivity_family" in gate

        # Verify FROC CSV has correct header and data
        froc_text = (out / "froc.csv").read_text()
        lines = froc_text.strip().split("\n")
        assert lines[0] == "thr,sensitivity,sensitivity_family,fp_per_scan"

    # Test that existing --out is refused
    with pytest.raises(FileExistsError, match="already exists"):
        mod.main(["--config", "2d", "--out", str(out)])

    # Test that missing environment variable is caught
    monkeypatch.delenv("nnUNet_raw")
    with pytest.raises(FileNotFoundError, match="nnUNet_raw"):
        mod.main(["--config", "2d", "--out", str(tmp_path / "output2")])


def test_script_missing_env_variables(tmp_path, monkeypatch):
    """Test that missing environment variables are caught with clear messages."""
    mod = _load_script()

    # Clear all nnUNet env vars
    monkeypatch.delenv("nnUNet_raw", raising=False)
    monkeypatch.delenv("nnUNet_preprocessed", raising=False)
    monkeypatch.delenv("nnUNet_results", raising=False)

    # Test missing nnUNet_raw
    with pytest.raises(FileNotFoundError, match="nnUNet_raw"):
        mod.main(["--config", "2d", "--out", str(tmp_path / "out1")])

    # Set nnUNet_raw, test missing nnUNet_preprocessed
    monkeypatch.setenv("nnUNet_raw", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="nnUNet_preprocessed"):
        mod.main(["--config", "2d", "--out", str(tmp_path / "out2")])

    # Set nnUNet_preprocessed, test missing nnUNet_results
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="nnUNet_results"):
        mod.main(["--config", "2d", "--out", str(tmp_path / "out3")])
