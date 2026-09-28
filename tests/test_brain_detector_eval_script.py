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
    """Return synthetic registry rows for 4 lesions: 3 detected, 1 missed.

    Hand-computable fixture:
    - case_les_1 fold 1: lesion 0 (score 0.91, found), lesion 1 (score 0.31, found),
                         lesion 3 (score 0.2, NEVER FOUND - outside any predicted box)
    - case_les_2 fold 0: lesion 2 (score 0.71, found)
    - case_norm fold 2: no GT lesions, 2 FPs (scores 0.81, 0.39)

    At threshold 0.30 (operating point):
    - n_gt = 4 (lesions 0, 1, 2, 3)
    - n_hit = 3 (lesion 3 missed: score 0.2 < 0.30 everywhere, no component detected)
    - sensitivity_family = 3/4 = 0.75
    - n_fp = 2, fp_per_scan = 2/3 ≈ 0.6667
    - gate.pass = True (0.75 >= 0.5 minimum)
    - gate.thr = 0.30 (best achievable with these 4 lesions)
    - normal_fp_per_scan = 2/1 = 2.0 (2 FPs in 1 normal case)
    - Stratum of lesion 3 (e.g., band=">4"): n_gt=1, n_hit=0 (shows missed detection)
    """
    return [
        {"lesion_id": 0, "file": "case_les_1", "band": "0", "n_slices": 1, "inplane_mm": 2.5, "stratum_geometry": "rect", "z0": 10, "z1": 10, "x0": 5, "y0": 5, "x1": 10, "y1": 10},
        {"lesion_id": 1, "file": "case_les_1", "band": "0-2", "n_slices": 2, "inplane_mm": 3.5, "stratum_geometry": "rect", "z0": 15, "z1": 16, "x0": 20, "y0": 20, "x1": 25, "y1": 25},
        {"lesion_id": 2, "file": "case_les_2", "band": "2-4", "n_slices": 1, "inplane_mm": 4.5, "stratum_geometry": "3d", "z0": 8, "z1": 8, "x0": 30, "y0": 30, "x1": 35, "y1": 35},
        {"lesion_id": 3, "file": "case_les_1", "band": ">4", "n_slices": 1, "inplane_mm": 5.5, "stratum_geometry": "ellipse", "z0": 5, "z1": 5, "x0": 0, "y0": 0, "x1": 2, "y1": 2},
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

    # Create synthetic validation outputs with hand-computed scores
    # Shape: (cols, rows, slices) = (64, 64, 20)
    shape = (64, 64, 20)
    for fold, split in enumerate(splits):
        fold_dir = results / DATASET_NAME / "nnUNetTrainer_250epochs__nnUNetPlans__2d" / f"fold_{fold}" / "validation"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for case in split["val"]:
            lab = np.zeros(shape, np.uint8)
            probs = np.zeros((2, shape[0], shape[1], shape[2]), np.float32)

            if case == "case_les_2" and fold == 0:
                # Fold 0, val: case_les_2 with lesion 2 (score 0.71)
                lab[30:35, 30:35, 8] = 1
                probs[1][30:35, 30:35, 8] = 0.71

            elif case == "case_les_1" and fold == 1:
                # Fold 1, val: case_les_1 with lesion 0 (score 0.91) and lesion 1 (score 0.31)
                lab[5:10, 5:10, 10] = 1
                probs[1][5:10, 5:10, 10] = 0.91
                lab[20:25, 20:25, 15:17] = 1
                probs[1][20:25, 20:25, 15:17] = 0.31

            elif case == "case_norm_1" and fold == 2:
                # Fold 2, val: case_norm_1 (normal, no GT) with 2 FPs (scores 0.81, 0.39)
                lab[40:45, 40:45, 5] = 1
                probs[1][40:45, 40:45, 5] = 0.81
                lab[50:55, 50:55, 5] = 1
                probs[1][50:55, 50:55, 5] = 0.39

            # Write label map
            img = nib.Nifti1Image(lab, np.eye(4))
            nib.save(img, str(fold_dir / f"{case}.nii.gz"))

            # Save in nnU-Net format (C, Z, Y, X) as stored by nnU-Net
            probs_nnunet = np.ascontiguousarray(probs.transpose(0, 3, 2, 1)).astype(np.float32)
            np.savez_compressed(fold_dir / f"{case}.npz", probabilities=probs_nnunet)

    # Load the script module
    mod = _load_script()

    # Monkeypatch the constants to match our test data
    with patch.object(mod, "EXPECTED_N_GT", 4), \
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

        # Verify gate JSON and check hand-computed values
        report_text = (out / "REPORT.md").read_text()
        assert "Gate" in report_text
        # Extract JSON from report (it's in a code block)
        json_start = report_text.find("```json\n") + 8
        json_end = report_text.find("\n```", json_start)
        gate_json_text = report_text[json_start:json_end]
        gate = json.loads(gate_json_text)

        # At threshold 0.30 (operating point):
        # - n_gt=4 (3 found + 1 missed), n_hit=3, sensitivity_family = 3/4 = 0.75
        # - n_fp=2, fp_per_scan = 2/3 ≈ 0.6667
        # - gate.pass = True (0.75 >= 0.5 minimum)
        # - normal_fp_per_scan = 2/1 = 2.0
        assert gate["pass"] is True, "Gate should pass with sensitivity 0.75 >= 0.5"
        assert gate["thr"] == 0.30, f"Operating threshold should be 0.30, got {gate['thr']}"
        assert abs(gate["sensitivity_family"] - 0.75) < 0.01, f"Sensitivity should be 0.75 (3/4), got {gate['sensitivity_family']}"
        assert abs(gate["fp_per_scan"] - 0.6667) < 0.01, f"FP per scan should be ~0.6667, got {gate['fp_per_scan']}"

        # Verify FROC CSV has correct header and data
        froc_text = (out / "froc.csv").read_text()
        lines = froc_text.strip().split("\n")
        assert lines[0] == "thr,sensitivity,sensitivity_family,fp_per_scan"
        # At threshold 0.30: sensitivity_family = 0.75, fp_per_scan = 0.6667
        froc_row_030 = next((l for l in lines if l.startswith("0.30,")), None)
        assert froc_row_030 is not None, "FROC table should have threshold 0.30"
        assert "0.750000" in froc_row_030, f"Sensitivity at 0.30 should be 0.75 (3/4): {froc_row_030}"

        # Verify normal-volume false positives: must see exactly "2.0000" in output
        output_text = (out / "output.txt").read_text()
        assert "Normal FP per volume: 2.0000" in output_text, \
            f"Must show 2.0000 normal FP per volume, got:\n{output_text}"

        # Check REPORT.md contains strata tables with expected counts
        # band="0": n_gt=1, n_hit=1 (lesion 0)
        # band="0-2": n_gt=1, n_hit=1 (lesion 1)
        # band="2-4": n_gt=1, n_hit=1 (lesion 2)
        # band=">4": n_gt=1, n_hit=0 (lesion 3 MISSED - proves test catches missed lesions)
        # n_slices=1: n_gt=3, n_hit=3 (lesions 0, 2, 3; note 3 is missed so 2 hits)
        assert "| 0 | 1 | 1 |" in report_text, "Band 0 should have n_gt=1, n_hit=1"
        assert "| 0-2 | 1 | 1 |" in report_text, "Band 0-2 should have n_gt=1, n_hit=1"
        assert "| 2-4 | 1 | 1 |" in report_text, "Band 2-4 should have n_gt=1, n_hit=1"
        # Missed lesion (lesion 3, band >4, single-slice): n_gt=1, n_hit=0
        assert "| >4 | 1 | 0 |" in report_text, "Band >4 should have n_gt=1, n_hit=0 (missed lesion)"

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
