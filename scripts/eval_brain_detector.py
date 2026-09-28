#!/usr/bin/env python
# scripts/eval_brain_detector.py
"""Out-of-fold evaluation of the brain small-lesion detector (spec 2026-09-28 §4).

  source scripts/nnunet_env.sh
  PYTHONPATH=. python scripts/eval_brain_detector.py --config 2d --out /path/to/output
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.eval.brain_detector import collect, evaluate, normal_fp_per_scan, strata_sensitivity  # noqa: E402
from anatobind.nnunet.brain_lesion import DATASET_NAME, gt_boxes  # noqa: E402
from anatobind.level_r.registry import load_registry, BANDS  # noqa: E402

# Module-level constants for testability
EXPECTED_N_GT = 1297
EXPECTED_N_SCANS = 253


def main(argv=None):
    ap = argparse.ArgumentParser(description="Out-of-fold evaluation of the brain small-lesion detector")
    ap.add_argument("--config", choices=("2d", "3d_fullres"), required=True, help="nnU-Net configuration")
    ap.add_argument("--out", type=Path, required=True, help="Output directory (must not exist)")
    a = ap.parse_args(argv)

    # Verify output directory doesn't exist
    if a.out.exists():
        raise FileExistsError(f"--out {a.out} already exists")

    # Get paths from environment, validating they are set and non-empty
    raw_val = os.environ.get("nnUNet_raw", "").strip()
    preprocessed_val = os.environ.get("nnUNet_preprocessed", "").strip()
    results_val = os.environ.get("nnUNet_results", "").strip()

    if not raw_val:
        raise FileNotFoundError("nnUNet_raw environment variable not set or empty")
    if not preprocessed_val:
        raise FileNotFoundError("nnUNet_preprocessed environment variable not set or empty")
    if not results_val:
        raise FileNotFoundError("nnUNet_results environment variable not set or empty")

    raw_root = Path(raw_val)
    preprocessed_root = Path(preprocessed_val)
    results_root = Path(results_val)

    if not raw_root.is_dir():
        raise FileNotFoundError(f"nnUNet_raw {raw_root} is not a directory")
    if not preprocessed_root.is_dir():
        raise FileNotFoundError(f"nnUNet_preprocessed {preprocessed_root} is not a directory")
    if not results_root.is_dir():
        raise FileNotFoundError(f"nnUNet_results {results_root} is not a directory")

    # Load cases.json and splits_final.json
    cases_json = raw_root / DATASET_NAME / "cases.json"
    splits_json = preprocessed_root / DATASET_NAME / "splits_final.json"

    if not cases_json.exists():
        raise FileNotFoundError(f"{cases_json} not found")
    if not splits_json.exists():
        raise FileNotFoundError(f"{splits_json} not found")

    info = json.loads(cases_json.read_text())
    splits = json.loads(splits_json.read_text())

    # Load registry
    registry = load_registry()

    # Build registry_of_file: map file (h5 stem) to list of registry rows
    registry_of_file = {}
    for r in registry:
        registry_of_file.setdefault(r["file"], []).append(r)

    # Build gt_of_case: map case name to list of ground truth boxes
    gt_of_case = {}
    for case, case_info in info.items():
        if case_info["kind"] == "lesion":
            # Case name is the h5 stem (file without .h5 extension)
            file_stem = case
            if file_stem in registry_of_file:
                gt_of_case[case] = gt_boxes(registry_of_file[file_stem])
            else:
                gt_of_case[case] = []
        else:  # normal
            gt_of_case[case] = []

    # Identify normal cases
    normal_cases = {c for c, ci in info.items() if ci["kind"] == "normal"}

    # The splits' val lists must partition cases.json: every case exactly once.
    val_cases = [c for split in splits for c in split["val"]]
    seen = set()
    duplicated = set()
    for c in val_cases:
        if c in seen:
            duplicated.add(c)
        seen.add(c)
    missing = set(info) - seen
    if missing or duplicated:
        raise AssertionError(
            f"splits val lists must partition cases.json: missing {sorted(missing)}, duplicated {sorted(duplicated)}"
        )

    # Collect and evaluate
    scans = collect(results_root, a.config, splits, gt_of_case)

    result = evaluate(scans, normal_cases)

    # Assert on what was actually scored, not merely on the inputs.
    if result["n_scans"] != EXPECTED_N_SCANS:
        raise AssertionError(f"Expected {EXPECTED_N_SCANS} scans, got {result['n_scans']}")
    if result["n_gt"] != EXPECTED_N_GT:
        raise AssertionError(f"Expected {EXPECTED_N_GT} total lesions, got {result['n_gt']}")

    # Build strata_of
    # 1. band
    band_of = {}
    for r in registry:
        band_of[r["lesion_id"]] = r["band"]

    # 2. n_slices (1 vs >1)
    n_slices_of = {}
    for r in registry:
        n_slices_of[r["lesion_id"]] = "1" if r["n_slices"] == 1 else ">1"

    # 3. in-plane tertile of inplane_mm over the 1297
    inplane_values = sorted([r["inplane_mm"] for r in registry])
    t33 = inplane_values[len(inplane_values) // 3]
    t67 = inplane_values[2 * len(inplane_values) // 3]

    inplane_tertile_of = {}
    for r in registry:
        mm = r["inplane_mm"]
        if mm <= t33:
            inplane_tertile_of[r["lesion_id"]] = "tertile_1"
        elif mm <= t67:
            inplane_tertile_of[r["lesion_id"]] = "tertile_2"
        else:
            inplane_tertile_of[r["lesion_id"]] = "tertile_3"

    # 4. stratum_geometry
    geometry_of = {}
    for r in registry:
        geometry_of[r["lesion_id"]] = r["stratum_geometry"]

    # Get operating threshold
    thr = result["gate"]["thr"]

    # Create output directory
    a.out.mkdir(parents=True, exist_ok=False)

    # Write output files
    # 1. REPORT.md
    report_lines = []
    report_lines.append("# Brain Small-Lesion Detector: Out-of-Fold Evaluation\n")

    # Gate line as JSON
    report_lines.append("## Gate\n")
    report_lines.append("```json\n")
    report_lines.append(json.dumps(result["gate"], indent=2))
    report_lines.append("\n```\n\n")

    # FROC table from rows
    report_lines.append("## FROC Curve\n\n")
    report_lines.append("| Threshold | Sensitivity | Sensitivity (Family) | FP per Scan |\n")
    report_lines.append("|-----------|-------------|----------------------|-------------|\n")
    for row in result["rows"]:
        report_lines.append(f"| {row['thr']:.2f} | {row['sensitivity']:.4f} | {row['sensitivity_family']:.4f} | {row['fp_per_scan']:.4f} |\n")
    report_lines.append("\n")

    # Normal FP per volume
    report_lines.append("## Normal-Volume False Positives\n\n")
    if thr is not None and result["normal_fp_per_scan"] is not None:
        report_lines.append(f"At operating threshold {thr:.2f}: {result['normal_fp_per_scan']:.4f} FP per volume\n\n")
    else:
        report_lines.append("Gate did not pass; no operating threshold.\n\n")

    # Strata tables
    report_lines.append("## Strata\n\n")

    if thr is not None:
        # Band
        report_lines.append("### Distance Band\n\n")
        band_strata = strata_sensitivity(scans, thr, band_of)
        report_lines.append("| Band | N (GT) | N (Hit) | Sensitivity |\n")
        report_lines.append("|------|--------|--------|-------------|\n")
        for band in BANDS:
            if band in band_strata:
                s = band_strata[band]
                report_lines.append(f"| {band} | {s['n_gt']} | {s['n_hit']} | {s['sensitivity']:.4f} |\n")
        report_lines.append("\n")

        # n_slices
        report_lines.append("### Number of Slices\n\n")
        n_slices_strata = strata_sensitivity(scans, thr, n_slices_of)
        report_lines.append("| Type | N (GT) | N (Hit) | Sensitivity |\n")
        report_lines.append("|------|--------|--------|-------------|\n")
        for key in ["1", ">1"]:
            if key in n_slices_strata:
                s = n_slices_strata[key]
                report_lines.append(f"| {key} slice(s) | {s['n_gt']} | {s['n_hit']} | {s['sensitivity']:.4f} |\n")
        report_lines.append("\n")

        # In-plane tertile
        report_lines.append("### In-Plane Size Tertile\n\n")
        inplane_strata = strata_sensitivity(scans, thr, inplane_tertile_of)
        report_lines.append("| Tertile | N (GT) | N (Hit) | Sensitivity |\n")
        report_lines.append("|---------|--------|--------|-------------|\n")
        for tertile in ["tertile_1", "tertile_2", "tertile_3"]:
            if tertile in inplane_strata:
                s = inplane_strata[tertile]
                report_lines.append(f"| {tertile} | {s['n_gt']} | {s['n_hit']} | {s['sensitivity']:.4f} |\n")
        report_lines.append("\n")

        # Stratum geometry
        report_lines.append("### Stratum Geometry\n\n")
        geometry_strata = strata_sensitivity(scans, thr, geometry_of)
        report_lines.append("| Geometry | N (GT) | N (Hit) | Sensitivity |\n")
        report_lines.append("|----------|--------|--------|-------------|\n")
        for geom in sorted(geometry_strata.keys()):
            s = geometry_strata[geom]
            report_lines.append(f"| {geom} | {s['n_gt']} | {s['n_hit']} | {s['sensitivity']:.4f} |\n")
        report_lines.append("\n")

    # Command
    report_lines.append("## Command\n\n")
    report_lines.append("```\n")
    report_lines.append(" ".join(sys.argv))
    report_lines.append("\n```\n")

    (a.out / "REPORT.md").write_text("".join(report_lines))

    # 2. froc.csv
    froc_lines = ["thr,sensitivity,sensitivity_family,fp_per_scan\n"]
    for row in result["rows"]:
        froc_lines.append(f"{row['thr']:.2f},{row['sensitivity']:.6f},{row['sensitivity_family']:.6f},{row['fp_per_scan']:.6f}\n")
    (a.out / "froc.csv").write_text("".join(froc_lines))

    # 3. output.txt
    output_lines = []
    output_lines.append(f"Scans: {result['n_scans']}\n")
    output_lines.append(f"Total ground truth lesions: {result['n_gt']}\n")
    output_lines.append(f"Gate pass: {result['gate']['pass']}\n")
    if result["gate"]["thr"] is not None:
        output_lines.append(f"Operating threshold: {result['gate']['thr']:.2f}\n")
        output_lines.append(f"Family sensitivity at operating point: {result['gate']['sensitivity_family']:.4f}\n")
        output_lines.append(f"FP per scan at operating point: {result['gate']['fp_per_scan']:.4f}\n")
        if result["normal_fp_per_scan"] is not None:
            output_lines.append(f"Normal FP per volume: {result['normal_fp_per_scan']:.4f}\n")
    (a.out / "output.txt").write_text("".join(output_lines))

    print(f"Evaluation complete. Results written to {a.out}")


if __name__ == "__main__":
    main()
