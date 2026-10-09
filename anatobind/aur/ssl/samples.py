"""Stage I manifests (SSL-first plan §2; decision Q6 of 2026-10-09).

The Stage I training set is every sequence volume of the four sources' train patients (the rows of the 1 mm sample
table whose split is train, with or without lesion labels: Stage I reads no label) plus HCP T1w / T2w (1113 healthy
adults, 0.7 mm, brain-extracted, SynthSeg maps present; resampled to 1 mm by anatobind.aur.resample before use). A
tenth of the train patients of every source, drawn once with a seed, is the validation set: it takes no gradient in
Stage I and is the same set Stage II validates on. No test patient appears in any Stage I file. The leakage report
counts what must be zero; the inventory lists source x sequence x split with the native spacing, for the records."""
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import nibabel as nib

from anatobind.anatomy.sources import FM

HCP_ROOT = "HCP_lh_T1T2"
HCP_SEG = "derived/synthseg/hcp/seg_native"
HCP_SEQUENCES = {"T1w": "T1", "T2w": "T2"}
HCP_FILE = "{seq}_acpc_dc_restore_brain.nii"
HCP_SEG_FILE = "{subject}_T1w_T1w_acpc_dc_restore_brain_seg.nii.gz"
VAL_SHARE = 0.1
HCP_EXPOSURE_CAP = 0.25          # the trainer's ceiling on HCP's share of the Stage I crop exposures (decision Q6)


def hcp_samples(root=FM):
    """(rows of HCP T1w / T2w with the SynthSeg map, subjects skipped for a missing file). One row per sequence; the
    anatomy map is the T1w SynthSeg output on the shared acpc grid; no lesions, so U and R are never supervised."""
    root = Path(root)
    rows, skipped = [], []
    for subject_dir in sorted((root / HCP_ROOT).iterdir()):
        subject = subject_dir.name
        files = {seq: subject_dir / "T1w" / HCP_FILE.format(seq=seq) for seq in HCP_SEQUENCES}
        seg = root / HCP_SEG / HCP_SEG_FILE.format(subject=subject)
        if not all(p.is_file() for p in files.values()) or not seg.is_file():
            skipped.append(subject)
            continue
        spacing = [float(v) for v in nib.load(str(files["T1w"])).header.get_zooms()[:3]]
        for seq, seq_type in HCP_SEQUENCES.items():
            rows.append({"case": f"HCP_{subject}", "source": "hcp", "patient": subject, "sequence": seq_type, "source_sequence": seq,
                         "image": str(files[seq]), "anatomy": str(seg), "lesion": None, "u_supervised": False, "u_values": [],
                         "a_ignore_values": [], "a_supervised": True, "r_supervised": False, "split": "train",
                         "native_spacing": spacing, "resampled": False, "thick_slice": False})
    return rows, skipped


def validation_patients(rows, share=VAL_SHARE, seed=0):
    """{source: sorted patients}: per source, ceil(share x train patients) patients drawn with the seed."""
    out = {}
    for source in sorted({r["source"] for r in rows}):
        patients = sorted({r["patient"] for r in rows if r["source"] == source and r["split"] == "train"})
        rng = random.Random(f"{seed}:{source}:val")
        rng.shuffle(patients)
        out[source] = sorted(patients[:math.ceil(share * len(patients))])
    return out


def ssl_rows(aur_rows, hcp_rows, val):
    """The Stage I rows: train rows of the sample table plus HCP, each with ssl_split train / val; test rows are left
    out entirely."""
    out = []
    for r in list(aur_rows) + list(hcp_rows):
        if r["split"] != "train":
            continue
        out.append({**r, "ssl_split": "val" if r["patient"] in set(val.get(r["source"], ())) else "train"})
    return out


def leakage_report(rows, aur_rows):
    """What must be zero: rows of a test patient, train rows of a validation patient, duplicate image paths; plus the
    counts and the patients that appear under two sources (a name clash to look at, not necessarily a leak)."""
    test_patients = {(r["source"], r["patient"]) for r in aur_rows if r["split"] == "test"}
    val_patients = {(r["source"], r["patient"]) for r in rows if r["ssl_split"] == "val"}
    images = [r["image"] for r in rows]
    by_patient = defaultdict(set)
    for r in rows:
        by_patient[r["patient"]].add(r["source"])
    report = {"rows": len(rows),
              "train_rows": sum(r["ssl_split"] == "train" for r in rows),
              "val_rows": sum(r["ssl_split"] == "val" for r in rows),
              "test_patients": len({p for _, p in test_patients}),
              "test_patient_rows_in_ssl": sum((r["source"], r["patient"]) in test_patients for r in rows),
              "val_patient_rows_in_ssl_train": sum((r["source"], r["patient"]) in val_patients and r["ssl_split"] == "train" for r in rows),
              "duplicate_images": len(images) - len(set(images)),
              "patients_under_two_sources": sorted(p for p, s in by_patient.items() if len(s) > 1),
              "per_source": {}}
    for source in sorted({r["source"] for r in rows}):
        sel = [r for r in rows if r["source"] == source]
        report["per_source"][source] = {split: {"rows": sum(r["ssl_split"] == split for r in sel),
                                                "patients": len({r["patient"] for r in sel if r["ssl_split"] == split})} for split in ("train", "val")}
    report["ok"] = report["test_patient_rows_in_ssl"] == 0 and report["val_patient_rows_in_ssl_train"] == 0 and report["duplicate_images"] == 0
    return report


def _spacing_key(r):
    spacing = r.get("native_spacing")
    if spacing is None:
        spacing = nib.load(r["anatomy"]).header.get_zooms()[:3]
    return "x".join(str(round(float(s), 3)) for s in spacing)


def inventory(rows):
    """[{source, sequence, ssl_split, spacing, n_rows, n_patients, thick_rows, resampled_rows}] sorted."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["sequence"], r["ssl_split"], _spacing_key(r))].append(r)
    out = []
    for (source, sequence, split, spacing), sel in sorted(groups.items()):
        out.append({"source": source, "sequence": sequence, "ssl_split": split, "spacing": spacing, "n_rows": len(sel),
                    "n_patients": len({r["patient"] for r in sel}), "thick_rows": sum(bool(r.get("thick_slice")) for r in sel),
                    "resampled_rows": sum(bool(r.get("resampled")) for r in sel)})
    return out


def write_manifest(out_dir, rows, val, report, inv):
    """Writes samples_ssl.json, val_patients.json, split_leakage_report.json and data_inventory.csv into a directory
    that must not exist yet. Returns {name: path}."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; nothing is overwritten here, use a new name")
    out_dir.mkdir(parents=True)
    paths = {"samples_ssl": out_dir / "samples_ssl.json", "val_patients": out_dir / "val_patients.json",
             "split_leakage_report": out_dir / "split_leakage_report.json", "data_inventory": out_dir / "data_inventory.csv"}
    paths["samples_ssl"].write_text(json.dumps(rows, indent=1))
    paths["val_patients"].write_text(json.dumps(val, indent=1, sort_keys=True))
    paths["split_leakage_report"].write_text(json.dumps(report, indent=1))
    with open(paths["data_inventory"], "w", newline="") as f:
        fields = ["source", "sequence", "ssl_split", "spacing", "n_rows", "n_patients", "thick_rows", "resampled_rows"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in inv:
            w.writerow(r)
    return {k: str(v) for k, v in paths.items()}
