#!/usr/bin/env python
"""Test-set evaluation of AnatoBind-Brain (spec §7; SSL-first plan T12; gates G2, G3).

    PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python scripts/aur_eval.py \
        --checkpoint <STAGE_III_DIR>/aur_stage3_best.pt --samples <samples_1mm_v1.json> --out <NEW_DIR> --gpu <idle card>
        [--split test|val --val-patients <val_patients.json>] [--sources pdgm bmsr] [--limit N] [--no-reference]

A on every row of the split (Dice against SynthSeg), U on the rows with lesion labels (sensitivity at the reference
detector's false positives per scan on the same patients, S7's out-of-fold nnU-Net on the native grid), R on the
lesion instances (controlled track against B0 / B0*). Writes aggregate.json, cases.json, reference.json and REPORT.md
into a new directory. Every number is NOT_EVIDENCE (pseudo-labels)."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.aur import eval as E  # noqa: E402
from anatobind.aur import infer as I  # noqa: E402
from anatobind.aur.crops import CROP  # noqa: E402
from anatobind.aur.ssl.checkpoint import code_sha, file_sha256  # noqa: E402
from anatobind.aur.train import split_rows  # noqa: E402


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="a new directory")
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--val-patients", type=Path, default=None, help="needed for --split val")
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="at most this many rows per source (sorted by case, sequence)")
    ap.add_argument("--no-reference", action="store_true", help="skip the nnU-Net reference (U then has no gate)")
    ap.add_argument("--u-threshold", action="append", default=[], metavar="SOURCE=THR",
                    help="a score threshold chosen on the validation split, per source (else the threshold is selected on this set and labelled so)")
    ap.add_argument("--no-e2e", action="store_true", help="skip binding the predicted lesions (the end-to-end R track)")
    dev = ap.add_mutually_exclusive_group(required=True)
    dev.add_argument("--gpu", type=int, default=None, help="CUDA device index")
    dev.add_argument("--cpu", action="store_true")
    ap.add_argument("--crop", type=int, nargs=3, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--instance-threshold", type=float, default=E.INSTANCE_THRESHOLD)
    return ap


def select_rows(rows, split, val_patients, sources, limit):
    if split == "test":
        sel = [r for r in rows if r.get("split") == "test"]
    else:
        if val_patients is None:
            raise ValueError("--split val needs --val-patients")
        _, sel = split_rows(rows, json.loads(Path(val_patients).read_text()))
    if sources:
        sel = [r for r in sel if r["source"] in sources]
    sel = sorted(sel, key=lambda r: (r["source"], r["case"], r["sequence"]))
    if limit is not None:
        out, seen = [], {}
        for r in sel:
            if seen.get(r["source"], 0) < limit:
                out.append(r)
                seen[r["source"]] = seen.get(r["source"], 0) + 1
        sel = out
    return sel


def main(argv=None):
    a = parser().parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists; the output directory must be new")
    if a.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu given but CUDA is not available")
        device = torch.device("cuda", a.gpu)
    fixed = {}
    for item in a.u_threshold:
        source, thr = item.split("=")
        fixed[source] = float(thr)
    rows = select_rows(json.loads(a.samples.read_text()), a.split, a.val_patients, a.sources, a.limit)
    reference = {}
    if not a.no_reference:                                         # before the inference loop: a missing file fails now, not hours later
        for source in sorted({r["source"] for r in rows if r["source"] in E.REFERENCE}):
            cases = sorted({r["case"] for r in rows if r["source"] == source and r.get("u_supervised")})
            if not cases:
                continue
            scans = E.reference_scans(source, cases)
            reference[source] = E.reference_summary(source, scans)
            print(f"reference {source}: {reference[source]}", flush=True)
    model, meta = I.model_from_export(a.checkpoint, device)
    crop = tuple(a.crop) if a.crop else tuple(meta.get("config", {}).get("crop", CROP))
    results, t0 = [], time.time()
    for i, r in enumerate(rows):
        results.append(E.evaluate_case(model, r, crop, device, batch_size=a.batch_size, instance_threshold=a.instance_threshold, bind_predicted=not a.no_e2e))
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"[{i + 1}/{len(rows)}] {r['source']} {r['case']} {r['sequence']} elapsed {time.time() - t0:.0f} s", flush=True)
    agg = E.aggregate(results, reference, fixed_thresholds=fixed)
    agg["run"] = {"checkpoint": str(a.checkpoint), "checkpoint_sha256": file_sha256(a.checkpoint), "code_sha": code_sha(), "split": a.split,
                  "samples": str(a.samples), "samples_sha256": file_sha256(a.samples), "crop": list(crop), "n_rows": len(rows),
                  "sources": a.sources, "limit": a.limit, "instance_threshold": a.instance_threshold, "fixed_thresholds": fixed,
                  "stage": meta.get("stage"), "e2e": not a.no_e2e,
                  "seconds": round(time.time() - t0, 1)}
    E.write_report(a.out, agg, results)
    (a.out / "reference.json").write_text(json.dumps(reference, indent=1, default=str))
    print(E.markdown(agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
