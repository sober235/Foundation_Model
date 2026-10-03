#!/usr/bin/env python
# scripts/brain_anatomy_prepare.py
"""Build the data of the S4 anatomy model (spec 2026-10-02 §3, §5, §6; decisions A2-A10). Nothing is ever rebuilt in
place: every stage refuses an output that exists, and no stage deletes or renames a file.

  PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python scripts/brain_anatomy_prepare.py --stage sources
  ... --stage simulate --workers 8      K simulated stacks per training case, one per test case -> <work>/sim/{train,test}
  ... --stage dataset907                nnU-Net raw Dataset907_BrainAnatomyFLAIR (symlinks into <work>/sim)
  ... --stage dataset908                fastMRI RSS stacks + outline labels -> Dataset908_FastMRIBrainOutline
  ... --stage splits                    splits_final.json of both datasets (five folds by patient; fold 0 is trained)
"""
import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.anatomy import outline as OUT  # noqa: E402
from anatobind.anatomy.labels import LABELS_JSON, to_student  # noqa: E402
from anatobind.anatomy.simulate import simulate, stack_affine  # noqa: E402
from anatobind.anatomy.sources import all_cases, split_by_patient  # noqa: E402
from anatobind.data_engine.fastmri import rss_h5_to_nifti  # noqa: E402
from anatobind.data_engine.fastmri_knee import volume_geometry  # noqa: E402
from anatobind.infer.knee import NNUNET_ROOT  # noqa: E402
from anatobind.nnunet.brain_disease import FM, binary_label  # noqa: E402
from anatobind.nnunet.brain_lesion import assign_normal_folds, make_splits  # noqa: E402

WORK = FM / "derived/brain_anatomy"
K_TRAIN = 4
DATASET907 = "Dataset907_BrainAnatomyFLAIR"
DATASET908 = "Dataset908_FastMRIBrainOutline"
FASTMRI_SEG = FM / "derived/synthseg/fastmri_brain/seg_native"
KROOT = FM / "fastMRI_lh_brain_knee/kspace/brain"
TEST_SHARE = 0.2
GRID_TOL = 1e-3


def load_1mm_ras(path, order):
    """A NIfTI as a (x, y, z) RAS array at 1 mm isotropic spacing (resampled when the file is not), with its zooms."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if not np.allclose(zooms, 1.0, atol=1e-3):
        data = ndimage.zoom(data.astype(np.float32 if order else data.dtype), zooms, order=order, mode="constant", cval=0.0, grid_mode=False)
    return (data.astype(np.float32) if order else data), zooms


def check_one_grid(path, ref_path, tol=GRID_TOL):
    """Refuse a file that is not on the reference file's grid: same shape, affines within tol (read from the headers)."""
    a, b = nib.load(str(path)), nib.load(str(ref_path))
    if a.shape != b.shape or float(np.abs(a.affine - b.affine).max()) > tol:
        raise ValueError(f"{path} is not on the grid of {ref_path} (shape {a.shape} against {b.shape}, or the affines differ)")


def case_arrays(rec):
    """(flair, student labels, lesion mask or None) at 1 mm RAS for one case record. The teacher map and the lesion
    mask must lie on the FLAIR's grid; labels are never carried between grids."""
    check_one_grid(rec["anatomy"], rec["flair"])
    flair, _ = load_1mm_ras(rec["flair"], 1)
    seg, _ = load_1mm_ras(rec["anatomy"], 0)
    lesion = None
    if rec.get("lesion"):
        check_one_grid(rec["lesion"], rec["flair"])
        raw, _ = load_1mm_ras(rec["lesion"], 0)
        lesion = binary_label(raw, tuple(rec["lesion_values"]), str(rec["lesion"]))
    return flair, to_student(seg), lesion


def sample_seed(case, k):
    return int(hashlib.sha256(f"{case}:{k}".encode()).hexdigest()[:8], 16)


def simulate_case(job):
    """job = (case, record with str paths, out_dir, ks). Writes <case>_s<k>_0000.nii.gz, <case>_s<k>.nii.gz and
    <case>_s<k>.json; returns one manifest row per sample."""
    case, rec, out_dir, ks = job
    out_dir = Path(out_dir)
    flair, student, lesion = case_arrays(rec)
    rows = []
    for k in ks:
        stack, labels, params = simulate(flair, student, lesion, np.random.default_rng(sample_seed(case, k)))
        aff = stack_affine(params["inplane_mm"], stack.shape)
        name = f"{case}_s{k}"
        nib.save(nib.Nifti1Image(stack, aff), str(out_dir / f"{name}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(labels, aff), str(out_dir / f"{name}.nii.gz"))
        row = {"sample": name, "case": case, "source": rec["source"], "patient": rec["patient"], "shape": list(stack.shape),
               "ignore_voxels": int((labels == LABELS_JSON["ignore"]).sum()),
               **{key: (list(v) if isinstance(v, tuple) else v) for key, v in params.items()}}
        (out_dir / f"{name}.json").write_text(json.dumps(row, indent=1))
        rows.append(row)
    return rows


def _refuse(path):
    if Path(path).exists():
        raise FileExistsError(f"{path} exists; outputs are never rebuilt in place")


def stage_sources(work):
    work = Path(work)
    _refuse(work / "cases.json")
    cases = all_cases()
    split = split_by_patient(cases, TEST_SHARE, seed=0)
    work.mkdir(parents=True, exist_ok=True)
    out = {c: {**{k: (str(v) if isinstance(v, Path) else (list(v) if isinstance(v, tuple) else v)) for k, v in r.items()}, "split": split[c]}
           for c, r in cases.items()}
    (work / "cases.json").write_text(json.dumps(out, indent=1))
    for source in sorted({r["source"] for r in cases.values()}):
        n = {s: sum(1 for c, r in cases.items() if r["source"] == source and split[c] == s) for s in ("train", "test")}
        p = {s: len({r["patient"] for c, r in cases.items() if r["source"] == source and split[c] == s}) for s in ("train", "test")}
        print(f"{source}: train {n['train']} cases / {p['train']} patients, test {n['test']} cases / {p['test']} patients")
    print(f"wrote {work / 'cases.json'}: {len(cases)} cases")


def stage_simulate(work, workers, k_train=K_TRAIN):
    work = Path(work)
    cases = json.loads((work / "cases.json").read_text())
    _refuse(work / "sim")
    (work / "sim" / "train").mkdir(parents=True)
    (work / "sim" / "test").mkdir()
    jobs = [(c, r, work / "sim" / r["split"], list(range(k_train)) if r["split"] == "train" else [0]) for c, r in sorted(cases.items())]
    rows = []
    with Pool(workers, initializer=os.nice, initargs=(19,)) as pool:
        for i, part in enumerate(pool.imap_unordered(simulate_case, jobs), start=1):
            rows.extend(part)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} cases simulated", flush=True)
    rows.sort(key=lambda r: r["sample"])
    (work / "sim" / "manifest.json").write_text(json.dumps(rows, indent=1))
    n_train = sum(1 for r in rows if cases[r["case"]]["split"] == "train")
    print(f"wrote {work / 'sim'}: {n_train} training samples, {len(rows) - n_train} test samples")


def stage_dataset907(raw_root, work):
    work, base = Path(work), Path(raw_root) / DATASET907
    _refuse(base)
    cases = json.loads((work / "cases.json").read_text())
    rows = json.loads((work / "sim" / "manifest.json").read_text())
    (base / "imagesTr").mkdir(parents=True)
    (base / "labelsTr").mkdir()
    (base / "imagesTs").mkdir()
    info, n_tr = {}, 0
    for r in rows:
        split = cases[r["case"]]["split"]
        src = work / "sim" / split
        if split == "train":
            os.symlink((src / f"{r['sample']}_0000.nii.gz").resolve(), base / "imagesTr" / f"{r['sample']}_0000.nii.gz")
            os.symlink((src / f"{r['sample']}.nii.gz").resolve(), base / "labelsTr" / f"{r['sample']}.nii.gz")
            n_tr += 1
        else:
            os.symlink((src / f"{r['sample']}_0000.nii.gz").resolve(), base / "imagesTs" / f"{r['sample']}_0000.nii.gz")
        info[r["sample"]] = {"case": r["case"], "patient": r["patient"], "source": r["source"], "split": split}
    meta = {"channel_names": {"0": "FLAIR"}, "labels": LABELS_JSON, "numTraining": n_tr, "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    (base / "cases.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {base}: {n_tr} training samples, {len(info) - n_tr} test samples")
    return base


def h5_of(stem):
    for split in ("multicoil_train", "multicoil_val"):
        p = KROOT / split / f"{stem}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def fastmri_stems(seg_dir=FASTMRI_SEG):
    return sorted(p.name[:-len("_seg.nii.gz")] for p in Path(seg_dir).glob("file_brain_AXFLAIR_*_seg.nii.gz"))


def outline_from_seg(seg_path):
    """(label, info, SynthSeg image) of one stack: the outline rule decides usability before anything is written."""
    seg = nib.load(str(seg_path))
    z = tuple(float(v) for v in seg.header.get_zooms()[:3])
    label, info = OUT.outline_label(np.asarray(seg.dataobj), z[0] * z[1], z[0] * z[1] * z[2])
    return label, info, seg


def write_outline_case(stem, image_path, label, seg, base, split):
    """Place one usable stack's label (training stacks only). image_path is the RSS NIfTI already written into the
    dataset folder; it must lie on the grid of the SynthSeg map the label was made from (shape and affine), else the
    stage stops: a label is never carried between grids."""
    img = nib.load(str(image_path))
    if img.shape != seg.shape or float(np.abs(img.affine - seg.affine).max()) > GRID_TOL:
        raise ValueError(f"{stem}: the RSS image is not on the grid of its SynthSeg map (shape {img.shape} against {seg.shape}, or the "
                         f"affines differ); the half-built dataset stays in {base} and is not reused")
    if split == "train":
        nib.save(nib.Nifti1Image(label, img.affine), str(Path(base) / "labelsTr" / f"{stem}.nii.gz"))


def stage_dataset908(raw_root, work, seg_dir=FASTMRI_SEG, convert=None, patient=None):
    """convert(stem, out_path) writes the RSS NIfTI (default: rss_h5_to_nifti of the fastMRI h5, pad_to_slices=0);
    patient: {stem: patient id} (default: the h5 header's patient_id)."""
    base = Path(raw_root) / DATASET908
    _refuse(base)
    stems = fastmri_stems(seg_dir)
    convert = convert or (lambda stem, out: rss_h5_to_nifti(h5_of(stem), out, pad_to_slices=0))
    patient = patient or {s: volume_geometry(h5_of(s))["patient_id"] for s in stems}
    return build_dataset908(base, stems, patient, seg_dir, convert)


def build_dataset908(base, stems, patient, seg_dir, convert):
    base = Path(base)
    (base / "imagesTr").mkdir(parents=True)
    (base / "labelsTr").mkdir()
    (base / "imagesTs").mkdir()
    split = split_by_patient({s: {"source": "fastmri", "patient": patient[s]} for s in stems}, TEST_SHARE, seed=0)
    info, excluded = {}, {}
    for i, s in enumerate(stems, start=1):
        label, res, seg = outline_from_seg(Path(seg_dir) / f"{s}_seg.nii.gz")
        if res["usable"]:
            folder = "imagesTr" if split[s] == "train" else "imagesTs"
            img_path = base / folder / f"{s}_0000.nii.gz"
            convert(s, img_path)
            write_outline_case(s, img_path, label, seg, base, split[s])
            info[s] = {"patient": patient[s], "split": split[s], **res}
        else:
            excluded[s] = {**res, "patient": patient[s]}
        if i % 50 == 0 or i == len(stems):
            print(f"{i}/{len(stems)} stacks", flush=True)
    n_tr = sum(1 for v in info.values() if v["split"] == "train")
    meta = {"channel_names": {"0": "FLAIR"}, "labels": OUT.LABELS_JSON, "numTraining": n_tr, "file_ending": ".nii.gz"}
    (base / "dataset.json").write_text(json.dumps(meta, indent=1))
    (base / "cases.json").write_text(json.dumps(info, indent=1))
    (base / "excluded.json").write_text(json.dumps(excluded, indent=1))
    print(f"wrote {base}: {n_tr} training stacks, {len(info) - n_tr} test stacks, {len(excluded)} excluded")
    return base


def patient_splits(cases_info, k=5, seed=0):
    """{case: fold} with every case of a patient in one fold; cases_info: {case: {'patient': ...}}."""
    fold_of = assign_normal_folds([v["patient"] for v in cases_info.values()], k, seed)
    return {c: fold_of[v["patient"]] for c, v in cases_info.items()}


def write_splits(preprocessed_root, dataset, case_fold, k=5):
    d = Path(preprocessed_root) / dataset
    if not d.is_dir():
        raise FileNotFoundError(f"{d} missing: run nnUNetv2_plan_and_preprocess first")
    p = d / "splits_final.json"
    _refuse(p)
    splits = make_splits(case_fold, k)
    p.write_text(json.dumps(splits, indent=1))
    return p, [len(s["val"]) for s in splits]


def stage_splits(raw_root, preprocessed_root):
    todo = []
    for dataset in (DATASET907, DATASET908):                 # every check first: no split file is written unless both can be
        d = Path(preprocessed_root) / dataset
        if not d.is_dir():
            raise FileNotFoundError(f"{d} missing: run nnUNetv2_plan_and_preprocess first")
        _refuse(d / "splits_final.json")
        info = json.loads((Path(raw_root) / dataset / "cases.json").read_text())
        todo.append((dataset, patient_splits({c: v for c, v in info.items() if v["split"] == "train"})))
    for dataset, case_fold in todo:
        p, sizes = write_splits(preprocessed_root, dataset, case_fold)
        print(f"wrote {p}: validation cases per fold {sizes}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the S4 anatomy datasets")
    ap.add_argument("--stage", choices=("sources", "simulate", "dataset907", "dataset908", "splits"), required=True)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.stage == "sources":
        stage_sources(a.work)
    elif a.stage == "simulate":
        stage_simulate(a.work, a.workers)
    elif a.stage == "dataset907":
        stage_dataset907(NNUNET_ROOT / "raw", a.work)
    elif a.stage == "dataset908":
        stage_dataset908(NNUNET_ROOT / "raw", a.work)
    else:
        stage_splits(NNUNET_ROOT / "raw", NNUNET_ROOT / "preprocessed")


if __name__ == "__main__":
    main()
