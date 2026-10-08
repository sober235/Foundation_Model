# Post-hoc diagnosis of the evaluation (2026-10-04): what the SynthSeg reference and the student each hold on the 433
# real stacks, inside the reliable slices and by slice index. Reads the evaluation work directory and the SynthSeg maps;
# prints the table and writes one row per stack as JSON to the path given as the second argument (must not exist).
#
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python \
#       docs/verification/2026-10-02/brain_anatomy_flair/checks/eval_ref_vs_student.py <work> <rows.json>
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from anatobind.anatomy.labels import NAMES, to_student  # noqa: E402
from anatobind.eval.brain_anatomy import reliable_slices  # noqa: E402

W, ROWS = Path(sys.argv[1]), Path(sys.argv[2])
if ROWS.exists():
    raise FileExistsError(f"{ROWS} exists")
SEG = Path("/data2/congcong/data/FM_data/derived/synthseg/fastmri_brain/seg_native")
stems = sorted(p.name[:-7] for p in (W / "pred_student").glob("*.nii.gz"))
G = ["white_matter", "cortex", "thalamus", "basal_ganglia", "brainstem", "cerebellum", "other_deep_grey", "ventricles"]
group = {c: next(g for g in G if NAMES[c].startswith(g)) for c in range(1, 15)}
vol = {k: {g: 0.0 for g in G} for k in ("ref", "student")}
has = {k: {g: 0 for g in G} for k in ("ref", "student")}
by_slice = {k: {g: np.zeros(16) for g in ("thalamus", "basal_ganglia", "ventricles")} for k in ("ref", "student")}
rows = []
for stem in stems:
    pi = nib.load(str(W / "pred_student" / f"{stem}.nii.gz"))
    pred = np.asarray(pi.dataobj)
    ref_raw = np.asarray(nib.load(str(SEG / f"{stem}_seg.nii.gz")).dataobj)
    ref = to_student(ref_raw)
    z = pi.header.get_zooms()
    rel = list(reliable_slices(ref_raw, float(z[0] * z[1])))
    ml = float(np.prod(z[:3])) / 1000.0
    row = {"stem": stem, "reliable": [rel[0], rel[-1]] if rel else None}
    for k, m in (("ref", ref), ("student", pred)):
        sub = m[:, :, rel]
        for g in G:
            ids = [c for c in range(1, 15) if group[c] == g]
            v = float(np.isin(sub, ids).sum()) * ml
            vol[k][g] += v
            has[k][g] += v > 0
            row[f"{k}_{g}_ml"] = round(v, 2)
        for g in by_slice[k]:
            ids = [c for c in range(1, 15) if group[c] == g]
            for s in range(min(16, m.shape[2])):
                by_slice[k][g][s] += float(np.isin(m[:, :, s], ids).sum()) * ml
    rows.append(row)
n = len(stems)
print(f"{n} stacks; volumes inside the reliable slices, mean mL per stack; stacks that hold the group at all")
print(f"{'group':<16} {'ref mL':>8} {'student mL':>10} {'ref has':>8} {'student has':>11}")
for g in G:
    print(f"{g:<16} {vol['ref'][g] / n:8.2f} {vol['student'][g] / n:10.2f} {has['ref'][g]:8d} {has['student'][g]:11d}")
print("\nmean mL per stack by slice index (0 = lowest), all slices")
for g in by_slice["ref"]:
    print(g)
    print("  ref    ", " ".join(f"{v / n:5.2f}" for v in by_slice["ref"][g]))
    print("  student", " ".join(f"{v / n:5.2f}" for v in by_slice["student"][g]))
first = [r["reliable"][0] for r in rows if r["reliable"]]
print("\nfirst reliable slice: counts", {int(k): int(v) for k, v in zip(*np.unique(first, return_counts=True))})
ROWS.write_text(json.dumps(rows, indent=1))
print(f"wrote {ROWS}")
