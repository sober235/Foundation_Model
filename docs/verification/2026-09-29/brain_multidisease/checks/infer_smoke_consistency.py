# Read-only: does the inference entry (fold 0 model, nnUNetv2_predict on one case) give the same lesions as the
# out-of-fold record written by the five-fold evaluation for that case (fold 0's validation prediction, same threshold)?
# Boxes and scores must agree to 1e-4 (plan Task 12 Step 3); host fields must be equal; volumes and host fractions are
# reported, not gated. The entry names the study after its first image file, the evaluation after the case id, so the
# out-of-fold record is found by that prefix. Prints the two sentences as well. Writes nothing.
#
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python \
#       docs/verification/2026-09-29/brain_multidisease/checks/infer_smoke_consistency.py
import json
from pathlib import Path

BD = Path("/data2/congcong/data/FM_data/derived/brain_disease")
TOL = 1e-4
FIELDS = ("type", "host", "host_rule", "side", "host_side", "host_sides", "host_distance_mm")

all_ok = True
for disease in ("glioma", "metastasis", "infarct"):
    smoke = json.loads((BD / "infer_smoke" / disease / "record.json").read_text())
    matches = [p for p in (BD / disease / "records").glob("*.json") if smoke["study"].startswith(p.stem)]
    assert len(matches) == 1, (smoke["study"], matches)
    oof = json.loads(matches[0].read_text())
    print(f"## {disease}: smoke study {smoke['study']!r}, out-of-fold record {matches[0].name}")
    print(f"smoke: threshold {smoke['threshold']}, model_folds {smoke['model_folds']}, {len(smoke['lesions'])} lesions")
    print(f"out-of-fold record: threshold {oof['threshold']}, model_folds {oof['model_folds']}, {len(oof['lesions'])} lesions")
    ok = smoke["threshold"] == oof["threshold"] and smoke["model_folds"] == oof["model_folds"]
    ok &= len(smoke["lesions"]) == len(oof["lesions"])
    worst_box, worst_score, field_diffs, worst_vol, worst_frac = 0, 0.0, 0, 0.0, 0.0
    for a, b in zip(smoke["lesions"], oof["lesions"]):
        worst_box = max(worst_box, max(abs(x - y) for x, y in zip(a["box"], b["box"])))
        worst_score = max(worst_score, abs(a["score"] - b["score"]))
        field_diffs += sum(1 for f in FIELDS if a.get(f) != b.get(f))
        worst_vol = max(worst_vol, abs(a["volume_mm3"] - b["volume_mm3"]) / max(a["volume_mm3"], b["volume_mm3"]))
        keys = set(a["host_fractions"]) | set(b["host_fractions"])
        worst_frac = max(worst_frac, max(abs(a["host_fractions"].get(k, 0.0) - b["host_fractions"].get(k, 0.0)) for k in keys))
    ok &= worst_box <= TOL and worst_score <= TOL and field_diffs == 0
    print(f"largest box difference {worst_box} voxels, largest score difference {worst_score:.1e}, host fields that differ {field_diffs}")
    print(f"not gated: largest relative volume difference {worst_vol:.2e}, largest host-fraction difference {worst_frac:.1e}")
    print(f"sentence equal: {smoke['sentence'] == oof['sentence']}; impression equal: {smoke['impression'] == oof['impression']}")
    print(f"smoke sentence: {smoke['sentence']}")
    if smoke["sentence"] != oof["sentence"]:
        print(f"out-of-fold sentence: {oof['sentence']}")
    print(f"boxes and scores within {TOL}, host fields equal: {ok}\n")
    all_ok &= ok
print("ALL AGREE" if all_ok else "DISAGREEMENT")
