# Read-only: the sentences the records would hold, built in memory from the out-of-fold predictions of one fold.
# Nothing is written. A sentence has to be read on real lesions before 1212 of them are written once.
#
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python \
#       docs/verification/2026-09-29/brain_multidisease/checks/sentence_preview.py <disease> <fold> <threshold> [n]
import json
import sys

from anatobind.eval.brain_disease import case_scan, jobs
from anatobind.infer.brain_disease import study_record
from anatobind.nnunet.brain_disease import DISEASES, FM, anatomy_path

disease, fold, thr = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NN = FM / "derived/nnunet"
name = DISEASES[disease]["name"]
info = json.loads((NN / "raw" / name / "cases.json").read_text())
splits = json.loads((NN / "preprocessed" / name / "splits_final.json").read_text())
todo = jobs(NN / "results", NN / "raw", disease, splits, [fold], info, lambda c: anatomy_path(disease, c, FM))
records = [study_record(s["case"], disease, thr, s["dets"], [fold]) for s in map(case_scan, todo)]

counts = sorted(len(r["lesions"]) for r in records)
places = [r["sentence"].split("（还见于")[1].split("）")[0].count("、") + 1 for r in records if "（还见于" in r["sentence"]]
print(f"{disease} fold {fold}, threshold {thr:.2f}: {len(records)} studies; lesions per study min {counts[0]}, "
      f"median {counts[len(counts) // 2]}, max {counts[-1]}; studies without a lesion {counts.count(0)}, with more than "
      f"five {sum(1 for c in counts if c > 5)}; places named after the count: "
      f"{'none' if not places else f'in {len(places)} studies, at most {max(places)}'}; longest sentence "
      f"{max(len(r['sentence']) for r in records)} characters")
by_count = sorted(records, key=lambda r: (-len(r["lesions"]), r["study"]))
one = [r for r in by_count if len(r["lesions"]) == 1]
few = [r for r in by_count if 2 <= len(r["lesions"]) <= 5]
none = [r for r in by_count if not r["lesions"]]
for title, group in (("most lesions", by_count), ("two to five lesions", few), ("one lesion", one), ("no lesion", none)):
    print(f"\n## {title}")
    for r in group[:n]:
        sizes = sorted((l["volume_mm3"] for l in r["lesions"]), reverse=True)
        print(f"- {r['study']} ({len(r['lesions'])} lesions{', largest ' + format(sizes[0], 'g') + ' mm3' if sizes else ''}): {r['sentence']}")
