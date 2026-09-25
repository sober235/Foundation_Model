"""Per-family localisation at four score thresholds on the clean view, and what the detector predicts as effusion (REPORT §3.2 diagnostic)."""
import importlib.util, sys
import numpy as np
sys.path.insert(0, "/data0/congcong/code/Project_Doing/foundation_model-aur")
from anatobind.eval.detection_metrics import per_family
from anatobind.eval.fastmri_knee_detection import scan_record
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT
spec = importlib.util.spec_from_file_location("ev", "scripts/eval_fastmri_knee_detection.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
raw, spacing_of, patient_of, missing = ev.load_view(EXPORT_ROOT, EXPORT_ROOT / "detections", "clean")
scans = [scan_record(f, gt, dets, normal, None) for f, (gt, dets, normal) in raw.items()]
for thr in (0.01, 0.05, 0.10, 0.18):
    pf = per_family(scans, thr)
    print(thr, {k: f"{v['n_hit']}/{v['n_gt']} loc, {v['n_hit_family']} fam" for k, v in pf.items()})
fam = {}
top = 0.0
for f, (gt, dets, normal) in raw.items():
    for d in dets:
        top = max(top, d["score"])
        if d["score"] >= 0.18:
            fam[d["family"]] = fam.get(d["family"], 0) + 1
print("predicted family counts at thr 0.18:", fam, "| max score on the clean view:", round(top, 3))
