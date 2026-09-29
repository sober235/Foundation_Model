# Header-only: for every case a cross run will use (fold 0 validation cases of the data's split), do the mapped
# channel files exist and share the grid of the label the overlap is computed against?
import importlib.util
import json
import numpy as np
import nibabel as nib
from anatobind.nnunet.brain_disease import DISEASES, FM, channel_path

spec = importlib.util.spec_from_file_location("crossrun", "scripts/brain_disease_crossrun.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
NN = FM / "derived/nnunet"
for (model, data), channels in c.CROSS.items():
    host = DISEASES[data]["name"]
    cases = json.loads((NN / "preprocessed" / host / "splits_final.json").read_text())[0]["val"]
    missing, off = [], []
    worst = 0.0
    for case in cases:
        lab = nib.load(str(NN / "raw" / host / "labelsTr" / f"{case}.nii.gz"))
        for ch in channels:
            p = channel_path(data, case, ch, FM)
            if not p.is_file():
                missing.append((case, ch))
                continue
            im = nib.load(str(p))
            dev = float(np.abs(im.affine - lab.affine).max()) if im.shape == lab.shape else float("inf")
            worst = max(worst, dev)
            if dev > 1e-3:
                off.append((case, ch, im.shape, lab.shape, dev))
    print(f"{model} model on {data} data: {len(cases)} cases x {len(channels)} channels {channels}; missing {len(missing)}; "
          f"off the label's grid {len(off)}; largest affine deviation {worst:.3e}")
    for x in missing[:5] + off[:5]:
        print("   ", x)
