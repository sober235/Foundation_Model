#!/usr/bin/env python
"""Turn nnU-Net's held-out validation segmentations into model-frame label maps.

  source scripts/nnunet_env.sh; PYTHONPATH=. python scripts/collect_nnunet_predictions.py
"""
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.nnunet.prepare import validation_path  # noqa: E402
from anatobind.train.cache import VIEWS  # noqa: E402
from anatobind.train.dataset import to_model_frame  # noqa: E402

FM = Path("/data2/congcong/data/FM_data/derived/skmtea")


def main():
    folds = json.loads((FM / "m1r" / "splits.json").read_text())["folds"]
    out = FM / "m1r_pred" / "nnunet"
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for scan, fold in sorted(folds.items()):
        for view in VIEWS:
            src = validation_path(os.environ["nnUNet_results"], fold, scan, view)
            if not src.exists():
                missing.append(str(src))
                continue
            lab = to_model_frame(np.asanyarray(nib.load(str(src)).dataobj).astype(np.uint8))
            np.savez_compressed(out / f"{scan}__{view}.npz", label_map=lab)
    print(f"{len(folds) * len(VIEWS) - len(missing)} label maps written, {len(missing)} missing")
    for m in missing[:10]:
        print("  missing", m)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
