# Montages of simulated stacks for a human look (spec 2026-10-02 §5; USER_REPORTED): two samples per source, every
# second slice, image on top and student labels below (15 = ignore). Reads <work>/sim, writes PNGs into the folder
# given as the second argument (must not exist). Nothing else is written.
#
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python \
#       docs/verification/2026-10-02/brain_anatomy_flair/checks/sim_montage.py /data2/congcong/data/FM_data/derived/brain_anatomy \
#       docs/verification/2026-10-02/brain_anatomy_flair/simulation
import json
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

work, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    raise FileExistsError(f"{out} exists")
rows = json.loads((work / "sim" / "manifest.json").read_text())
cases = json.loads((work / "cases.json").read_text())
out.mkdir(parents=True)
for source in sorted({r["source"] for r in rows}):
    picked = [r for r in rows if r["source"] == source and r["sample"].endswith("_s0")][:2]
    for r in picked:
        d = work / "sim" / cases[r["case"]]["split"]
        stack = np.asarray(nib.load(str(d / f"{r['sample']}_0000.nii.gz")).dataobj)
        labels = np.asarray(nib.load(str(d / f"{r['sample']}.nii.gz")).dataobj)
        ks = list(range(0, stack.shape[2], 2))
        fig, axes = plt.subplots(2, len(ks), figsize=(3 * len(ks), 6.5))
        for i, k in enumerate(ks):
            for row in (0, 1):
                axes[row, i].imshow(stack[:, :, k].T, cmap="gray", vmin=0, vmax=1000, origin="lower")
                axes[row, i].axis("off")
            axes[0, i].set_title(f"slice {k}")
            axes[1, i].imshow(np.ma.masked_where(labels[:, :, k].T == 0, labels[:, :, k].T), cmap="tab20", vmin=0, vmax=15, alpha=0.5, origin="lower")
        fig.suptitle(f"{source} {r['sample']}: {r['n_slices']} slices, {r['inplane_mm']} mm, empty top {r['empty_top']}, tilt {r['theta_lr']:.1f}/{r['theta_ap']:.1f}; "
                     f"bottom: image, labels (15 = ignore)")
        fig.tight_layout()
        fig.savefig(out / f"sim_{source}_{r['sample']}.png", dpi=72)
        plt.close(fig)
        print(f"wrote {out}/sim_{source}_{r['sample']}.png")
