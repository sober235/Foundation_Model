# Post-hoc diagnosis of the evaluation (2026-10-04): slices 0-5 of fastMRI stacks, stripped stack / student / SynthSeg.
# The evaluation's sided-host Dice came out at 0.27 while the host agreement and the outline passed; this montage shows
# the low part of the stack, where the deep structures lie, so that a human can see which of the two maps holds them
# (USER_REPORTED). Same drawing as eval_montage.py; the output folder must not exist.
#
#   PYTHONNOUSERSITE=1 PYTHONPATH=. nice -n 19 ~/anaconda3/envs/nvgen/bin/python \
#       docs/verification/2026-10-02/brain_anatomy_flair/checks/eval_low_slices.py <work> <seg_dir> <out_dir> [n_stacks]
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from anatobind.anatomy.labels import to_student  # noqa: E402
from anatobind.eval.brain_anatomy import reliable_slices  # noqa: E402

work, seg_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6
if out.exists():
    raise FileExistsError(f"{out} exists")
stems = sorted(p.name[:-len("_0000.nii.gz")] for p in (work / "stripped").glob("*_0000.nii.gz"))
step = max(1, len(stems) // n)
out.mkdir(parents=True)
for stem in stems[::step][:n]:
    img = nib.load(str(work / "stripped" / f"{stem}_0000.nii.gz"))
    data, student = np.asarray(img.dataobj), np.asarray(nib.load(str(work / "pred_student" / f"{stem}.nii.gz")).dataobj)
    seg_img = nib.load(str(seg_dir / f"{stem}_seg.nii.gz"))
    teacher = to_student(np.asarray(seg_img.dataobj))
    z = seg_img.header.get_zooms()
    rel = reliable_slices(np.asarray(seg_img.dataobj), float(z[0]) * float(z[1]))
    ks = list(range(min(6, data.shape[2])))
    fig, axes = plt.subplots(3, len(ks), figsize=(3.2 * len(ks), 9.5))
    for i, k in enumerate(ks):
        for row in range(3):
            axes[row, i].imshow(data[:, :, k].T, cmap="gray", vmin=0, vmax=np.percentile(data, 99.5), origin="lower")
            axes[row, i].axis("off")
        axes[0, i].set_title(f"slice {k}{' (reliable)' if k in rel else ''}")
        axes[1, i].imshow(np.ma.masked_where(student[:, :, k].T == 0, student[:, :, k].T), cmap="tab20", vmin=0, vmax=15, alpha=0.5, origin="lower")
        axes[2, i].imshow(np.ma.masked_where(teacher[:, :, k].T == 0, teacher[:, :, k].T), cmap="tab20", vmin=0, vmax=15, alpha=0.5, origin="lower")
    fig.suptitle(f"{stem}: stripped stack / student / SynthSeg, slices 0-5 (reliable slices {rel.start}-{rel.stop - 1 if len(rel) else 'none'})")
    fig.tight_layout()
    fig.savefig(out / f"low_{stem}.png", dpi=72)
    plt.close(fig)
    print(f"wrote {out}/low_{stem}.png")
