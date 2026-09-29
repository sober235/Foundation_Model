# Read-only probe for S4: is the outline of SynthSeg's label map on fastMRI FLAIR usable as a skull-strip?
# Numbers over all annotated FLAIR volumes, and montages of a few volumes with the outline drawn on the slices.
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("/data2/congcong/data/FM_data/derived")
OUT = Path.home() / "figs/foundation_model/s4probe"
import csv
MAN = {r["stem"]: r for r in csv.DictReader(open(D / "synthseg/fastmri_brain/manifest.csv"))}
segs = sorted((D / "synthseg/fastmri_brain/seg_native").glob("file_brain_AXFLAIR_*_seg.nii.gz"))
rng = np.random.default_rng(0)
show = set(rng.choice(len(segs), 6, replace=False).tolist())
rows = []
for i, sp in enumerate(segs):
    stem = sp.name.replace("_seg.nii.gz", "")
    ip = Path(MAN[stem]["nii"]) if stem in MAN else Path("/nonexistent")
    if not ip.exists():
        rows.append((stem, None))
        continue
    seg_img, img = nib.load(str(sp)), nib.load(str(ip))
    seg, a = np.asarray(seg_img.dataobj), np.asarray(img.dataobj, dtype=np.float32)
    if seg.shape != a.shape:
        rows.append((stem, "shape"))
        continue
    zooms = img.header.get_zooms()[:3]
    vox = float(np.prod(zooms))
    mask = seg > 0
    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    largest = float(sizes.max() / max(1, mask.sum()))
    thr = float(np.percentile(a[a > 0], 20)) if (a > 0).any() else 0.0           # the head against the background air
    head = ndimage.binary_fill_holes(a > thr)
    inside = float((mask & head).sum() / max(1, mask.sum()))                      # brain outline inside the head
    share = float(mask.sum() / max(1, head.sum()))                                # brain as a share of the head
    shell = head & ~ndimage.binary_erosion(head, iterations=6)                    # the outer 6 voxels of the head: scalp, skull
    in_shell = float((mask & shell).sum() / max(1, mask.sum()))                   # brain outline reaching into scalp
    per_slice = [float(mask[:, :, k].sum() * zooms[0] * zooms[1] / 100.0) for k in range(mask.shape[2])]   # cm2
    rows.append((stem, dict(ml=mask.sum() * vox / 1000.0, largest=largest, inside=inside, share=share, in_shell=in_shell,
                            n_slices=mask.shape[2], empty_slices=sum(1 for v in per_slice if v < 1.0), zooms=zooms)))
    if i in show:
        ks = np.linspace(0, mask.shape[2] - 1, 8).round().astype(int)
        fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
        for axx, k in zip(ax.ravel(), ks):
            sl = a[:, :, k].T
            axx.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=np.percentile(a, 99.5))
            axx.contour(mask[:, :, k].T.astype(float), levels=[0.5], colors="#ffb000", linewidths=1.0)
            axx.set_title(f"slice {k}", fontsize=11)
            axx.axis("off")
        fig.suptitle(f"{stem}: outline of SynthSeg's label map (labels > 0) on the FLAIR; "
                     f"{mask.sum() * vox / 1000.0:.0f} mL, {zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.1f} mm", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"skullstrip_{stem}.png", dpi=110)
        plt.close(fig)
        print("figure", OUT / f"skullstrip_{stem}.png", flush=True)
ok = [r[1] for r in rows if isinstance(r[1], dict)]
print(f"{len(segs)} FLAIR label maps; image missing {sum(1 for r in rows if r[1] is None)}; shape differs {sum(1 for r in rows if r[1] == 'shape')}; measured {len(ok)}")
for key, name in (("ml", "volume inside the outline (mL)"), ("largest", "share of the outline in its largest component"),
                  ("inside", "share of the outline inside the head"), ("share", "outline as a share of the head"),
                  ("in_shell", "share of the outline in the outer 6 voxels of the head"), ("empty_slices", "slices with under 1 cm2 of outline")):
    v = np.array([o[key] for o in ok], float)
    print(f"  {name}: min {v.min():.3f}, 5th percentile {np.percentile(v, 5):.3f}, median {np.median(v):.3f}, "
          f"95th percentile {np.percentile(v, 95):.3f}, max {v.max():.3f}")
worst = sorted(((o["in_shell"], r[0]) for r, o in zip([r for r in rows if isinstance(r[1], dict)], ok)), reverse=True)[:5]
print("  most outline in the outer shell:", [(round(a, 3), b) for a, b in worst])
small = sorted(((o["ml"], r[0]) for r, o in zip([r for r in rows if isinstance(r[1], dict)], ok)))[:5]
print("  smallest volumes:", [(round(a), b) for a, b in small])
