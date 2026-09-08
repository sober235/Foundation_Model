"""Render box / host-mask / host_label overlays for human verification.

Step 5 of docs/superpowers/plans/2026-09-07-m1-arm-b-minimal-path.md: the gate
that no automatic test can pass.  Everything is drawn from the **model frame**
(Z, Y, X) -- the arrays the network actually sees -- so the figure verifies the
axis permutation, not just the export.

    PYTHONNOUSERSITE=1 PYTHONPATH=. ~/anaconda3/envs/nvgen/bin/python \
        scripts/render_binding_overlay.py --scans MTR_005,MTR_010
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from anatobind.train.dataset import SEG_NAMES, UNKNOWN_HOST, to_model_frame

ROOT = Path("/data2/congcong/data/FM_data/derived/skmtea/m1")


def _panel(ax, image, title, seg=None, host=None):
    ax.imshow(image, cmap="gray", origin="lower",
              vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    if seg is not None:
        ax.imshow(np.ma.masked_where(seg == 0, seg), cmap="tab10", origin="lower",
                  vmin=0, vmax=9, alpha=0.45, interpolation="nearest")
    if host:
        m = (seg == host).astype(float) if seg is not None else None
        if m is not None and m.any():
            ax.contour(m, levels=[0.5], colors="cyan", linewidths=1.2)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def render_scan(scan_id, root, out_dir):
    import csv

    d = root / scan_id
    img = nib.load(d / "image_clean_e1.nii.gz")
    image = to_model_frame(img.get_fdata(dtype=np.float32))
    seg = to_model_frame(np.asanyarray(nib.load(d / "seg.nii.gz").dataobj).astype(np.int16))
    written = []

    with open(d / "boxes.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["layer"] == "in_seg"]

    for r in rows:
        z0, y0, x0 = int(r["z0"]), int(r["y0"]), int(r["x0"])
        z1, y1, x1 = int(r["z1"]), int(r["y1"]), int(r["x1"])
        host = int(r["host_label"]) if r["host_label"].strip() else UNKNOWN_HOST
        host_name = SEG_NAMES.get(host, "UNRESOLVED (unknown truth)")
        z = (z0 + z1) // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
        _panel(axes[0], image[z], f"{scan_id}  slice z={z}  (model frame Z,Y,X)")
        _panel(axes[1], image[z], f"+ segmentation, host {host} = {host_name}",
               seg=seg[z], host=host)

        pad = 24
        yl, yh = max(y0 - pad, 0), min(y1 + pad, image.shape[1])
        xl, xh = max(x0 - pad, 0), min(x1 + pad, image.shape[2])
        _panel(axes[2], image[z, yl:yh, xl:xh], "zoom on the box",
               seg=seg[z, yl:yh, xl:xh], host=host)
        axes[2].add_patch(plt.Rectangle((x0 - xl, y0 - yl), x1 - x0, y1 - y0,
                                        fill=False, edgecolor="red", linewidth=1.8))

        for ax in axes[:2]:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="red", linewidth=1.6))

        overlap = float((seg[z0:z1, y0:y1, x0:x1] == host).mean())
        fig.suptitle(
            f"{scan_id} · ann {r['ann_id']} · {r['supercategory']} · tissue_id {r['tissue_id']}"
            f" → host_label {host} ({host_name}) · side {r['host_side']}"
            f" · host_ratio {r['host_ratio']} · host voxels in box {100 * overlap:.1f}%",
            fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = out_dir / f"{scan_id}_ann{r['ann_id']}_host{host}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", required=True, help="comma-separated scan ids")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=Path.home() / "figs" / "anatobind_m1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for scan_id in a.scans.split(","):
        for p in render_scan(scan_id.strip(), a.root, a.out):
            print(p, flush=True)


if __name__ == "__main__":
    main()
